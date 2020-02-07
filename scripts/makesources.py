import db
import io
import os
import subprocess
import shlex
import time
from tqdm import tqdm
import sys
import shutil
import publish
import numpy as np
import traceback
import pandas as pd
import datetime
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from sklearn.cluster import DBSCAN

from scipy.sparse import csr_matrix

ASSOC_RB_MIN = 0.4

fmap = {1: 'zg',
        2: 'zr',
        3: 'zi'}


# association radius 2 arcsec

ASSOC_RADIUS_ARCSEC = 2
ASSOC_RADIUS = ASSOC_RADIUS_ARCSEC * 0.0002777

N_PREV_SINGLE = 1
N_PREV_MULTI = 1
DEFAULT_GROUP = 1
DEFAULT_INSTRUMENT = 1

def submit_thumbs(thumbids):

    ndt = datetime.datetime.utcnow()
    nightdate = f'{ndt.year}{ndt.month:02d}{ndt.day:02d}'
    curdir = os.getcwd()

    scriptname = Path(f'/global/cscratch1/sd/dgold/zuds/'
                      f'nightly/{nightdate}/{ndt}.thumb.sh'.replace(' ', '_'))
    scriptname.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(scriptname.parent)

    thumbinname = f'{scriptname}'.replace('.sh', '.in')
    with open(thumbinname, 'w') as f:
        f.write('\n'.join([str(t) for t in thumbids]) + '\n')

    jobscript = f"""#!/bin/bash
#SBATCH --image=registry.services.nersc.gov/dgold/ztf:latest
#SBATCH --volume="/global/homes/d/dgold/lensgrinder/pipeline/:/pipeline;/global/homes/d/dgold:/home/desi;/global/homes/d/dgold/skyportal:/skyportal"
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q realtime
#SBATCH --exclusive
#SBATCH -J zuds
#SBATCH -t 00:60:00
#SBATCH -L SCRATCH
#SBATCH -A ***REMOVED***

HDF5_USE_FILE_LOCKING=FALSE srun -n 64 -c1 --cpu_bind=cores shifter python $HOME/lensgrinder/scripts/dothumb.py {thumbinname}

"""

    with open(scriptname, 'w') as f:
        f.write(jobscript)

    cmd = f'sbatch {scriptname}'
    process = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = process.communicate()
    print(stdout)

    if process.returncode != 0:
        raise RuntimeError(
            f'Non-zero exit code from sbatch, output was '
            f'"{str(stdout)}", "{str(stdout)}".'
        )

    os.chdir(curdir)
    _ = stdout.strip().split()[-1].decode('ascii')


def source_bestdet_from_solution(df):
    source_bestdet = {}
    for sourceid, group in df.groupby('source'):
        source_bestdet[sourceid] = group['snr'].idxmax()
    return source_bestdet


def associate(debug=False):

    db.DBSession().execute('LOCK TABLE ONLY objectswithflux IN EXCLUSIVE MODE;')

    r = db.DBSession().execute(
        'update objectswithflux set source_id = s.id, '
        'modified = now() from  detections d join objectswithflux o '
        'on d.id = o.id join sources s on '
        'q3c_join(d.ra, d.dec, s.ra, s.dec,  0.0002777*2) '
        'where  o.source_id is NULL  and '
        'objectswithflux.id = d.id returning d.id, s.id'
    )
    r = list(r)
    triggers_alert = [row[0] for row in r]

    print(f'associated {len(r)} detections with existing sources')

    db.DBSession().execute('''
    update sources set ra=dummy.ra, dec=dummy.dec, modified=now()
    from (select g.id, g.ra, g.dec from
    (select s.id, d.ra, d.dec, rank() over
    (partition by o.source_id order by o.flux / o.fluxerr desc)
    from sources s join objectswithflux o on o.source_id = s.id
    join detections d on o.id = d.id ) g where rank = 1)
    dummy where sources.id = dummy.id;
    ''')


    q = f'''select d.id, d.ra, d.dec, o.flux / o.fluxerr as snr 
    from detections d join objectswithflux o on d.id = o.id join realbogus
    rb on rb.detection_id = d.id where o.source_id is NULL 
    and rb.rb_score > {ASSOC_RB_MIN} order by o.id asc'''

    if debug:
        q += ' LIMIT 10000'

    df = pd.DataFrame(
        list(db.DBSession().execute(q)),
        columns=['id', 'ra', 'dec', 'snr']
    )

    df = df.set_index('id')

    coord = SkyCoord(df['ra'], df['dec'], unit='deg')
    idx1, idx2, sep, _ = search_around_sky(coord, coord, ASSOC_RADIUS_ARCSEC * u.arcsec)
    dropdupes = idx1 != idx2
    idx1 = idx1[dropdupes]
    idx2 = idx2[dropdupes]
    sep = sep[dropdupes]

    # cluster the detections into sources using DBSCAN
    clustering = DBSCAN(
        eps=ASSOC_RADIUS_ARCSEC,
        min_samples=2,
        metric='precomputed'
    )

    # construct the sparse pairwise distance matrix
    nobj = len(df)
    darcsec = sep.to('arcsec').value
    distmat = csr_matrix((darcsec, (idx1, idx2)), shape=(nobj, nobj))
    clustering.fit(distmat)

    df['source'] = clustering.labels_
    df = df[df['source'] != -1]

    with db.DBSession().no_autoflush:
        default_group = db.DBSession().query(
            db.models.Group
        ).get(DEFAULT_GROUP)

        default_instrument = db.DBSession().query(
            db.models.Instrument
        ).get(DEFAULT_INSTRUMENT)

    bestdets = source_bestdet_from_solution(df)

    # cache d1 and d2
    # get thumbnail pics
    from sqlalchemy.orm import joinedload
    d1 = db.DBSession().query(db.Detection).filter(db.Detection.id.in_(
        [int(v) for v in bestdets.values()]
    )).options(joinedload(db.Detection.thumbnails)).all()

    detcache = {d.id: d for d in d1}
    sourceid_map = {}

    curval = db.DBSession().execute("select nextval('namenum')").first()[0]
    for sourceid in tqdm(bestdets):
        bestdet = detcache[bestdets[sourceid]]

        name = publish.get_next_name(num=curval)
        curval += 1
        source = db.models.Source(
            id=name,
            groups=[default_group],
            ra=bestdet.ra,
            dec=bestdet.dec
        )

        db.DBSession().execute('INSERT INTO sources (id, ra, dec, created_at, modified) VALUES '
                               f'({name}, {bestdet.ra}, {bestdet.dec}, now(), now())')
        db.DBSession().execute('INSERT INTO group_sources (source_id, group_id, created_at, modified) '
                               f'VALUES ({name}, 1, now(), now())')


        sourceid_map[sourceid] = source

        # need this to make stamps.
        dummy_phot = db.models.Photometry(
            source=source,
            instrument=default_instrument
        )

        pid = db.DBSession().exeucte('INSERT INTO photometry (source_id, instrument_id, created_at, modified) '
                                     f'VALUES ({name}, 1, now(), now())) RETURNING ID')[0][0]



        db.DBSession().execute(f'UPDATE thumbnails SET modified=now(), photometry_id={pid}, '
                               f'source_id={name} where detection_id={bestdet.id}')


        """
        bestdet.source = source
        db.DBSession().add(bestdet)
        db.DBSession().add(source)
        db.DBSession().add(dummy_phot)


        for t in bestdet.thumbnails:
            t.photometry = dummy_phot
            t.source = bestdet.source
            #t.persist()
            db.DBSession().add(t)
            
        """
        db.DBSession().execute(
            'insert into thumbnails (created_at, modified, photometry_id, public_url, type) '
            f"VALUES (now(), now(), pid, {source.sdss_url}, 'sdss')")
        db.DBSession().execute(
            'insert into thumbnails (created_at, modified, photometry_id, public_url, type) '
            f"VALUES (now(), now(), pid, {source.desi_dr8_url}, 'dr8')")

        """

        sdss_thumb = db.models.Thumbnail(photometry=dummy_phot,
                                  public_url=source.sdss_url,
                                  type='sdss')
        dr8_thumb = db.models.Thumbnail(photometry=dummy_phot,
                                 public_url=source.desi_dr8_url,
                                 type='dr8')
        db.DBSession().add_all([sdss_thumb, dr8_thumb])
        """

    db.DBSession().execute(f"select setval('namenum', {curval})")
    db.DBSession().flush()

    for sourceid, group in df.groupby('source'):
        realid = sourceid_map[sourceid].id
        dets = group.index.tolist()
        db.DBSession().execute(
            f'''
            update objectswithflux set source_id = '{realid}'
            where objectswithflux.id in {tuple(dets)}
            '''
        )

    print(f'triggering alerts and forced photometry for {len(detection_ids)} detections')
    db.DBSession().execute(
        f'''update detections set triggers_alert = 't'
        where detections.id in {tuple(triggers_alert)}'''
    )

    # need to commit so that sources will be there for forced photometry
    # jobs running via slurm
    db.DBSession().commit()

    thumbids = []
    for d in detcache.values():
        for t in d.thumbnails:
            thumbids.append(t.id)

    if os.getenv('NERSC_HOST') == 'cori':
        submit_thumbs(thumbids)


if __name__ == '__main__':
    db.DBSession().get_bind().echo=True
    associate(debug=True)
