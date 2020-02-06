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
from astropy.coordinates import SkyCoord, match_coordinates_sky

ASSOC_RB_MIN = 0.4

fmap = {1: 'zg',
        2: 'zr',
        3: 'zi'}


# association radius 2 arcsec
ASSOC_RADIUS = 2 * 0.0002777
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


def source_coords_from_solution(grouped, inverted, df):
    source_coords = {}
    for source in grouped:
        dets = grouped[source]
        snrs = df.loc[dets]['snr']
        best = snrs.idxmax()
        source_coords[source] = tuple(df.loc[best][['ra', 'dec']].tolist())
    return source_coords

    
def iterate_sources(grouped, inverted, df):

    source_coords = source_coords_from_solution(grouped, inverted, df)
    
    sources = np.asarray([s for s in source_coords])
    coords = SkyCoord([source_coords[s][0] for s in sources],
                      [source_coords[s][1] for s in sources],
                      unit='deg')

    idx, d2d, d3d = match_coordinates_sky(coords, coords, nthneighbor=2)

    ok = sources[d2d < 2 * u.arcsec]
    match = sources[idx[d2d < 2 * u.arcsec]]

    if len(match) == 0:
        return False

    for s1, s2 in zip(ok, match):
        s2dets = grouped[s2]
        grouped[s1].extend(s2dets)
        del grouped[s2]
        for d in s2dets:
            inverted[d] = s1

    return True

def associate(debug=False):

    """
    db.DBSession().execute('LOCK TABLE ONLY objectswithflux IN EXCLUSIVE MODE;')

    r = db.DBSession().execute(
        'update objectswithflux set source_id = s.id, '
        'modified = now() from  detections d join objectswithflux o '
        'on d.id = o.id join sources s on '
        'q3c_join(d.ra, d.dec, s.ra, s.dec,  0.0002777*2) '
        'where  o.source_id is NULL  and '
        "o.created_at > now() - interval '48 hours' and "
        'objectswithflux.id = d.id returning d.id, s.id'
    )
    r = list(r)

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

    

    q = f'''select d.id as id1,  dd.id as id2, oo.source_id, 
    q3c_dist(d.ra, d.dec, dd.ra, dd.dec) * 3600  sep 
    from detections d join objectswithflux o on d.id=o.id 
    join realbogus rb on rb.detection_id = d.id 
    join detections dd on 
    q3c_join(d.ra, d.dec, dd.ra, dd.dec, 0.0002777 * 2) 
    join objectswithflux oo on oo.id = dd.id join 
    realbogus rr on rr.detection_id = dd.id 
    where o.source_id is NULL  and 
    o.created_at > now() - interval '48 hours'  
    and d.id != dd.id and rr.rb_score >= {ASSOC_RB_MIN} 
    and rb.rb_score >= {ASSOC_RB_MIN}'''
    """

    q = f'''select d.id, d.ra, d.dec, o.flux / o.fluxerr as snr 
    from detections d join objectswithflux o on d.id = o.id join realbogus
    rb on rb.detection_id = d.id where o.source_id is NULL 
    and rb.rb_score > {ASSOC_RB_MIN} order by o.created_at asc LIMIT 10000'''

    df = pd.DataFrame(
        list(db.DBSession().execute(q)),
        columns=['id', 'ra', 'dec', 'snr']
    )

    
    df = df.set_index('id')
    df['id'] = df.index.tolist()
    
    #df = df[pd.isna(df['source_id'])]
    #df = df.sort_values('sep').drop_duplicates(subset='id1', keep='first')
    #df = df[df['id1'] > df['id2']]
    
    coord = SkyCoord(df['ra'], df['dec'], unit='deg')
    idx, d2d, d3d = match_coordinates_sky(coord, coord, nthneighbor=2)
    
    ok = df['id'][d2d < 2 * u.arcsec]
    match = df.iloc[idx[d2d < 2 * u.arcsec]]['id']

    grouped = {}
    inverted = {}
    scounter = 0
    scoords = {}
    
    for d1, d2 in zip(ok, match):
        if d1 in inverted and not d2 in inverted:
            s1 = inverted[d1]
            inverted[d2] = s1
            grouped[s1].append(d2)
        if d2 in inverted and not d1 in inverted:
            s2 = inverted[d2]
            inverted[d1] = s2
            grouped[s2].append(d1)
        if d1 in inverted and d2 in inverted:
            s1 = inverted[d1]
            s2 = inverted[d2]
            if s1 != s2:
                s2dets = grouped[s2]
                grouped[s1].extend(s2dets)
                del grouped[s2]
                for d in s2dets:
                    inverted[d] = s1
        else:
            inverted[d1] = scounter
            inverted[d2] = scounter
            grouped[scounter] = [d1, d2]
            scounter += 1

    while iterate_sources(grouped, inverted, df):
        continue

    if debug:
        df = df.iloc[:10]

    with db.DBSession().no_autoflush:
        default_group = db.DBSession().query(
            db.models.Group
        ).get(DEFAULT_GROUP)

        default_instrument = db.DBSession().query(
            db.models.Instrument
        ).get(DEFAULT_INSTRUMENT)

    # cache d1 and d2
    from sqlalchemy.orm import joinedload
    d1 = db.DBSession().query(db.Detection).filter(db.Detection.id.in_([
        int(i) for i in df['id1']
    ])).options(joinedload(db.Detection.thumbnails)).all()

    detcache = {d.id: d for d in d1}

    d2 = db.DBSession().query(db.Detection).filter(db.Detection.id.in_([
        int(i) for i in df['id2']
    ])).options(joinedload(db.Detection.thumbnails)).all()

    for d in d2:
        detcache[d.id] = d

    sources = {}
    curval = db.DBSession().execute("select nextval('namenum')")
    for i, row in tqdm(df.iterrows()):
        d = detcache[row['id1']]
        d2 = detcache[row['id2']]
        
        if row['id1'] in sources and not row['id2'] in sources:
            sources[row['id2']] = sources[row['id1']]
            d2.source = sources[row['id1']]
            db.DBSession().add(d2)
        
        if row['id2'] in sources and not row['id1'] in sources:
            sources[row['id1']] = sources[row['id2']]
            d.source = sources[row['id2']]
            
        else:


            #d2 = db.DBSession().query(db.Detection).get(row['id2'])

            bestdet = d if d.flux / d.fluxerr > d2.flux / d2.fluxerr else d2
            # need to create a new source
            name = publish.get_next_name(num=curval)
            curval += 1
            source = db.models.Source(
                id=name,
                groups=[default_group],
                ra=bestdet.ra,
                dec=bestdet.dec
            )

            # need this to make stamps.
            dummy_phot = db.models.Photometry(
                source=source,
                instrument=default_instrument
            )

            d.source = source
            d2.source = source
            db.DBSession().add(d)
            db.DBSession().add(d2)
            db.DBSession().add(source)
            db.DBSession().add(dummy_phot)

            sources[row['id1']] = source
            sources[row['id2']] = source

            for t in d.thumbnails:
                t.photometry = dummy_phot
                t.source = d.source
                #t.persist()
                db.DBSession().add(t)

            sdss_thumb = db.models.Thumbnail(photometry=dummy_phot,
                                      public_url=source.sdss_url,
                                      type='sdss')
            dr8_thumb = db.models.Thumbnail(photometry=dummy_phot,
                                     public_url=source.desi_dr8_url,
                                     type='dr8')
            db.DBSession().add_all([sdss_thumb, dr8_thumb])

    db.DBSession().execute(f"select setval('namenum', {curval})")
    db.DBSession().flush()

    # now that new sources have been flushed to the database,
    # associate any nearby detections with them

    r = db.DBSession().execute(
        'update objectswithflux set source_id = s.id, '
        'modified = now() from  detections d join objectswithflux o '
        'on d.id = o.id join sources s on '
        'q3c_join(d.ra, d.dec, s.ra, s.dec,  0.0002777*2) '
        'where  o.source_id is NULL  and '
        "o.created_at > now() - interval '48 hours' and "
        'objectswithflux.id = d.id returning d.id, s.id'
    )
    r = list(r)

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

    detection_ids = []
    for row in r:
        detection_ids.append(row[0])
    for i, row in df.iterrows():
        detection_ids.append(row['id1'])

    thumbids = []
    for d in detcache.values():
        for t in d.thumbnails:
            thumbids.append(t.id)

    if os.getenv('NERSC_HOST') == 'cori':
        submit_thumbs(thumbids)

    print(f'triggering alerts and forced photometry for {len(detection_ids)} detections')
    db.DBSession().execute(
        f'''update detections set triggers_alert = 't'
        where detections.id in {tuple(detection_ids)}'''
    )

    # need to commit so that sources will be there for forced photometry
    # jobs running via slurm
    db.DBSession().commit()

if __name__ == '__main__':
    db.DBSession().get_bind().echo=True
    associate()
