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

DEFAULT_GROUP = 1
DEFAULT_INSTRUMENT = 1
JOB_SIZE = 64 * 15


# query for the images to process

ZUDS_FIELDS = [523,524,574,575,576,623,624,625,626,
               627,670,671,672,673,674,675,714,715,
               716,717,718,719,754,755,756,757,758,
               759,789,790,791,792,793,819,820,821,
               822,823,843,844,845,846,861,862,863]
fid_map = {1: 'zg', 2:'zr', 3:'zi'}

FORCEPHOT_IMAGE_LIMIT = 150000


def _update_source_coordinate(source_object, detections):
    # assumes source_object and detections are locked

    snrs = [d.flux / d.fluxerr for d in detections]
    top = np.argmax(snrs)
    det = detections[top]
    source_object.ra = det.ra
    source_object.dec = det.dec


def get_job_statuses():
    if os.getenv('NERSC_HOST') != 'cori':
        raise RuntimeError('Can only check job statuses on cori.')

    cmd = f'squeue -r  -u {os.environ.get("USER")}'
    process = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(
            f'Non-zero exit code from squeue, output was '
            f'"{str(stdout)}", "{str(stderr)}".'
        )

    buf = io.BytesIO(stdout)
    df = pd.read_csv(buf, delim_whitespace=True)
    return df


def submit_job(images):

    ndt = datetime.datetime.utcnow()
    nightdate = f'{ndt.year}{ndt.month:02d}{ndt.day:02d}'

    curdir = os.getcwd()

    scriptname = Path(f'/global/cscratch1/sd/dgold/zuds/'
                      f'nightly/{nightdate}/{ndt}.sh'.replace(' ', '_'))
    scriptname.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(scriptname.parent)

    copies = [i[0] for i in images]
    inname = f'{scriptname}'.replace('.sh', '.in')
    with open(inname, 'w') as f:
        f.write('\n'.join(copies) + '\n')

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

HDF5_USE_FILE_LOCKING=FALSE srun -n 64 -c1 --cpu_bind=cores shifter python $HOME/lensgrinder/scripts/donightly.py {inname} zuds4


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
    jobid = stdout.strip().split()[-1].decode('ascii')
    return jobid


def submit_forcephot_chain():

    ndt = datetime.datetime.utcnow()
    nightdate = f'{ndt.year}{ndt.month:02d}{ndt.day:02d}'

    curdir = os.getcwd()

    scriptname = Path(f'/global/cscratch1/sd/dgold/zuds/'
                      f'nightly/{nightdate}/{ndt}.phot.sh'.replace(' ', '_'))
    scriptname.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(scriptname.parent)

    image_names = db.DBSession().query(db.SingleEpochSubtraction.basename).join(
        db.ReferenceImage,
        db.SingleEpochSubtraction.reference_image_id == db.ReferenceImage.id
    ).filter(
        db.ReferenceImage.version == 'zuds4',
    ).all()

    image_names = sorted([i[0] for i in image_names], key=lambda s: s.split('ztf_')[1].split('_')[0], reverse=True)
    image_names = image_names[:FORCEPHOT_IMAGE_LIMIT]

    imginname = f'{scriptname}'.replace('.sh', '.in')
    outnames = []
    with open(imginname, 'w') as f:
        for name in image_names:
            #name = name[0]
            g = name.split('_sciimg')[0].split('_')
            q = g[-1]
            c = g[-3]
            b = g[-4]
            field = g[-5]
            outnames.append(f'/global/cscratch1/sd/dgold/zuds/{field}/{c}/'
                            f'{q}/{b}/{name}')

        f.write('\n'.join(outnames) + '\n')

    jobscript = f"""#!/bin/bash
#SBATCH --image=registry.services.nersc.gov/dgold/ztf:latest
#SBATCH --volume="/global/homes/d/dgold/lensgrinder/pipeline/:/pipeline;/global/homes/d/dgold:/home/desi;/global/homes/d/dgold/skyportal:/skyportal"
#SBATCH -N 17
#SBATCH -C haswell
#SBATCH -q realtime
#SBATCH --exclusive
#SBATCH -J zuds
#SBATCH -t 00:60:00
#SBATCH -L SCRATCH
#SBATCH -A ***REMOVED***

HDF5_USE_FILE_LOCKING=FALSE srun -n 1088 -c1 --cpu_bind=cores shifter python $HOME/lensgrinder/scripts/dophot.py {imginname} zuds4

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
    jobid = stdout.strip().split()[-1].decode('ascii')

    scriptname = Path(f'/global/cscratch1/sd/dgold/zuds/'
                      f'nightly/{nightdate}/{ndt}.alert.sh'.replace(' ', '_'))
    scriptname.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(scriptname.parent)


    # get the alerts
    detids = db.DBSession().query(db.Detection.id).outerjoin(
        db.Alert, db.Alert.detection_id == db.Detection.id
    ).filter(
        db.Detection.triggers_alert == True,
        db.Alert.id == None
    ).all()

    detids = [d[0] for d in detids]

    detinname = f'{scriptname}'.replace('.sh', '.in')
    with open(detinname, 'w') as f:
        f.write('\n'.join([str(i) for i in detids]) + '\n')


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
#SBATCH --dependency=afterany:{jobid}
#SBATCH --array=0-17

HDF5_USE_FILE_LOCKING=FALSE srun -n 64 -c1 --cpu_bind=cores shifter python $HOME/lensgrinder/scripts/doalert.py {detinname}

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
    jobid = stdout.strip().split()[-1].decode('ascii')

    return jobid


if __name__ == '__main__':
    while True:

        # look for failed jobs and mark them
        currently_processing = db.DBSession().query(
            db.Job
        ).filter(db.Job.status == 'processing')

        also_processing = db.DBSession().query(
            db.ForcePhotJob
        ).filter(db.ForcePhotJob.status == 'processing')

        # get the slurm jobs and their statuses

        try:
            job_statuses = get_job_statuses()
        except RuntimeError as e:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            print(f'continuing...', flush=True)
            continue

        for job in currently_processing:
            if job.slurm_id not in list(map(str, job_statuses['JOBID'].tolist())):
                job.status = 'done'
                db.DBSession().add(job)

        for job in also_processing:
            if job.slurm_id not in list(map(str, job_statuses['JOBID'].tolist())):
                job.status = 'done'
                db.DBSession().add(job)

        db.DBSession().commit()

        imq = '''select final.ap, final.sid from
                (select dummy.ap, dummy.sid from
                    (
                       select h.archive_path as ap , s.id as sid, zf.basename as zbase
                       from ztffiles zf
                       join scienceimages s on zf.id = s.id
                       join ztffilecopies z on z.product_id = s.id
                       join httparchivecopies h on h.id=z.id
                       join (
                           ztffiles z1
                           join referenceimages r
                           on z1.id = r.id
                       ) on (
                          z1.field = zf.field
                          and z1.ccdid=zf.ccdid
                          and z1.qid=zf.qid
                          and z1.fid=zf.fid
                       ) where r.version = 'zuds4'
                       and zf.field  = %d
                       and s.ipac_gid = 2
                       and s.seeing < 4
                       and s.maglimit > 19
                       and zf.basename > 'ztf_20200107'
                     ) dummy
                     left outer join failedsubtractions f
                     on f.target_image_id = dummy.sid
                     left outer join singleepochsubtractions ss
                     on ss.target_image_id = dummy.sid
                     where ss.id is null and f.id is null order by dummy.zbase desc
                ) final left outer join (
                    select calibratableimage_id as cid, job_id as jid
                    from job_images ji join jobs j on ji.job_id = j.id
                    where j.status = 'processing'
                ) disqualifying on final.sid = disqualifying.cid
                where disqualifying.jid is null
'''

        results = []
        for field in ZUDS_FIELDS:
            r = db.DBSession().execute(imq % field)
            results.extend(r.fetchall())

        if len(results) == 0:
            print(f'{datetime.datetime.utcnow()}: No images to process, moving to forced photometry...')
        else:

            nchunks = len(results) // JOB_SIZE
            nchunks += 1 if len(results) % JOB_SIZE != 0 else 0

            for group in np.array_split(results, nchunks):

                try:
                    slurm_id = submit_job(group.tolist())
                except RuntimeError as e:
                    exc_info = sys.exc_info()
                    traceback.print_exception(*exc_info)
                    print(f'continuing...', flush=True)
                    continue

                job = db.Job(status='processing', slurm_id=slurm_id)
                db.DBSession().add(job)
                db.DBSession().flush()

                for row in group:
                    ji = db.JobImage(calibratableimage_id=row[1], job_id=job.id)
                    db.DBSession().add(ji)

                db.DBSession().commit()

        r = db.DBSession().execute(
            'update objectswithflux set source_id = s.id, '
            'modified = now() from  detections d join objectswithflux o '
            'on d.id = o.id  join ztffiles z on '
            'z.id = o.image_id join sources s on '
            'q3c_join(d.ra, d.dec, s.ra, s.dec,  0.000277*2) '
            'where  o.source_id is NULL  and '
            "z.created_at > now() - interval '48 hours' and "
            'objectswithflux.id = d.id returning d.id, s.id')

        print(f'associated {len(list(r))} detections with existing sources')

        db.DBSession().execute('''
        update sources set ra=dummy.ra, dec=dummy.dec, modified=now() 
        from (select g.id, g.ra, g.dec from 
        (select s.id, d.ra, d.dec, rank() over 
        (partition by o.source_id order by o.flux / o.fluxerr desc) 
        from sources s join objectswithflux o on o.source_id = s.id 
        join detections d on o.id = d.id ) g where rank = 1) 
        dummy where sources.id = dummy.id;
        ''')

        q = '''select d.id as id1,  dd.id as id2, oo.source_id, 
        q3c_dist(d.ra, d.dec, dd.ra, dd.dec) * 3600  sep 
        from detections d join objectswithflux o on d.id=o.id 
        join realbogus rb on rb.detection_id = d.id 
        join ztffiles z on z.id = o.image_id  
        join detections dd on 
        q3c_join(d.ra, d.dec, dd.ra, dd.dec, 0.0002777 * 2) 
        join objectswithflux oo on oo.id = dd.id join 
        realbogus rr on rr.detection_id = dd.id 
        where o.source_id is NULL  and 
        z.created_at > now() - interval '48 hours'  
        and d.id != dd.id and rr.rb_score > 0.2 and rb.rb_score > 0.2'''

        df = pd.read_sql(q, db.DBSession().get_bind())
        df = df[pd.isna(df['source_id'])]
        df = df.sort_values('sep').drop_duplicates(subset='id1', keep='first')

        sources = {}
        for i, row in tqdm(df.iterrows()):
            print(i)
            d = db.DBSession().query(db.Detection).get(row['id1'])
            if row['id2'] in sources:
                sources[row['id1']] = sources[row['id2']]
                d.source = sources[row['id2']]
            else:

                d2 = db.DBSession().query(db.Detection).get(row['id2'])
                with db.DBSession().no_autoflush:
                    default_group = db.DBSession().query(
                        db.models.Group
                    ).get(DEFAULT_GROUP)

                    default_instrument = db.DBSession().query(
                        db.models.Instrument
                    ).get(DEFAULT_INSTRUMENT)

                # need to create a new source
                name = publish.get_next_name()
                source = db.models.Source(
                    id=name,
                    groups=[default_group]
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
                    t.persist()
                    db.DBSession().add(t)

                lthumbs = d.source.return_linked_thumbnails()
                db.DBSession().add_all(lthumbs)

        db.DBSession().flush()
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
            detection_ids.append(r[0])
        for i, row in df.iterrows():
            detection_ids.append(row['id1'])

        print(f'triggering alerts and forced photometry for {len(detection_ids)} detections')
        db.DBSession().execute(
            f'''update detections set triggers_alert = 't'
            where detections.id in {tuple(detection_ids)}'''
        )

        # need to commit so that sources will be there for forced photometry
        # jobs running via slurm
        db.DBSession().commit()

        # see if a forcephot chain should be launched
        current_forcephot_jobs = db.DBSession().query(db.ForcePhotJob).filter(
            db.ForcePhotJob.status == 'processing'
        ).all()

        if len(current_forcephot_jobs) == 0:

            try:
                slurm_id = submit_forcephot_chain()
            except RuntimeError as e:
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                print(f'continuing...', flush=True)
                continue

            job = db.ForcePhotJob(status='processing', slurm_id=slurm_id)
            db.DBSession().add(job)

        db.DBSession().commit()
        #submit_Jobs()


