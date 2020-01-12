import db
import io
import os
import subprocess
import shlex
import sys
import shutil
import numpy as np
import traceback
import pandas as pd
import datetime
from pathlib import Path

JOB_SIZE = 64 * 5

# query for the images to process

ZUDS_FIELDS = [523,524,574,575,576,623,624,625,626,
               627,670,671,672,673,674,675,714,715,
               716,717,718,719,754,755,756,757,758,
               759,789,790,791,792,793,819,820,821,
               822,823,843,844,845,846,861,862,863]
fid_map = {1: 'zg', 2:'zr', 3:'zi'}

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


if __name__ == '__main__':
    while True:

        # look for failed jobs and mark them
        currently_processing = db.DBSession().query(
            db.Job
        ).filter(db.Job.status == 'processing')

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
            print(f'{datetime.datetime.utcnow()}: Nothing to do, trying again...')
            continue


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



