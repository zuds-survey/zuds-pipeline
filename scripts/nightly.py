import db
import os
import subprocess
import shlex
import sys
import traceback
import pandas as pd
import datetime

JOB_SIZE = 64 * 3

# query for the images to process

ZUDS_FIELDS = [523,524,574,575,576,623,624,625,626,
               627,670,671,672,673,674,675,714,715,
               716,717,718,719,754,755,756,757,758,
               759,789,790,791,792,793,819,820,821,
               822,823,843,844,845,846,861,862,863]

def get_job_statuses():
    if os.getenv('NERSC_HOST') != 'cori':
        raise RuntimeError('Can only check job statuses on cori.')

    cmd = f'squeue -r -h  -u {os.getuid()}'
    process = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(
            f'Non-zero exit code from squeue, output was '
            f'"{stdout.read()}", "{stderr.read()}".'
        )


    df = pd.read_csv(stdout, header=True, delim_whitespace=True)
    return df


def submit_job(images):

    ndt = datetime.datetime.utcnow()
    datestring = f'{ndt.year}{ndt.month:02d}{ndt.day:02d}'

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

HDF5_USE_FILE_LOCKING=FALSE srun -n 64 -c1 --cpu_bind=cores shifter python $HOME/lensgrinder/scripts/donightly.py  zuds4

"""


if __name__ == '__main__':
    while True:

        disqualifying = db.DBSession().query(
            db.Job.id.label('jobid'), db.ScienceImage.id.label('imgid')
        ).select_from(db.Job).join(db.JobImage).join(
            db.ScienceImage,
            db.JobImage.id == db.ScienceImage.id
        ).filter(
            db.Job.status.in_(['processing', 'complete'])
        ).subquery()

        imq = db.DBSession().query(
            db.ScienceImage,
        ).join(
            db.HTTPArchiveCopy
        ).outerjoin(
            disqualifying,
            disqualifying.c.imgid == db.ScienceImage.id
        ).filter(
            disqualifying.c.jobid == None,
            db.ScienceImage.field.in_(
                ZUDS_FIELDS
            ),
            db.ScienceImage.ipac_gid == 2
        ).with_for_update(
            skip_locked=True, of=db.ScienceImage
        ).limit(
            JOB_SIZE
        ).options(
            db.sa.orm.joinedload(
                db.ScienceImage.copies
            )
        ).order_by(
            db.ScienceImage.filefracday
        )

        images = imq.all()

        try:
            slurm_id = submit_job(images)
        except RuntimeError as e:
            exc_info = sys.exc_info()
            traceback.print_exc(*exc_info)
            print(f'continuing...', flush=True)
            continue

        job = db.Job(images=images, status='processing', slurm_id=slurm_id)
        db.DBSession().add(job)
        db.DBSession().commit()

        # look for failed jobs and mark them
        currently_processing = db.DBSession().query(
            db.Job
        ).with_for_update(
            skip_locked=True, of=db.Job
        ).filter(db.Job.status == 'processing')

        # get the slurm jobs and their statuses

        try:
            job_statuses = get_job_statuses()
        except RuntimeError as e:
            exc_info = sys.exc_info()
            traceback.print_exc(*exc_info)
            print(f'continuing...', flush=True)
            continue

        for job in currently_processing:
            if job.slurm_id not in job_statuses['JOBID']:
                job.status = 'failed'
                db.DBSession().add(job)
        db.DBSession().commit()
