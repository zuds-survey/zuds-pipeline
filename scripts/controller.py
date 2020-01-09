import db
import os
import subprocess
import shlex
import sys
import shutil
import traceback
import pandas as pd
import datetime
from pathlib import Path

JOB_SIZE = 64 * 1

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

    cmd = f'squeue -r -h  -u {os.getusername()}'
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


    df = pd.read_csv(stdout, header=True, delim_whitespace=True)
    return df


def submit_job(images):

    ndt = datetime.datetime.utcnow()
    nightdate = f'{ndt.year}{ndt.month:02d}{ndt.day:02d}'

    fnames = [
        f'/global/cscratch1/sd/dgold/zuds/{s.field:06d}/'
        f'c{s.ccdid:02d}/q{s.qid}/{fid_map[s.fid]}/{s.basename}'
        for s in images
    ]

    final = '\n'.join(fnames)

    scriptname = Path(f'/global/cscratch1/sd/dgold/zuds/'
                      f'nightly/{nightdate}/{ndt}.sh'.replace(' ', '_'))
    scriptname.parent.mkdir(parents=True, exist_ok=True)

    copies = [
        list(filter(lambda x: isinstance(x, db.HTTPArchiveCopy),
                    image.copies))[0].archive_path for image in images
    ]

    for source, dest in zip(copies, fnames):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(source, dest)
        shutil.copy(source.replace('sciimg', 'mskimg'),
                    dest.replace('sciimg', 'mskimg'))

    inname = f'{scriptname}'.replace('.sh', '.in')
    with open(inname, 'w') as f:
        f.write(final)

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

    if process.returncode != 0:
        raise RuntimeError(
            f'Non-zero exit code from sbatch, output was '
            f'"{str(stdout)}", "{str(stdout)}".'
        )


    jobid = str(stdout).strip().split()[-1]

    return jobid


if __name__ == '__main__':
    while True:

        disqualifying = db.DBSession().query(
            db.Job.id.label('jobid'), db.ScienceImage.id.label('imgid')
        ).select_from(db.Job).join(db.JobImage).join(
            db.ScienceImage,
            db.JobImage.calibratableimage_id == db.ScienceImage.id
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
        ).outerjoin(
            db.SingleEpochSubtraction,
            db.SingleEpochSubtraction.target_image_id == db.ScienceImage.id
        ).filter(
            disqualifying.c.jobid == None,
            db.SingleEpochSubtraction.id == None,
            db.ScienceImage.field.in_(
                ZUDS_FIELDS
            ),
            db.ScienceImage.ipac_gid == 2,
            db.ScienceImage.basename > 'ztf_20200107'
        ).order_by(
            db.ScienceImage.filefracday.desc()
        ).with_for_update(
            skip_locked=True, of=db.ZTFFile
        ).limit(
            JOB_SIZE
        )

        images = imq.all()

        try:
            slurm_id = submit_job(images)
        except RuntimeError as e:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
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
            traceback.print_exception(*exc_info)
            print(f'continuing...', flush=True)
            continue

        for job in currently_processing:
            if job.slurm_id not in job_statuses['JOBID']:
                job.status = 'failed'
                db.DBSession().add(job)
        db.DBSession().commit()
