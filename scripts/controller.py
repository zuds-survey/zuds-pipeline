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

    copies = [
        list(filter(lambda x: isinstance(x, db.HTTPArchiveCopy),
                    image.copies))[0].archive_path for image in images
    ]
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


        try:
            job_statuses = get_job_statuses()
        except RuntimeError as e:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            print(f'continuing...', flush=True)
            continue

        imq = db.DBSession().query(
            db.HTTPArchiveCopy.archive_path,
        ).select_from(db.ScienceImage).join(
            db.HTTPArchiveCopy
        ).join(
            # make sure it has a reference Image
            db.ReferenceImage,
            db.sa.and_(
                db.ReferenceImage.field == db.ScienceImage.field,
                db.ReferenceImage.ccdid == db.ScienceImage.ccdid,
                db.ReferenceImage.qid == db.ScienceImage.qid,
                db.ReferenceImage.fid == db.ScienceImage.fid
            )
        ).outerjoin(
            db.SingleEpochSubtraction,
            db.SingleEpochSubtraction.target_image_id == db.ScienceImage.id
        ).outerjoin(
            db.FailedSubtraction,
            db.FailedSubtraction.target_image_id == db.ScienceImage.id
        ).filter(
            db.SingleEpochSubtraction.id == None,
            db.ReferenceImage.version == 'zuds4',
            db.FailedSubtraction.id == None,
            db.ScienceImage.field.in_(
                ZUDS_FIELDS
            ),
        ).with_for_update(
            skip_locked=True, of=db.ScienceImage.__table__
        )

        images = imq.all()

        if len(images) == 0:
            print(f'{datetime.datetime.utcnow()}: Nothing to do, trying again...')
            continue


        nchunks = len(images) // JOB_SIZE
        nchunks += 1 if len(images) % JOB_SIZE != 0 else 0

        for group in np.array_split(images, nchunks):

            try:
                slurm_id = submit_job(group.tolist())
            except RuntimeError as e:
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                print(f'continuing...', flush=True)
                continue

            job = db.Job(images=group.tolist(), status='processing', slurm_id=slurm_id)
            db.DBSession().add(job)
            db.DBSession().commit()

