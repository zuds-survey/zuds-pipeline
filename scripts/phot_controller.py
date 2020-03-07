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
import random
import numpy as np
import traceback
import pandas as pd
import datetime
from pathlib import Path
from secrets import get_secret

DEFAULT_GROUP = 1
DEFAULT_INSTRUMENT = 1
JOB_SIZE = 64 * 15

ASSOC_RB_MIN = 0.4

# query for the images to process

ZUDS_FIELDS = [523, 524, 574, 575, 576, 623, 624, 625, 626,
               627, 670, 671, 672, 673, 674, 675, 714, 715,
               716, 717, 718, 719, 754, 755, 756, 757, 758,
               759, 789, 790, 791, 792, 793, 819, 820, 821,
               822, 823, 843, 844, 845, 846, 861, 862, 863]
fid_map = {1: 'zg', 2: 'zr', 3: 'zi'}

FORCEPHOT_IMAGE_LIMIT = 1000000


def execute(cmd):
    popen = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True
                             )
    lines = []
    for stdout_line in iter(popen.stdout.readline, ""):
        lines.append(stdout_line)
        print(stdout_line)
        #yield stdout_line

    stdout = '\n'.join(lines)
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

    if 'COMMIT' not in stdout:
        raise RuntimeError(
            f'Transaction rolled back in psql, output was '
            f'"{str(stdout)}"'
        )


def load_output(job):

    detids = np.genfromtxt(job.detection_file, dtype=None, encoding='ascii')
    detids = str(tuple(detids.tolist()))
    mydir = Path(os.path.dirname(__file__))
    sql = mydir / 'loadphot.sql'
    with open(sql, 'r') as f:
        g = f.read().replace('FILENAME', f"'{job.output_file}'")

    sqlo = job.output_file + '.sql'
    with open(sqlo, 'w') as f:
        f.write(g)

    # do the load
    cmd = f"psql -h {get_secret('hpss_dbhost')} -p {get_secret('hpss_dbport')} " \
          f"-d {get_secret('hpss_dbname')} -U {get_secret('hpss_dbusername')} " \
          f"-f {sqlo}"


    execute(cmd.split())

    query = "update detections set alert_ready = 't' where id in %s" % (detids,)
    db.DBSession().execute(query)
    db.DBSession().commit()


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


def submit_forcephot_chain():
    # get the what needs alerts
    detids = db.DBSession().query(db.Detection.id).outerjoin(
        db.Alert, db.Alert.detection_id == db.Detection.id
    ).filter(
        db.Detection.triggers_alert == True,
        db.sa.or_(
            db.Detection.alert_ready == None,
            db.Detection.alert_ready == False
        ),
        db.Alert.id == None
    )

    detids = [d[0] for d in detids]

    if len(detids) == 0:
        raise RuntimeError('No detections to run make alerts for, abandoning '
                           'forced photometry and alerting ')

    ndt = datetime.datetime.utcnow()
    nightdate = f'{ndt.year}{ndt.month:02d}{ndt.day:02d}'

    curdir = os.getcwd()

    scriptname = Path(f'/global/cscratch1/sd/dgold/zuds/'
                      f'nightly/{nightdate}/{ndt}.phot.sh'.replace(' ', '_'))
    scriptname.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(scriptname.parent)

    image_names = db.DBSession().query(db.SingleEpochSubtraction.basename,
                                       db.SingleEpochSubtraction.id).join(
        db.ReferenceImage,
        db.SingleEpochSubtraction.reference_image_id == db.ReferenceImage.id
    ).filter(
        db.ReferenceImage.version == 'zuds5',
    ).all()

    image_names = sorted(image_names,
                         key=lambda s: s[0].split('ztf_')[1].split('_')[0],
                         reverse=True)
    image_names = image_names[:FORCEPHOT_IMAGE_LIMIT]
    random.shuffle(image_names)

    imginname = f'{scriptname}'.replace('.sh', '.in')
    photoutname = imginname.replace('.in', '.output')

    outnames = []
    with open(imginname, 'w') as f:
        for name, idnum in image_names:
            # name = name[0]
            g = name.split('_sciimg')[0].split('_')
            q = g[-1]
            c = g[-3]
            b = g[-4]
            field = g[-5]
            outnames.append(
                f'/global/cfs/cdirs/m937/www/data/scratch/{field}/{c}/'
                f'{q}/{b}/{name} {idnum}')

        f.write('\n'.join(outnames) + '\n')

    scriptname = f'{scriptname}'.replace('.sh', '.load.sh')
    detinname = f'{scriptname}'.replace('.sh', '.in')

    with open(detinname, 'w') as f:
        f.write('\n'.join([str(i) for i in detids]) + '\n')

        jobscript = f"""#!/bin/bash
#SBATCH --image=registry.services.nersc.gov/dgold/ztf:latest
#SBATCH --volume="/global/homes/d/dgold/lensgrinder/pipeline/:/pipeline;/global/homes/d/dgold:/home/desi;/global/homes/d/dgold/skyportal:/skyportal"
#SBATCH -N 13
#SBATCH -C haswell
#SBATCH -q realtime
#SBATCH --exclusive
#SBATCH -J forcephot
#SBATCH -t 00:60:00
#SBATCH -L SCRATCH
#SBATCH -A ***REMOVED***
#SBATCH -o {str(scriptname).replace('.sh', '.out')}

HDF5_USE_FILE_LOCKING=FALSE srun -n 832 -c1 --cpu_bind=cores shifter python $HOME/lensgrinder/scripts/dophot.py {imginname} {photoutname}

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

    jobid = stdout.strip().split()[-1].decode('ascii')

    os.chdir(curdir)
    return jobid, detinname, photoutname


if __name__ == '__main__':
    while True:

        db.DBSession().rollback()

        # look for active jobs
        active_jobs = db.DBSession().query(
            db.ForcePhotJob
        ).filter(
            db.sa.or_(
                db.ForcePhotJob.status == 'processing',
                db.ForcePhotJob.status == 'ready_for_loading',
            )
        )

        # get the slurm jobs and their statuses

        try:
            job_statuses = get_job_statuses()
        except RuntimeError as e:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            print(f'continuing...', flush=True)
            continue

        active_slurm_ids = list(map(str, job_statuses['JOBID'].tolist()))

        for job in active_jobs:
            if job.slurm_id not in active_slurm_ids and job.status == 'processing':
                job.status = 'done'
            elif job.status == 'ready_for_loading':
                now = datetime.datetime.utcnow()
                if not (0 < now.hour < 14):
                    load_output(job)
                    job.status = 'loaded'
            db.DBSession().add(job)
        db.DBSession().commit()

        # reget active jobs

        # look for active jobs
        active_jobs = db.DBSession().query(
            db.ForcePhotJob
        ).filter(
            db.sa.or_(
                db.ForcePhotJob.status == 'processing',
                db.ForcePhotJob.status == 'ready_for_loading',
            )
        ).all()

        if len(active_jobs) == 0:
            try:
                slurm_id, det, out = submit_forcephot_chain()
            except RuntimeError as e:
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                print(f'continuing...', flush=True)
                continue

            job = db.ForcePhotJob(status='processing', slurm_id=slurm_id,
                                  detection_file=det, output_file=out)
            db.DBSession().add(job)

        db.DBSession().commit()

        # submit_Jobs()
