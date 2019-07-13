import os
import string
import numpy as np
from astropy.io import fits
import time
import libztf
import galsim
import paramiko
import tempfile
from galsim import des
from calibrate import calibrate

from pathlib import Path

__all__ = ['make_variance']


# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


def chunk(iterable, chunksize):
    isize = len(iterable)
    nchunks = isize // chunksize if isize % chunksize == 0 else isize // chunksize + 1
    for i in range(nchunks):
        yield i, iterable[i * chunksize : (i + 1) * chunksize]


def _read_clargs(val):
    if val[0].startswith('@'):
        # then its a list
        val = np.genfromtxt(val[0][1:], dtype=None, encoding='ascii')
        val = np.atleast_1d(val)
    return np.asarray(val)


def submit_makevariance(frames, masks, batch_size=1024, job_script_destination=None,
                        log_destination='.', frame_destination='.', task_name=None):


    nersc_username = os.getenv('NERSC_USERNAME')
    nersc_password = os.getenv('NERSC_PASSWORD')
    nersc_host = os.getenv('NERSC_HOST')
    nersc_account = os.getenv('NERSC_ACCOUNT')
    shifter_image = os.getenv('SHIFTER_IMAGE')
    volumes = os.getenv('VOLUMES')


    log_destination = Path(log_destination)
    frame_destination = Path(frame_destination)

    dependency_dict = {}

    for i, ch in chunk(list(zip(frames, masks)), batch_size):

        # unzip the list
        cframes, cmasks = list(zip(*ch))

        absframes = [f'{frame_destination / frame}' for frame in cframes]
        absmasks = [f'{frame_destination / mask}' for mask in cmasks]

        gframes = '\n'.join(absframes)
        gmasks = '\n'.join(absmasks)

        scriptstr = f'''#!/bin/bash
#SBATCH -N 1
#SBATCH -J var.{task_name}.{i}
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -A {nersc_account}
#SBATCH --partition=realtime
#SBATCH --image={shifter_image}
#SBATCH --exclusive
#SBATCH -C haswell
#SBATCH --volume="{volumes}"
#SBATCH -o {log_destination.resolve()}/var.{task_name}.{i}.out

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

news="{gframes}"
masks="{gmasks}"
srun -n 64 shifter python /pipeline/bin/makevariance.py --input-frames $news --input-masks $masks --wait
'''

        if job_script_destination is None:
            jobscript = tempfile.NamedTemporaryFile()
        else:
            job_script_destination = Path(job_script_destination)
            jobscript = open(job_script_destination.resolve() / f'var.{task_name}.{i}.sh', 'w')

        jobscript.write(scriptstr)
        jobscript.seek(0)


        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh_client.connect(hostname=nersc_host, username=nersc_username, password=nersc_password)

        syscall = f'sbatch {jobscript.name}'
        stdin, stdout, stderr = ssh_client.exec_command(syscall)

        out = stdout.read()
        err = stderr.read()

        print(out, flush=True)
        print(err, flush=True)

        retcode = stdout.channel.recv_exit_status()
        if retcode != 0:
            raise RuntimeError(f'Unable to submit job with script: "{scriptstr}", nonzero retcode')

        jobid = int(out.strip().split()[-1])
        jobscript.close()

        for frame in cframes:
            dependency_dict[frame] = jobid

    return dependency_dict


def make_variance(frames, masks, logger=None, extra={}):

    for frame, mask in zip(frames, masks):

        if logger is not None:
            logger.info('Working image %s' % frame, extra=extra)

        # get the zeropoint from the fits header using fortran
        calibrate(frame)



if __name__ == '__main__':

    import argparse
    from mpi4py import MPI

    import logging

    # set up the inter-rank communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    FORMAT = '[Rank %(rank)d %(asctime)-15s]: %(message)s'
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fmter = logging.Formatter(fmt=FORMAT)
    extra = {'rank': rank}

    # set up the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-frames', dest='frames', required=True,
                        help='Frames to make variance maps for. Prefix with "@" to read from a list.', nargs='+')
    parser.add_argument('--input-masks', dest='masks', nargs='+', required=True,
                        help='Masks corresponding to input frames. Prefix with "@" to read from a list.')
    parser.add_argument('--wait', help='Wait for frames and masks to exist if they dont already exist.',
                        action='store_true', default=False)
    args = parser.parse_args()

    # distribute the work to each processor
    if rank == 0:
        frames = _read_clargs(args.frames)
        masks = _read_clargs(args.masks)
    else:
        frames = None
        masks = None

    frames = comm.bcast(frames, root=0)
    masks = comm.bcast(masks, root=0)

    frames = _split(frames, size)[rank]
    masks = _split(masks, size)[rank]

    print(f'rank {rank} has frames {frames}', flush=True)
    print(f'rank {rank} has masks {masks}', flush=True)

    if args.wait:
        while not all([Path(p).exists() for p in frames.tolist() + masks.tolist()]):
            time.sleep(2.)
        time.sleep(2.)

    make_variance(frames, masks, logger, extra=extra)
