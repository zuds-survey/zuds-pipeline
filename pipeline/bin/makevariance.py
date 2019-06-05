import os
import string
import numpy as np
from liblg import medg, mkivar, execute, make_rms, solve_zeropoint
from astropy.io import fits
#from calibrate import calibrate
import paramiko
import tempfile

from pathlib import Path

__all__ = ['make_variance']


# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


def chunk(iterable, chunksize):
    isize = len(iterable)
    nchunks = isize // chunksize if isize % chunksize == 0 else isize // chunksize + 1
    for i in range(nchunks):
        yield iterable[i * chunksize : (i + 1) * chunksize]


def _read_clargs(val):
    if val[0].startswith('@'):
        # then its a list
        val = np.genfromtxt(val[0][1:], dtype=None, encoding='ascii')
        val = np.atleast_1d(val)
    return np.asarray(val)


def submit_makevariance(frames, masks, batch_size=1024, dependencies=None, job_script_destination=None,
                        log_destination='.', task_name=None):


    nersc_username = os.getenv('NERSC_USERNAME')
    nersc_password = os.getenv('NERSC_PASSWORD')
    nersc_host = os.getenv('NERSC_HOST')
    nersc_account = os.getenv('NERSC_ACCOUNT')
    shifter_image = os.getenv('SHIFTER_IMAGE')
    volumes = os.getenv('VOLUMES')

    log_destination = Path(log_destination)

    dependency_dict = {}

    for i, (cframes, cmasks) in chunk(list(zip(frames, masks)), batch_size):

        gdeps = ':'.join(list(map(str, set([dependencies[frame] for frame in cframes]))))

        gframes = '\n'.join([Path(frame).resolve() for frame in cframes])
        gmasks = '\n'.join([Path(mask).resolve() for mask in cmasks])

        scriptstr = f'''#!/bin/bash
#SBATCH -N 1
#SBATCH -J v{task_name}.{i}
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -A {nersc_account}
#SBATCH --mail-type=ALL
#SBATCH --partition=realtime
#SBATCH --image={shifter_image}
#SBATCH --exclusive
#SBATCH -C haswell
#SBATCH --volume="{volumes}"
#SBATCH -o {log_destination.resolve()}/v{task_name}.{i}.out
#SBATCH --dependency=afterok:{gdeps}

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

news="{gframes}"
masks="{gmasks}"
srun -n 64 shifter python /pipeline/bin/makevariance.py --input-frames $news --input-masks $masks
'''

        if job_script_destination is None:
            jobscript = tempfile.NamedTemporaryFile()
        else:
            job_script_destination = Path(job_script_destination)
            jobscript = open(job_script_destination.resolve() / f'v{task_name}.{i}.sh', 'w')

        jobscript.write(scriptstr)
        jobscript.seek(0)


        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh_client.connect(hostname=nersc_host, username=nersc_username, password=nersc_password)

        syscall = f'sbatch {jobscript.name}'
        stdin, stdout, stderr = ssh_client.exec_command(syscall)

        retcode = stdout.channel.return_code
        if retcode != 0:
            raise RuntimeError(f'Unable to submit job with script: "{scriptstr}"')

        out = stdout.read()
        err = stderr.read()

        print(out, flush=True)
        print(err, flush=True)

        jobid = int(out.strip().split()[-1])
        jobscript.close()

        for frame in cframes:
            dependency_dict[frame] = jobid

    return dependency_dict


def make_variance(frames, masks, logger=None, extra={}):

    # now set up a few pointers to auxiliary files read by sextractor
    wd = os.path.dirname(__file__)
    confdir = os.path.join(wd, '..', 'astromatic', 'makevariance')
    sexconf = os.path.join(confdir, 'scamp.sex')
    nnwname = os.path.join(confdir, 'default.nnw')
    filtname = os.path.join(confdir, 'default.conv')
    paramname = os.path.join(confdir, 'scamp.param')

    clargs = '-PARAMETERS_NAME %s -FILTER_NAME %s -STARNNW_NAME %s' % (paramname, filtname, nnwname)

    for frame, mask in zip(frames, masks):

        if logger is not None:
            logger.info('Working image %s' % frame, extra=extra)

        # get the zeropoint from the fits header using fortran
        #calibrate(frame)

        with fits.open(frame) as f:
            zp = f[0].header['MAGZP']

        # calculate some properties of the image (skysig, lmtmag, etc.)
        # and store them in the header. note: this call is to compiled fortran
        medg(frame)

        # now get ready to call source extractor
        syscall = 'sex -c %s -CATALOG_NAME %s -CHECKIMAGE_NAME %s -MAG_ZEROPOINT %f %s'
        catname = frame.replace('fits', 'cat')
        chkname = frame.replace('fits', 'noise.fits')
        syscall = syscall % (sexconf, catname, chkname, zp, frame)
        syscall = ' '.join([syscall, clargs])

        # do it
        stdout, stderr = execute(syscall)

        # parse the results into something legible
        stderr = str(stderr, encoding='ascii')
        filtered_string = ''.join(list(filter(lambda x: x in string.printable, stderr)))
        splf = filtered_string.split('\n')
        splf = [line for line in splf if '[1M>' not in line]
        filtered_string = '\n'.join(splf)
        filtered_string = '\n' + filtered_string.replace('[1A', '')

        # log it
        if logger is not None:
            logger.info(filtered_string, extra=extra)

        # now make the inverse variance map using fortran
        wgtname = frame.replace('fits', 'weight.fits')
        mkivar(frame, mask, chkname, wgtname)

        # and make the bad pixel masks and rms images
        make_rms(frame, wgtname)


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

    make_variance(frames, masks, logger, extra=extra)
