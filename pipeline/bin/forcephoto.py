from astropy.io import fits
from astropy.wcs import WCS
import os
from astropy.visualization import ZScaleInterval
import db
import logging
from pathlib import Path
import paramiko
import tempfile

from uuid import uuid4


APER_RAD_FRAC_SEEING_FWHM = 0.6731
DB_FTP_DIR = '/skyportal/static/thumbnails'
DB_FTP_ENDPOINT = os.getenv('DB_FTP_ENDPOINT')
DB_FTP_USERNAME = 'root'
DB_FTP_PASSWORD = 'root'
DB_FTP_PORT = 222

CHUNK_SIZE = 1


# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


def chunk(iterable, chunksize):
    isize = len(iterable)
    nchunks = isize // chunksize if isize % chunksize == 0 else isize // chunksize + 1
    for i in range(nchunks):
        yield i, iterable[i * chunksize : (i + 1) * chunksize]


def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')

def submit_forcephoto(subtraction_dependencies, batch_size=1024, job_script_destination='.',
                      log_destination='.', frame_destination='.', task_name=None):


    log_destination = Path(log_destination)
    frame_destination = Path(frame_destination)
    job_script_destination = Path(job_script_destination)

    nersc_username = os.getenv('NERSC_USERNAME')
    nersc_password = os.getenv('NERSC_PASSWORD')
    nersc_host = os.getenv('NERSC_HOST')
    nersc_account = os.getenv('NERSC_ACCOUNT')
    shifter_image = os.getenv('SHIFTER_IMAGE')
    volumes = os.getenv('VOLUMES')


    estring = os.getenv("ESTRING").replace(r"\x27", "'")

    for i, ch in chunk(list(subtraction_dependencies.keys()), batch_size):

        my_deps = list(set([subtraction_dependencies[key] for key in ch]))
        dependency_string = ':'.join(list(map(str, set(my_deps))))
        sublist = ch
        jobname = f'forcephoto.{task_name}.{i}'

        jobstr = f'''#!/bin/bash
#SBATCH -N 1
#SBATCH -J {jobname}
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -A {nersc_account}
#SBATCH --partition=realtime
#SBATCH --image={shifter_image}
#SBATCH --dependency=afterok:{dependency_string}
#SBATCH -C haswell
#SBATCH --exclusive
#SBATCH --volume="{volumes}"
#SBATCH -o {Path(log_destination).resolve() / jobname}.out

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

srun -n 64 shifter {estring} python /pipeline/bin/forcephoto.py {' '.join(list(map(str, sublist)))}  

'''

        if len(my_deps) == 0:
            jobstr = jobstr.replace('#SBATCH --dependency=afterok:\n', '')

        if job_script_destination is None:
            jobscript = tempfile.NamedTemporaryFile()
        else:
            jobscript = open(Path(job_script_destination / f'{jobname}.sh').resolve(), 'w')

        jobscript.write(jobstr)
        jobscript.seek(0)

        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=nersc_host, username=nersc_username, password=nersc_password)


        command = f'sbatch {jobscript.name}'
        stdin, stdout, stderr = ssh_client.exec_command(command)

        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError(f'SSH Command returned nonzero exit status: {command}')

        out = stdout.read()
        err = stderr.read()

        print(out, flush=True)
        print(err, flush=True)

        jobscript.close()
        ssh_client.close()



if __name__ == '__main__':

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()
    from libztf import ipac_authenticate

    import sys
    sub_ids = list(map(int, sys.argv[1:]))

    env, cfg = db.load_env()
    db.init_db(**cfg['database'])

    FORMAT = '[%(asctime)-15s]: %(message)s'
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fmter = logging.Formatter(fmt=FORMAT)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmter)
    logger.addHandler(handler)

    if rank == 0:
        cookie = ipac_authenticate()
    else:
        cookie = None

    cookie = comm.bcast(cookie, root=0)

    if rank == 0:

        num_workers = size - 1
        task_index = 1
        closed_workers = 0

        images = db.DBSession().query(db.SingleEpochSubtraction)\
                               .filter(db.sa.and_(db.SingleEpochSubtraction.id.in_(sub_ids),
                                                  db.SingleEpochSubtraction.image.has(db.Image.disk_psf_path != None)))\
                               .all()

        #  expunge all the images from the session before sending them to other ranks
        db.DBSession().expunge_all()

        div = len(images) // CHUNK_SIZE
        mod = len(images) % CHUNK_SIZE
        n_tasks = div if mod == 0 else div + 1

        def task_generator():
            for i in range(n_tasks):
                yield (images[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE],)

        tasks = task_generator()

        while closed_workers < num_workers:
            comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == tags.READY:
                # Worker is ready, so send it a task
                if task_index <= n_tasks:
                    task = next(tasks) + (task_index,)
                    comm.send(task, dest=source, tag=tags.START)
                    task_index += 1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)

            elif tag == tags.EXIT:
                closed_workers += 1
                logging.info('Closing worker %d.' % source)

    else:

        while True:
            comm.send(None, dest=0, tag=tags.READY)
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == tags.START:
                # Do the work here
                images, task_index = task
                subtask_max = len(images)

                # re-bind the images to this rank's session
                for i, image in enumerate(images):
                    db.DBSession().add(image)

                    my_subtask = (task_index - 1) * CHUNK_SIZE + i + 1
                    logging.info(f'Forcing photometry on image "{image.disk_path}"')
                    try:
                        image.force_photometry(cookie, logger)
                    except FileNotFoundError as e:
                        logging.error(e)

            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)
        MPI.Finalize()
