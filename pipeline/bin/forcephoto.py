from astropy.io import fits
from astropy.wcs import WCS
import os
from astropy.visualization import ZScaleInterval
import db
import logging

from uuid import uuid4


APER_RAD_FRAC_SEEING_FWHM = 0.6731
DB_FTP_DIR = '/skyportal/static/thumbnails'
DB_FTP_ENDPOINT = os.getenv('DB_FTP_ENDPOINT')
DB_FTP_USERNAME = 'root'
DB_FTP_PASSWORD = 'root'
DB_FTP_PORT = 222

CHUNK_SIZE = 1000


# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')



if __name__ == '__main__':

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()

    env, cfg = db.load_env()
    db.init_db(**cfg['database'])

    logging.basicConfig(format=f'[Rank {rank:04d}//%(asctime)s] %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S',
                        level=logging.INFO)

    if rank == 0:
        num_workers = size - 1
        task_index = 1
        closed_workers = 0

        images = db.DBSession().query(db.Image)\
                               .filter(db.sa.and_(db.Image.ipac_gid == 2,
                                                  db.Image.disk_sub_path != None,
                                                  db.Image.disk_psf_path != None,
                                                  db.Image.subtraction_exists != False))\
                               .limit(64000).all()

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
            systems = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
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
                subtask_max = task_index * CHUNK_SIZE

                # re-bind the images to this rank's session
                for i, image in enumerate(images):
                    db.DBSession().add(image)

                    my_subtask = (task_index - 1) * CHUNK_SIZE + i + 1
                    logging.info(f'Forcing photometry on image "{image.path}" ({my_subtask} / {subtask_max})')
                    try:
                        image.force_photometry()
                    except FileNotFoundError as e:
                        logging.error(e)

            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)
        MPI.Finalize()
