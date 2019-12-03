import numpy as np
import os


def default_reader(f):
    return np.atleast_1d(np.genfromtxt(f, dtype=None, encoding='ascii'))


def get_my_share_of_work(fname, reader=default_reader):

    try:
        from mpi4py import MPI
    except ImportError:
        return reader(fname)
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        is_jobarray = os.getenv('SLURM_ARRAY_JOB_ID') is not None

        if rank == 0:
            files = reader(fname)

            # this is the job array part
            if is_jobarray:
                job_array_index = int(os.getenv('SLURM_ARRAY_TASK_ID'))
                job_array_ntasks = int(os.getenv('SLURM_ARRAY_TASK_MAX')) + 1
                files = np.array_split(files, job_array_ntasks)[job_array_index]

            # this is the standard MPI part (within a single job)
            files = np.array_split(files, size)
        else:
            files = None

        files = comm.scatter(files, root=0)
        return files
