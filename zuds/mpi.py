import os
import numpy as np

__all__ = ['get_nthreads', 'get_my_share_of_work',
           'has_mpi']


from .constants import NTHREADS_PER_NODE


def default_reader(f):
    return np.atleast_1d(np.genfromtxt(f, dtype=None, encoding='ascii'))


def get_nthreads():
    try:
        from mpi4py import MPI
    except ImportError:
        return NTHREADS_PER_NODE
    else:
        slurm_count = os.getenv('SLURM_CPUS_PER_TASK')
        if slurm_count is None:
            return NTHREADS_PER_NODE
        else:
            return slurm_count


def has_mpi():
    try:
        from mpi4py import MPI
        return True
    except ImportError:
        return False


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
