# cython: c_string_type=str, c_string_encoding=ascii

import os
import numpy as np
cimport numpy as np


cdef extern from "gethead.hpp":
    void readheader(char* fname, char* key, int datatype, void* value) except +

cdef extern from "fitsio.h":
    int TFLOAT;
    int TSTRING;



if __name__ == '__main__':

    import argparse
    from mpi4py import MPI

    # set up the inter-rank communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # set up the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-frames', dest='frames', required=True,
                        help='List of frames to make variance maps for.', nargs=1)
    parser.add_argument('--input-masks', dest='masks', nargs=1, required=True,
                        help='List of masks corresponding to input frames.')
    args = parser.parse_args()

    # distribute the work to each processor
    if rank == 0:
        frames = np.genfromtxt(args.frames, dtype=None)
        masks = np.genfromtxt(args.masks, dtype=None)
    else:
        frames = None
        masks = None

    frames = comm.scatter(frames, root=0)
    masks = comm.scatter(masks, root=0)

    # now do the work - first allocate memory

    cdef:
        float zp
        void* vp = &zp

    for frame, mask in zip(frames, masks):

        # get the zeropoint from the fits header
        readheader(frame, 'MAGZP', TFLOAT, vp)

        # now call source extractor

