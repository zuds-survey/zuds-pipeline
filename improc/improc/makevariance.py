import os
import numpy as np
import ffits

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

    # now set up a few pointers to auxiliary files read by sextractor
    wd = os.path.dirname(__file__)
    sexconf = os.path.join(wd, 'config', 'makevariance', 'scamp.sex')

    # and a few c variables used below

    for frame, mask in zip(frames, masks):

        # get the zeropoint from the ffits header using fortran
        zp = ffits.imageclass.get_header_real(frame, 'MAGZP')

        # calculate some properties of the image (skysig, lmtmag, etc.)
        # and store them in the header. note: this call is to compiled fortran
        ffits.medg(frame)

        # now get ready to call source extractor
        syscall = 'sex -c %s -CATALOG_NAME %s -CHECKIMAGE_NAME %s -MAG_ZEROPOINT %f %s'
        catname = frame.replace('fits', 'cat')
        chkname = frame.replace('fits', 'noise.fits')
        syscall = syscall % (sexconf, catname, chkname, zp, frame)

        # do it
        os.system(syscall)

        # now make the inverse variance map using fortran
        wgtname = frame.replace('fits', 'weight.fits')
        ffits.mkivar(frame, mask, chkname, wgtname)
