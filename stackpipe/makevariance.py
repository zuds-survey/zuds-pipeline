import os
import string
import numpy as np
from pipelib import medg, mkivar, execute
from astropy.io import fits

# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []

if __name__ == '__main__':

    import argparse
    from mpi4py import MPI

    import logging

    # set up the inter-rank communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    FORMAT = '[Rank %(rank)d %(asctime)-15s]: %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    extra = {'rank': rank}

    # set up the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-frames', dest='frames', required=True,
                        help='List of frames to make variance maps for.', nargs=1)
    parser.add_argument('--input-masks', dest='masks', nargs=1, required=True,
                        help='List of masks corresponding to input frames.')
    args = parser.parse_args()

    # distribute the work to each processor
    if rank == 0:
        frames = np.genfromtxt(args.frames[0], dtype=None, encoding='ascii')
        masks = np.genfromtxt(args.masks[0], dtype=None, encoding='ascii')
        frames = np.atleast_1d(frames)
        masks = np.atleast_1d(masks)
    else:
        frames = None
        masks = None

    frames = comm.bcast(frames, root=0)
    masks = comm.bcast(masks, root=0)

    frames = _split(frames, size)[rank]
    masks = _split(masks, size)[rank]

    # now set up a few pointers to auxiliary files read by sextractor
    wd = os.path.dirname(__file__)
    confdir = os.path.join(wd, 'config', 'makevariance')
    sexconf = os.path.join(confdir, 'scamp.sex')
    nnwname = os.path.join(confdir, 'default.nnw')
    filtname = os.path.join(confdir, 'default.conv')
    paramname = os.path.join(confdir, 'scamp.param')

    clargs = '-PARAMETERS_NAME %s -FILTER_NAME %s -STARNNW_NAME %s' % (paramname, filtname, nnwname)

    for frame, mask in zip(frames, masks):

        logging.info('Working image %s' % frame, extra=extra)

        # get the zeropoint from the fits header using fortran
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
        logging.info(filtered_string, extra=extra)

        # now make the inverse variance map using fortran
        wgtname = frame.replace('fits', 'weight.fits')
        mkivar(frame, mask, chkname, wgtname)
