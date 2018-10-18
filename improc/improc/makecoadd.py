import os
import numpy as np
import fits
import zplib
from astropy.io import fits as afits
from numpy.ma import fix_invalid

# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


def make_rms(im, weight):
    """Make the RMS image"""
    saturval = fits.read_header_float(im, 'SATURATE')

    # make rms map
    weighthdul = fits.open(weight)
    weightmap = weighthdul[0].data
    rawrms = np.sqrt(weightmap**-1)
    fillrms = np.sqrt(saturval)
    rms = fix_invalid(rawrms, fill_value=fillrms).data

    rmshdu = fits.PrimaryHDU(rms)
    rmshdul = fits.HDUList([rmshdu])
    rmshdul.writeto(weight.replace('weight', 'rms'))

    # make bpm
    bpm = np.zeros_like(rawrms, dtype='int16')
    bpm[~np.isfinite(rawrms)] = 256

    bpmhdu = fits.PrimaryHDU(bpm)
    hdul = fits.HDUList([bpmhdu])

    hdul.writeto(weight.replace('weight', 'bpm'))



if __name__ == '__main__':

    import argparse
    from mpi4py import MPI

    # set up the inter-rank communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # set up the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-basename', dest='name', required=True,
                        help='Basename of output coadd.', nargs=1)
    parser.add_argument('--input-catalogs', dest='cats', required=True,
                        help='List of catalogs to use for astrometric alignment.', nargs=1)
    parser.add_argument('--input-frames', dest='frames', nargs=1, required=True,
                        help='List of frames to coadd.')
    args = parser.parse_args()

    # distribute the work to each processor
    if rank == 0:
        frames = np.genfromtxt(args.frames[0], dtype=None, encoding='ascii')
        cats = np.genfromtxt(args.cats[0], dtype=None, encoding='ascii')
    else:
        frames = None
        cats = None

    frames = comm.bcast(frames, root=0)
    cats = comm.bcast(cats, root=0)

    if rank == 0:
        # save copies of the full lists on root
        aframes = frames
        acats = cats

    frames = _split(frames, size)[rank]
    cats = _split(cats, size)[rank]

    # now set up a few pointers to auxiliary files read by sextractor
    wd = os.path.dirname(__file__)
    confdir = os.path.join(wd, 'config', 'makecoadd')
    sexconf = os.path.join(confdir, 'scamp.sex')
    scampparam = os.path.join(confdir, 'scamp.param')
    filtname = os.path.join(confdir, 'default.conv')
    nnwname = os.path.join(confdir, 'default.nnw')
    scampconf = os.path.join(confdir, 'scamp.conf')
    swarpconf = os.path.join(confdir, 'default.swarp')

    clargs = '-PARAMETERS_NAME %s -FILTER_NAME %s -STARNNW_NAME %s' % (scampparam, filtname, nnwname)

    mycats = ' '.join(cats)

    # First scamp everything
    syscall = 'scamp -c %s %s' % (scampconf, mycats)
    os.system(syscall)

    # Now make the coadd
    comm.Barrier()

    if rank == 0:
        allims = ' '.join(frames)
        out = args.output_basename + '.fits'
        oweight = args.output_basename + '.weight.fits'
        syscall = 'SWarp -c %s %s -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s' % (swarpconf, allims, out, oweight)
        os.system(syscall)

        # Now postprocess it a little bit
        band = fits.read_header_string(out, 'FILTER')

        if 'r' in band.lower():
            fits.update_header_string(out, 'FILTER', 'r')
        elif 'g' in band.lower():
            fits.update_header_string(out, 'FILTER', 'g')
        elif 'i' in band.lower():
            fits.update_header_string(out, 'FILTER', 'i')
        else:
            raise ValueError('Invalid filter "%s."' % band)

        # TODO make this more general
        fits.update_header_float(out, 'PIXSCALE', 1.0)

        # Add the sky back into the image as a constant
        with afits.open(out, mode='update') as f:
            f[0].data += 150.

        # Make a new catalog
        outcat = args.output_basename + '.cat'
        noise = args.output_basename + '.noise.fits'
        syscall = 'sextractor -c %s -CATALOG_NAME %s -CHECKIMAGE_NAME %s -MAG_ZEROPOINT 27.5 %s'
        syscall = syscall % (sexconf, outcat, noise, out)
        os.system(syscall)

        # And zeropoint the coadd, putting results in the header
        zplib.solve_zeropoint(out, outcat)

        # Now retrieve the zeropoint
        zp = fits.get_header_float(out, 'MAGZP')




    else:
        # I'm done
        MPI.Finalize()
