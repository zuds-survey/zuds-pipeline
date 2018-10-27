import numpy as np
from numpy.ma import fix_invalid

__all__ = ['make_rms']


def make_rms(im, weight):
    """Make the RMS image and the bad pixel mask.

    Arguments
    ---------

    im (str): Path to fits file to make the mask and bpm for.

    weight (str): Path to weight map corresponding to `im`.

    Returns
    -------
    Nothing, the new images are written with the same name as `im` but
    with extensions .rms.fits and .weight.fits.

    """

    from astropy.io import fits as afits

    with afits.open(im) as f:
        saturval = f[0].header['SATURATE']

    # make rms map
    weighthdul = afits.open(weight)
    weightmap = weighthdul[0].data

    # turn off warnings for this - can get some infs and nans but they will be
    # filled with saturval indicating bad pixels
    with np.errstate(all='ignore'):
        rawrms = np.sqrt(weightmap**-1)
        fillrms = np.sqrt(saturval)

    rms = fix_invalid(rawrms, fill_value=fillrms).data

    # write it out
    rmshdu = afits.PrimaryHDU(rms)
    rmshdul = afits.HDUList([rmshdu])
    rmshdul.writeto(weight.replace('weight', 'rms'), overwrite=True)

    # make bpm
    bpm = np.zeros_like(rawrms, dtype='int16')
    bpm[~np.isfinite(rawrms)] = 256

    # write it out
    bpmhdu = afits.PrimaryHDU(bpm)
    hdul = afits.HDUList([bpmhdu])

    hdul.writeto(weight.replace('weight', 'bpm'), overwrite=True)
