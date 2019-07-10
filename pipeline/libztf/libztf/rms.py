import numpy as np
from numpy.ma import fix_invalid
from astropy.io import fits

__all__ = ['make_rms', 'medg', 'mkivar']


def mkivar(frame, mask, chkname, wgtname):
    with fits.open(frame) as f, fits.open(mask) as m, fits.open(chkname) as rms:

        # For example, to find all science - image pixels which are uncontaminated, but
        # which may contain clean extracted - source signal(bits 1 or 11), one
        # would “logically AND” the corresponding mask - image
        # with the template value 6141 (= 2^0 + 2^2 +2^3 + 2^4 + 2^5 + 2^6 + 2^7 + 2^8 + 2^9 + 2^10 + 21^2)
        # #and retain pixels where this operation yields zero.
        # To then find which of these pixels contain source signal,
        # one would “AND” the resulting image with 2050 (= 21 + 211)
        # and retain pixels where this operation is non-zero.

        ind = (m[0].data & 6141) == 0
        havesat = False
        try:
            saturval = f[0].header['SATURATE']
        except KeyError:
            pass
        else:
            havesat = True
            saturind = f[0].data >= 0.9 * saturval

        f[0].data[ind] = 1 / rms[0].data[ind]**2
        f[0].data[~ind] = 0.

        if havesat:
            f[0].data[saturind] = 0.

        f.writeto(wgtname, overwrite=True)


def medg(frame, weight=None):
    with fits.open(frame, mode='update') as f:

        # mask stuff where the data is zero, or if a weight is provided, where the weight is zero
        if weight is None:
            data = np.ma.masked_equal(f[0].data, 0., copy=True)
        else:
            with fits.open(weight) as w:
                wmask = np.ma.masked_equal(w[0].data, 0.)
            data = np.ma.masked_array(data=f[0].data, mask=wmask.mask, copy=True)
        seepix = f[0].header['SEEING'] / f[0].header['PIXSCALE']
        zp = f[0].header['MAGZP']
        med = np.median(data.compressed())
        res = data - med
        var = np.median(np.abs(res.compressed()))
        lmt = -2.5 * np.log10(3.0 * np.sqrt(3.14159 * seepix * seepix) * var * 1.48) + zp

        f[0].header['LMT_MG'] = lmt
        f[0].header['SKYSIG'] = var
        f[0].header['MEDSKY'] = med

        f[0].header.comments['LMT_MG'] = '3-sig limiting mag'
        f[0].header.comments['SKYSIG'] = 'Median skysig in cts'
        f[0].header.comments['MEDSKY'] = 'Median sky in cts'



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
        impix = f[0].data

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

    # correct for ghosting
    med = np.median(impix)
    threesig = 3 * (0.5 * (np.percentile(impix, 84) - np.percentile(impix, 16)))
    bpm[impix < med - threesig] = 64 # ghosted

    # write it out
    bpmhdu = afits.PrimaryHDU(bpm)
    hdul = afits.HDUList([bpmhdu])

    hdul.writeto(weight.replace('weight', 'bpm'), overwrite=True)
