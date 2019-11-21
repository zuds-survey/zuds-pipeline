import photutils
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u


APERTURE_RADIUS = 3 * u.pixel
APER_KEY = 'APCOR4'


def aperture_photometry(calibratable, ra, dec, apply_calibration=False):

    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    coord = SkyCoord(ra, dec, unit='deg')
    apertures = photutils.SkyCircularAperture(coord, r=APERTURE_RADIUS)

    # something that is photometerable implements mask, background, and wcs
    pixels_bkgsub = calibratable.data - calibratable.background_image.data
    bkgrms = calibratable.rms_image.data
    mask = calibratable.mask_image.data
    wcs = calibratable.wcs

    phot_table = photutils.aperture_photometry(pixels_bkgsub, apertures,
                                               error=bkgrms,
                                               wcs=wcs)

    pixap = apertures.to_pixel(wcs)
    annulus_masks = pixap.to_mask(method='center')
    maskpix = [annulus_mask.cutout(mask.data) for annulus_mask in annulus_masks]


    if apply_calibration:
        magzp = calibratable.header['MAGZP']
        apcor = calibratable.header[APER_KEY]

        phot_table['mag'] = -2.5 * np.log10(phot_table['aperture_sum']) + magzp + apcor
        phot_table['magerr'] = 1.0826 * phot_table['aperture_sum_err'] / phot_table['aperture_sum']


    # check for invalid photometry on masked pixels
    phot_table['flags'] = [np.bitwise_or.reduce(m, axis=(0, 1)).astype(int) for
                           m in maskpix]

    # rename some columns
    phot_table.rename_column('aperture_sum', 'flux')
    phot_table.rename_column('aperture_sum_err', 'fluxerr')

    return phot_table
