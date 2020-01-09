import photutils
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.table import vstack


APERTURE_RADIUS = 3 * u.pixel
APER_KEY = 'APCOR4'


def aperture_photometry(calibratable, ra, dec, apply_calibration=False,
                        assume_background_subtracted=False, use_cutout=False):

    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    coord = SkyCoord(ra, dec, unit='deg')

    wcs = calibratable.wcs

    if not use_cutout:

        apertures = photutils.SkyCircularAperture(coord, r=APERTURE_RADIUS)

        # something that is photometerable implements mask, background, and wcs
        if not assume_background_subtracted:
            pixels_bkgsub = calibratable.background_subtracted_image.data
        else:
            pixels_bkgsub = calibratable.data

        bkgrms = calibratable.rms_image.data
        mask = calibratable.mask_image.data


        phot_table = photutils.aperture_photometry(pixels_bkgsub, apertures,
                                                   error=bkgrms,
                                                   wcs=wcs)

        pixap = apertures.to_pixel(wcs)
        annulus_masks = pixap.to_mask(method='center')
        maskpix = [annulus_mask.cutout(mask.data) for annulus_mask in annulus_masks]

    else:
        phot_table = []
        maskpix = []
        for s in coord:
            pixcoord = wcs.all_world2pix([[s.ra.deg.value, s.dec.deg.value]], 0)[0]
            pixx, pixy = pixcoord

            nx = calibratable.header['NAXIS1']
            ny = calibratable.header['NAXIS2']

            xmin = max(0, pixx - 1.5 * APERTURE_RADIUS.value)
            xmax = min(nx, pixx + 1.5 * APERTURE_RADIUS.value)

            ymin = max(0, pixy - 1.5 * APERTURE_RADIUS.value)
            ymax = min(ny, pixy + 1.5 * APERTURE_RADIUS.value)

            ixmin = int(np.floor(xmin))
            ixmax = int(np.ceil(xmax))

            iymin = int(np.floor(ymin))
            iymax = int(np.ceil(ymax))

            ap = photutils.CircularAperture([pixx - ixmin, pixy - iymin],
                                            APERTURE_RADIUS)

            # something that is photometerable implements mask, background, and wcs
            if not assume_background_subtracted:
                with fits.open(
                    calibratable.background_subtracted_image.local_path,
                    memmap=True
                ) as f:
                    pixels_bkgsub = f[0].data[iymin:iymax, ixmin:ixmax]
            else:
                with fits.open(
                    calibratable.local_path,
                    memmap=True
                ) as f:
                    pixels_bkgsub = f[0].data[iymin:iymax, ixmin:ixmax]

            with fits.open(calibratable.rms_image.local_path, memmap=True) as f:
                bkgrms = f[0].data[iymin:iymax, ixmin:ixmax]

            with fits.open(calibratable.mask_image.local_path, memmap=True) as f:
                mask = f[0].data[iymin:iymax, ixmin:ixmax]

            pt = photutils.aperture_photometry(pixels_bkgsub, ap,
                                                       error=bkgrms)

            annulus_mask = ap.to_mask(method='center')
            mp = annulus_mask.cutout(mask.data)
            maskpix.append(mp)

            phot_table.append(pt)

        phot_table = vstack(phot_table)

    if apply_calibration:
        magzp = calibratable.header['MAGZP']
        apcor = calibratable.header[APER_KEY]

        phot_table['mag'] = -2.5 * np.log10(phot_table['aperture_sum']) + magzp + apcor
        phot_table['magerr'] = 1.0826 * phot_table['aperture_sum_err'] / phot_table['aperture_sum']


    # check for invalid photometry on masked pixels
    phot_table['flags'] = [int(np.bitwise_or.reduce(m, axis=(0, 1))) for
                           m in maskpix]

    # rename some columns
    phot_table.rename_column('aperture_sum', 'flux')
    phot_table.rename_column('aperture_sum_err', 'fluxerr')

    return phot_table
