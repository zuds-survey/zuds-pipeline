import sep
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from skyportal.models import DBSession, Instrument, ForcedPhotometry

APER_RAD_FRAC_SEEING_FWHM = 0.6731


def force_photometry(sources, sub_list):

    instrument = DBSession().query(Instrument).filter(Instrument.name.like('%ZTF%')).first()

    for image in sub_list:
        with fits.open(image) as hdulist:
            hdu = hdulist[1]
            zeropoint = hdu.header['MAGZP']
            seeing = hdu.header['SEEING']  # FWHM of seeing
            mjd = hdu.header['OBSMJD']
            r_aper_arcsec = APER_RAD_FRAC_SEEING_FWHM * seeing
            r_aper_pix = r_aper_arcsec / 1.013 # arcsec per pixel
            wcs = WCS(hdu.header)
            maglim = hdu.header['MAGLIM']
            band = hdu.header['FILTER'][-1].lower()

            image = hdu.data
            rms = 1.4826 * np.median(np.abs(image - np.median(image)))

        for source in sources:

            # get the RA and DEC of the source
            ra, dec = source.ra, source.dec

            # convert the ra and dec into pixel coordinates
            x, y = wcs.wcs_world2pix([[ra, dec]], 0.)[0]
            flux, fluxerr, flag = sep.sum_circle(image, x, y, r_aper_pix, err=rms)

            flux = flux[()]
            fluxerr = fluxerr[()]

            if flag == 0:

                force_point = ForcedPhotometry(mjd=mjd, flux=flux, fluxerr=fluxerr,
                                               zp=zeropoint, lim_mag=maglim, filter=band,
                                               source=source, instrument=instrument)
                DBSession().add(force_point)


    DBSession().commit()
