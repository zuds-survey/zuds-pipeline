import photutils
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import vstack
from astropy.wcs import WCS

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as psql
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.schema import UniqueConstraint

from .core import Base
from .constants import APER_KEY, APERTURE_RADIUS

__all__ = ['ForcedPhotometry', 'raw_aperture_photometry', 'aperture_photometry']


class ForcedPhotometry(Base):
    id = sa.Column(sa.Integer, primary_key=True)
    __tablename__ = 'forcedphotometry'

    flags = sa.Column(sa.Integer)
    ra = sa.Column(psql.DOUBLE_PRECISION)
    dec = sa.Column(psql.DOUBLE_PRECISION)

    @property
    def mag(self):
        return -2.5 * np.log10(self.flux) + self.image.header['MAGZP'] + \
               self.image.header[APER_KEY]

    @property
    def magerr(self):
        return 1.08573620476 * self.fluxerr / self.flux

    image_id = sa.Column(sa.Integer, sa.ForeignKey('calibratedimages.id',
                                                   ondelete='CASCADE'),
                         index=True)
    image = relationship('CalibratedImage', back_populates='forced_photometry',
                         cascade='all')

    # thumbnails = relationship('Thumbnail', cascade='all')

    source_id = sa.Column(sa.Text,
                          sa.ForeignKey('sources.id', ondelete='CASCADE'),
                          index=True)
    source = relationship('Source', cascade='all')

    flux = sa.Column(sa.Float)
    fluxerr = sa.Column(sa.Float)

    zp = sa.Column(sa.Float)
    filtercode = sa.Column(sa.Text)
    obsjd = sa.Column(sa.Float)

    uniq = UniqueConstraint(image_id, source_id)
    reverse_idx = sa.Index('source_image', source_id, image_id)

    @hybrid_property
    def snr(self):
        return self.flux / self.fluxerr



def raw_aperture_photometry(sci_path, rms_path, mask_path, ra, dec,
                            apply_calibration=False):

    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    coord = SkyCoord(ra, dec, unit='deg')

    with fits.open(sci_path, memmap=False) as shdu:
        header = shdu[0].header
        swcs = WCS(header)
        scipix = shdu[0].data

    with fits.open(rms_path, memmap=False) as rhdu:
        rmspix = rhdu[0].data

    with fits.open(mask_path, memmap=False) as mhdu:
        maskpix = mhdu[0].data


    apertures = photutils.SkyCircularAperture(coord, r=APERTURE_RADIUS)
    phot_table = photutils.aperture_photometry(scipix, apertures,
                                               error=rmspix,
                                               wcs=swcs)


    pixap = apertures.to_pixel(swcs)
    annulus_masks = pixap.to_mask(method='center')
    maskpix = [annulus_mask.cutout(maskpix) for annulus_mask in annulus_masks]


    magzp = header['MAGZP']
    apcor = header[APER_KEY]

    # check for invalid photometry on masked pixels
    phot_table['flags'] = [int(np.bitwise_or.reduce(m, axis=(0, 1))) for
                           m in maskpix]

    phot_table['zp'] = magzp + apcor
    phot_table['obsjd'] = header['OBSJD']
    phot_table['filtercode'] = 'z' + header['FILTER'][-1]


    # rename some columns
    phot_table.rename_column('aperture_sum', 'flux')
    phot_table.rename_column('aperture_sum_err', 'fluxerr')

    return phot_table


def aperture_photometry(calibratable, ra, dec, apply_calibration=False,
                        assume_background_subtracted=False, use_cutout=False,
                        direct_load=None):

    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    coord = SkyCoord(ra, dec, unit='deg')

    if not use_cutout:

        wcs = calibratable.wcs

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

        phot_table['zp'] = calibratable.header['MAGZP'] + calibratable.header['APCOR4']
        phot_table['obsjd'] = calibratable.header['OBSJD']
        phot_table['filtercode'] = 'z' + calibratable.header['FILTER'][-1]


        pixap = apertures.to_pixel(wcs)
        annulus_masks = pixap.to_mask(method='center')
        maskpix = [annulus_mask.cutout(mask.data) for annulus_mask in annulus_masks]

    else:
        phot_table = []
        maskpix = []
        for s in coord:

            if direct_load is not None and 'sci' in direct_load:
                sci_path = direct_load['sci']
            else:
                if assume_background_subtracted:
                    sci_path = calibratable.local_path
                else:
                    sci_path = calibratable.background_subtracted_image.local_path

            if direct_load is not None and 'mask' in direct_load:
                mask_path = direct_load['mask']
            else:
                mask_path = calibratable.mask_image.local_path

            if direct_load is not None and 'rms' in direct_load:
                rms_path = direct_load['rms']
            else:
                rms_path = calibratable.rms_image.local_path

            with fits.open(
                sci_path,
                memmap=True
            ) as f:
                wcs = WCS(f[0].header)

            pixcoord = wcs.all_world2pix([[s.ra.deg, s.dec.deg]], 0)[0]
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
                                            APERTURE_RADIUS.value)


            # something that is photometerable implements mask, background, and wcs
            with fits.open(
                sci_path,
                memmap=True
            ) as f:
                pixels_bkgsub = f[0].data[iymin:iymax, ixmin:ixmax]

            with fits.open(rms_path, memmap=True) as f:
                bkgrms = f[0].data[iymin:iymax, ixmin:ixmax]

            with fits.open(mask_path, memmap=True) as f:
                mask = f[0].data[iymin:iymax, ixmin:ixmax]

            pt = photutils.aperture_photometry(pixels_bkgsub, ap, error=bkgrms)

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
