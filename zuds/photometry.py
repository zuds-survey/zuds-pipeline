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

__all__ = ['ForcedPhotometry', 'raw_aperture_photometry']


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


def raw_aperture_photometry(sci_path, rms_path, mask_path, ra, dec):
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
