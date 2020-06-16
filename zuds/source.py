import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy import func, Index
from sqlalchemy.dialects import postgresql as psql

from .core import DBSession, Base
from .utils import fid_map
from .image import CalibratableImage
from .photometry import ForcedPhotometry
from .detections import Detection
from .subtraction import SingleEpochSubtraction
from .constants import MJD_TO_JD
from .thumbnails import Thumbnail
from .spatial import SpatiallyIndexed

import numpy as np

__all__ = ['Source']


class Source(Base, SpatiallyIndexed):
    id = sa.Column(sa.String, primary_key=True)
    # TODO should this column type be decimal? fixed-precison numeric
    ra = sa.Column(sa.Float)
    dec = sa.Column(sa.Float)

    offset = sa.Column(sa.Float, default=0.0)
    redshift = sa.Column(sa.Float, nullable=True)
    neighbor_info = sa.Column(psql.JSONB)

    altdata = sa.Column(psql.JSONB, nullable=True)
    score = sa.Column(sa.Float, nullable=True)
    accumulated_rb = sa.Column(sa.Float)

    thumbnails = relationship('Thumbnail', cascade='all')
    detections = relationship('Detection', cascade='all')
    forced_photometry = relationship('ForcedPhotometry', cascade='all')

    def add_linked_thumbnails(self):
        sdss_thumb = Thumbnail(public_url=self.sdss_url,
                               type='sdss')
        dr8_thumb = Thumbnail(public_url=self.desi_dr8_url,
                              type='dr8')
        DBSession().add_all([sdss_thumb, dr8_thumb])
        DBSession().commit()

    @property
    def sdss_url(self):
        """Construct URL for public Sloan Digital Sky Survey (SDSS) cutout."""
        return (f"http://skyservice.pha.jhu.edu/DR9/ImgCutout/getjpeg.aspx"
                f"?ra={self.ra}&dec={self.dec}&scale=0.3&width=200&height=200"
                f"&opt=G&query=&Grid=on")

    @property
    def desi_dr8_url(self):
        """Construct URL for public DESI DR8 cutout."""
        return (f"http://legacysurvey.org/viewer/jpeg-cutout?ra={self.ra}"
                f"&dec={self.dec}&size=200&layer=dr8&pixscale=0.262&bands=grz")

    def images(self, type=CalibratableImage):

        candidates = DBSession().query(type).filter(
            func.q3c_radial_query(type.ra,
                                  type.dec,
                                  self.ra, self.dec,
                                  0.64)
        ).filter(
            func.q3c_poly_query(self.ra, self.dec, type.poly)
        )

        return candidates.all()

    @property
    def best_detection(self):
        return DBSession().query(
            Detection
        ).join(Source).filter(
            Source.id == self.id
        ).order_by(
            (Detection.flux / Detection.fluxerr).desc()
        ).first()

    @property
    def light_curve(self):
        from astropy.table import Table
        lc_raw = []

        phot = DBSession().query(
            ForcedPhotometry.obsjd - MJD_TO_JD,
            ForcedPhotometry.filtercode,
            ForcedPhotometry.zp,
            ForcedPhotometry.flux,
            ForcedPhotometry.fluxerr,
            ForcedPhotometry.flags,
            ForcedPhotometry.id
        ).filter(
            ForcedPhotometry.source_id == self.id
        )

        for photpoint in phot:
            photd = {'mjd': photpoint[0],
                     'filter': 'ztf' + photpoint[1][-1],
                     'zp': photpoint[2],  # ap-correct the ZP
                     'zpsys': 'ab',
                     'flux': photpoint[3],
                     'fluxerr': photpoint[4],
                     'flags': photpoint[5],
                     'lim_mag': -2.5 * np.log10(5 * photpoint[4]) + photpoint[2],
                     'id': photpoint[-1]}
            lc_raw.append(photd)

        return Table(lc_raw)

    @property
    def unphotometered_images(self):
        subq = DBSession().query(ForcedPhotometry.id,
                                 ForcedPhotometry.image_id).filter(
            ForcedPhotometry.source_id == self.id
        ).subquery()

        q = DBSession().query(SingleEpochSubtraction).filter(
            func.q3c_radial_query(SingleEpochSubtraction.ra,
                                  SingleEpochSubtraction.dec,
                                  self.ra, self.dec,
                                  0.64)
        ).filter(
            func.q3c_poly_query(self.ra, self.dec, SingleEpochSubtraction.poly)
        ).outerjoin(
            subq, subq.c.image_id == SingleEpochSubtraction.id
        ).filter(
            subq.c.id == None
        )

        return q.all()

    def force_photometry(self, assume_background_subtracted=True):
        out = []
        for i in self.unphotometered_images:
            sci_path = f'/global/cscratch1/sd/dgold/zuds/{i.field:06d}/' \
                       f'c{i.ccdid:02d}/q{i.qid}/{fid_map[i.fid]}/' \
                       f'{i.basename}'

            mask_path = sci_path.replace('.fits', '.mask.fits')
            rms_path = sci_path.replace('.fits', '.rms.fits')

            fp = i.force_photometry(
                self, assume_background_subtracted=assume_background_subtracted,
                use_cutout=True, direct_load={'mask': mask_path, 'sci': sci_path,
                                              'rms': rms_path}
            )

            out.extend(fp)
        return out
