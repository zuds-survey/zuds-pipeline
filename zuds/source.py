import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy import func, Index
from sqlalchemy.dialects import postgresql as psql
from astropy.table import Table

from skyportal.models import Source

from .core import DBSession
from .utils import fid_map
from .image import CalibratableImage
from .photometry import ForcedPhotometry
from .detections import Detection
from .subtraction import SingleEpochSubtraction
from .constants import MJD_TO_JD

import numpy as np

__all__ = ['Source']

# Extension of the skyportal default Source class
Source.thumbnails = relationship('Thumbnail', cascade='all')
Source.accumulated_rb = sa.Column(sa.Float)


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


def best_detection(self):
    return DBSession().query(
        Detection
    ).join(Source).filter(
        Source.id == self.id
    ).order_by(
        (Detection.flux / Detection.fluxerr).desc()
    ).first()

Source.images = images
Source.q3c = Index(
    f'sources_q3c_ang2ipix_idx',
    func.q3c_ang2ipix(
        Source.ra,
        Source.dec)
)

Source.detections = relationship('Detection', cascade='all')
Source.forced_photometry = relationship('ForcedPhotometry',
                                               cascade='all')
Source.best_detection = property(best_detection)

Source.neighbor_info = sa.Column(psql.JSONB)


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

    return q


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


Source.unphotometered_images = property(unphotometered_images)
Source.force_photometry = force_photometry


def light_curve(sourceid):
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
        ForcedPhotometry.source_id == sourceid
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


def lcprop(self):
    return light_curve(self.id)


Source.light_curve = lcprop

