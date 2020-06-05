import io
import os
import gzip
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.visualization.interval import ZScaleInterval
from astropy.io import fits

import sqlalchemy as sa
from sqlalchemy.orm import deferred
from sqlalchemy.dialects import postgresql as psql
from sqlalchemy.orm import relationship

from pathlib import Path

from .core import Base
from .constants import CUTOUT_SIZE
from .archive import URL_PREFIX, STAMP_PREFIX, _mkdir_recursive
from .utils import fid_map


__all__ = ['make_stamp', 'Thumbnail']


class Thumbnail(Base):
    # TODO delete file after deleting row
    type = sa.Column(sa.Enum('new', 'ref', 'sub', 'sdss', 'dr8', "new_gz",
                             'ref_gz', 'sub_gz',
                             name='thumbnail_types', validate_strings=True))
    file_uri = sa.Column(sa.String(), nullable=True, index=False, unique=False)
    public_url = sa.Column(sa.String(), nullable=True, index=False, unique=False)
    origin = sa.Column(sa.String, nullable=True)
    bytes = deferred(sa.Column(psql.BYTEA))

    photometry_id = sa.Column(sa.ForeignKey('photometry.id', ondelete='CASCADE'),
                              nullable=False, index=True)
    photometry = relationship('Photometry', back_populates='thumbnails')

    image_id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id',
                                                   ondelete='CASCADE'),
                         index=True, nullable=True)

    image = relationship('CalibratableImage', cascade='all', back_populates='thumbnails',
                         foreign_keys=[image_id])

    source_id = sa.Column(sa.Text, sa.ForeignKey('sources.id', ondelete='CASCADE'),
                          index=True, nullable=True)
    source = relationship('Source', cascade='all', back_populates='thumbnails',
        foreign_keys=[source_id]
    )

    detection_id = sa.Column(sa.Integer, sa.ForeignKey('detections.id',
                                                       ondelete='CASCADE'),
                             index=True, nullable=True)

    detection = relationship('Detection', cascade='all',
                             back_populates='thumbnails',
        foreign_keys=[detection_id]
    )

    @classmethod
    def from_detection(cls, detection, image):

        from .subtraction import Subtraction
        from .coadd import ReferenceImage

        if isinstance(image, Base):
            linkimage = image
        else:
            linkimage = image.parent_image

        stamp = cls(image=linkimage, detection=detection)

        if isinstance(linkimage, Subtraction):
            stamp.type = 'sub'
        elif isinstance(linkimage, ReferenceImage):
            stamp.type = 'ref'
        else:
            stamp.type = 'new'

        cutout = make_stamp(
            None, detection.ra, detection.dec, None,
            None, image.data, image.wcs, save=False,
            size=CUTOUT_SIZE
        )

        # convert the cutout data to bytes
        # and store that in the database

        fitsbuf = io.BytesIO()
        gzbuf = io.BytesIO()

        fits.writeto(fitsbuf, cutout.data, header=cutout.wcs.to_header())
        with gzip.open(gzbuf, 'wb') as fz:
            fz.write(fitsbuf.getvalue())

        stamp.bytes = gzbuf.getvalue()
        stamp.detection = detection

        return stamp

    def persist(self):
        """Persist a thumbnail to the disk. Currently only works on cori."""

        if os.getenv('NERSC_HOST') != 'cori':
            raise RuntimeError('Must be on cori to persist stamps.')

        data = self.array
        self.source = self.detection.source
        vmin, vmax = ZScaleInterval().get_limits(data)
        img = self.image
        base = f'stamp.{self.id}.jpg'
        relpath = f'stamps/{img.field:06d}/c{img.ccdid:02d}/' \
                  f'q{img.qid}/{fid_map[img.fid]}/{base}'
        name = Path(URL_PREFIX) / relpath
        self.public_url = f'{name}'
        self.file_uri = f'{Path(STAMP_PREFIX) / relpath}'
        _mkdir_recursive(Path(self.file_uri).parent)

        import matplotlib.pyplot as plt
        plt.imsave(self.file_uri, data, vmin=vmin, vmax=vmax,
                   cmap='gray')

        os.chmod(self.file_uri, 0o774)

    @property
    def array(self):
        """Convert the bytes property of a Thumbnail to a numpy array
        representing the equivalent pixel values"""
        if self.bytes is None:
            raise ValueError('Cannot coerce array from empty bytes attribute')
        fitsbuf = io.BytesIO(gzip.decompress(self.bytes))
        array = np.flipud(fits.open(fitsbuf)[0].data)
        return array


def make_stamp(name, ra, dec, vmin, vmax, data, wcs, save=True,
               size=CUTOUT_SIZE):
    coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
    cutout = Cutout2D(data, coord, size, wcs=wcs, fill_value=0.)

    if save:
        import matplotlib.pyplot as plt
        plt.imsave(name, np.flipud(cutout.data), vmin=vmin, vmax=vmax,
                   cmap='gray')
        os.chmod(name, 0o774)
    return cutout
