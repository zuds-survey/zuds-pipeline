import os
import numpy as np
import warnings

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from .core import DBSession
from .constants import GROUP_PROPERTIES
from .utils import ensure_images_have_the_same_properties
from .seeing import estimate_seeing
from .archive import HTTPArchiveCopy
from .image import CalibratableImage, ScienceImage
from .archive import archive

__all__ = ['Coadd', 'ReferenceImage', 'ScienceCoadd']


def _coadd_from_images(cls, images, outfile_name, nthreads=1, data_product=False,
                       tmpdir='/tmp', copy_inputs=False, swarp_kws=None,
                       calculate_seeing=True, addbkg=True):
    """Make a coadd from a bunch of input images"""
    from .swarp import run_coadd

    images = np.atleast_1d(images)
    mskoutname = outfile_name.replace('.fits', '.mask.fits')

    basename = os.path.basename(outfile_name)

    # see if a file with this name already exists in the DB
    predecessor = cls.get_by_basename(basename)

    if predecessor is not None:
        warnings.warn(f'WARNING: A "{cls}" object with the basename '
                      f'"{basename}" already exists. The record will be '
                      f'updated...')

    properties = GROUP_PROPERTIES

    # make sure all images have the same field, filter, ccdid, qid:
    ensure_images_have_the_same_properties(images, properties)

    # call swarp
    coadd = run_coadd(cls, images, outfile_name, mskoutname,
                      addbkg=addbkg, nthreads=nthreads,
                      tmpdir=tmpdir, copy_inputs=copy_inputs,
                      swarp_kws=swarp_kws)
    coaddmask = coadd.mask_image

    coadd.header['FIELD'] = coadd.field = images[0].field
    coadd.header['CCDID'] = coadd.ccdid = images[0].ccdid
    coadd.header['QID'] = coadd.qid = images[0].qid
    coadd.header['FID'] = coadd.fid = images[0].fid

    if calculate_seeing:
        estimate_seeing(coadd)
    coadd.save()

    if data_product:
        coadd_copy = HTTPArchiveCopy.from_product(coadd)
        coaddmask_copy = HTTPArchiveCopy.from_product(coaddmask)
        archive(coadd_copy)
        archive(coaddmask_copy)

    return coadd


class Coadd(CalibratableImage):
    id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id',
                                             ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {
        'polymorphic_identity': 'coadd',
        'inherit_condition': id == CalibratableImage.id
    }

    input_images = relationship('CalibratableImage',
                                secondary='coadd_images',
                                cascade='all')


    @property
    def mjd(self):
        from .joins import CoaddImage
        return DBSession().query(
            sa.func.percentile_cont(
                0.5
            ).within_group(
                (ScienceImage.obsjd - 2400000.5).asc()
            )
        ).join(
            CoaddImage, CoaddImage.calibratableimage_id == ScienceImage.id
        ).join(
            type(self), type(self).id == CoaddImage.coadd_id
        ).filter(
            type(self).id == self.id
        ).first()[0]

    @property
    def min_mjd(self):
        images = self.input_images
        return min([image.mjd for image in images])

    @property
    def max_mjd(self):
        images = self.input_images
        return max([image.mjd for image in images])


    @declared_attr
    def __table_args__(cls):
        return tuple()

    from_images = classmethod(_coadd_from_images)



class ReferenceImage(Coadd):
    id = sa.Column(sa.Integer, sa.ForeignKey('coadds.id', ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'ref',
                       'inherit_condition': id == Coadd.id}


    version = sa.Column(sa.Text)

    single_epoch_subtractions = relationship('SingleEpochSubtraction',
                                             cascade='all')
    multi_epoch_subtractions = relationship('MultiEpochSubtraction',
                                            cascade='all')


class ScienceCoadd(Coadd):
    id = sa.Column(sa.Integer, sa.ForeignKey('coadds.id', ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'scicoadd',
                       'inherit_condition': id == Coadd.id}
    subtraction = relationship('MultiEpochSubtraction', uselist=False,
                               cascade='all')

    binleft = sa.Column(sa.DateTime(timezone=False), nullable=False)
    binright = sa.Column(sa.DateTime(timezone=False), nullable=False)

    @hybrid_property
    def winsize(self):
        return self.binright - self.binleft


