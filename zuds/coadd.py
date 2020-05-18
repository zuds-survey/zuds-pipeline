import os
import numpy as np
import warnings
import shutil
import uuid
import subprocess

import sqlalchemy as sa
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
from pathlib import Path

from .core import DBSession
from .constants import GROUP_PROPERTIES, BKG_VAL
from .utils import ensure_images_have_the_same_properties, get_time
from .seeing import estimate_seeing
from .archive import HTTPArchiveCopy
from .image import CalibratableImage, ScienceImage
from .archive import archive

__all__ = ['Coadd', 'ReferenceImage', 'ScienceCoadd']


def _coadd_from_images(cls, images, outname, data_product=False,
                       tmpdir='/tmp', sci_swarp_kws=None,
                       mask_swarp_kws=None, calculate_seeing=True, addbkg=True,
                       enforce_partition=True, solve_astrometry=False,
                       swarp_zp_key='MAGZP', scamp_kws=None, set_date=True):

    """Make a coadd from a bunch of input images"""
    from .swarp import prepare_swarp_sci, prepare_swarp_mask

    images = np.atleast_1d(images)
    mskoutname = outname.replace('.fits', '.mask.fits')

    basename = os.path.basename(outname)

    # see if a file with this name already exists in the DB
    predecessor = cls.get_by_basename(basename)

    if predecessor is not None:
        warnings.warn(f'WARNING: A "{cls}" object with the basename '
                      f'"{basename}" already exists. The record will be '
                      f'updated...')

    properties = GROUP_PROPERTIES

    if enforce_partition:

        # make sure all images have the same field, filter, ccdid, qid:
        ensure_images_have_the_same_properties(images, properties)

    # call swarp

    from .core import ZTFFile
    from .image import FITSImage, CalibratableImageBase
    from .catalog import PipelineFITSCatalog
    from .mask import MaskImageBase, MaskImage

    # create the directory for the transaction
    directory = Path(tmpdir) / uuid.uuid4().hex
    directory.mkdir(exist_ok=True, parents=True)

    # we are gonna copy the inputs into the directory, then reload them
    # off of disk to keep things transactionally isolated

    transact_images = []

    for image in images:
        shutil.copy(image.local_path, directory)
        if image.mask_image is None:
            raise ValueError(f'Image "{image.basename}" does not have a mask. '
                             f'Map this image to a mask and try again.')
        shutil.copy(image.mask_image.local_path, directory)

        # TODO: I'm not currently certain whether .head files will also
        # automatically translate to .weight or .rms files. Need to
        # investigate this to see if these should be remade by calling
        # sextractor after the .head files are already present, to
        # propagate the new WCS solutions to the weight/noise maps.

        # For now assume the .head files also apply to the weight / rms maps
        # and copy them over

        if hasattr(image, '_rmsimg'):
            shutil.copy(image.rms_image.local_path, directory)
        elif hasattr(image, '_weightimg'):
            shutil.copy(image.weight_image.local_path, directory)

        make_catalog = (not hasattr(image, 'catalog')) or image.catalog is None
        if not make_catalog:
            shutil.copy(image.catalog.local_path, directory)

        # create the transaction elements
        transact_name = directory / image.basename
        transact_mask_name = directory / image.mask_image.basename
        transact_image = CalibratableImageBase.from_file(transact_name)
        transact_mask = MaskImageBase.from_file(transact_mask_name)
        transact_image.mask_image = transact_mask

        if not 'MJD-OBS' in transact_image.header:
            mjd = get_time(transact_image, 'mjd')
            transact_image.header['MJD-OBS'] = mjd
            transact_image.header_comments['MJD-OBS'] = 'MJD of observation (DG)'
            transact_image.save()

        # make the catalog if needed
        if make_catalog:
            transact_image.catalog = PipelineFITSCatalog.from_image(
                transact_image
            )
        else:
            transact_cat_name = directory / image.catalog.basename
            transact_cat = PipelineFITSCatalog.from_file(transact_cat_name)
            transact_image.catalog = transact_cat

        transact_images.append(transact_image)

    if solve_astrometry:
        from .scamp import calibrate_astrometry
        calibrate_astrometry(transact_images, scamp_kws=scamp_kws,
                             tmpdir=tmpdir)

    transact_outname = f'{directory / os.path.basename(outname)}'
    command = prepare_swarp_sci(transact_images, transact_outname, directory,
                                swarp_kws=sci_swarp_kws,
                                swarp_zp_key=swarp_zp_key)

    # run swarp
    while True:
        try:
            subprocess.check_call(command.split())
        except OSError as e:
            if e.errno == 14:
                continue
            else:
                raise e
        else:
            break

    # now swarp together the masks
    transact_masks = [image.mask_image for image in transact_images]

    transact_mskoutname = f'{directory / os.path.basename(mskoutname)}'
    mskoutweightname = directory / Path(mskoutname.replace('.fits',
                                                           '.weight.fits')).name

    command = prepare_swarp_mask(transact_masks, transact_mskoutname,
                                 mskoutweightname, directory,
                                 swarp_kws=mask_swarp_kws)

    # run swarp
    while True:
        try:
            subprocess.check_call(command.split())
        except OSError as e:
            if e.errno == 14:
                continue
            else:
                raise e
        else:
            break

    transact_weightname = transact_outname.replace('.fits', '.weight.fits')
    weight_outname = outname.replace('.fits', '.weight.fits')

    # move things back over
    product_map = {
        transact_outname: outname,
        transact_weightname: weight_outname,
        transact_mskoutname: mskoutname
    }

    for key in product_map:
        shutil.copy(key, product_map[key])

    # load the result
    coadd = cls.from_file(outname, load_others=False)
    coadd._weightimg = FITSImage.from_file(weight_outname)

    coaddmask = MaskImage.from_file(mskoutname)
    coaddmaskweight = FITSImage.from_file(mskoutweightname)
    coaddmask.update_from_weight_map(coaddmaskweight)

    # keep a record of the images that went into the coadd
    coadd.input_images = images.tolist()
    coadd.mask_image = coaddmask
    coaddmask.parent_image = coadd

    # set the ccdid, qid, field, fid for the coadd
    # (and mask) based on the input images if `enforce_partition`
    if enforce_partition:
        for prop in GROUP_PROPERTIES:
            for img in [coadd, coaddmask]:
                setattr(img, prop, getattr(images[0], prop))
                img.header[prop.upper()] = getattr(images[0], prop)

    # set a mean date in the header
    if set_date:
        mjds = [get_time(i, 'mjd') for i in images]
        coadd.header['MJD-OBS'] = np.median(mjds)
        coadd.header_comments['MJD-OBS'] = 'Median MJD of the coadd inputs (DG)'

    if addbkg:
        coadd.data += BKG_VAL

    # save the coadd to disk
    coadd.save()
    coaddmask.save()

    # clean up
    for im in [coadd] + images.tolist():
        if f'{directory}' in im.weight_image.local_path:
            del im._weightimg

    shutil.rmtree(directory)

    if solve_astrometry:
        from .scamp import calibrate_astrometry
        calibrate_astrometry([coadd], inplace=True, scamp_kws=scamp_kws,
                             tmpdir=tmpdir)
        coadd.catalog = PipelineFITSCatalog.from_image(coadd)

    if calculate_seeing:
        estimate_seeing(coadd)

    coadd.save()

    if data_product:
        coadd_copy = HTTPArchiveCopy.from_product(coadd)
        coaddmask_copy = HTTPArchiveCopy.from_product(coadd.mask_image)
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


