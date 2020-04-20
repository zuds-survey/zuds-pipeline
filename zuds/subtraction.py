import os
import shutil
import uuid
import subprocess
import numpy as np
from pathlib import Path
from astropy.io import fits
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship

from .core import DBSession
from .fitsfile import HasWCS
from .image import (CalibratableImageBase, ScienceImage, CalibratableImage,
                    FITSImage, CalibratedImage)
from .mask import MaskImageBase, MaskImage
from .coadd import _coadd_from_images, ScienceCoadd
from .constants import APER_KEY
from .archive import archive


__all__ = ['sub_name', 'Subtraction', 'SingleEpochSubtraction',
           'MultiEpochSubtraction']


def sub_name(frame, template):
    frame = f'{frame}'
    template = f'{template}'

    refp = os.path.basename(template)[:-5]
    newp = os.path.basename(frame)[:-5]

    outdir = os.path.dirname(frame)

    subp = '_'.join([newp, refp])

    sub = os.path.join(outdir, 'sub.%s.fits' % subp)
    return sub


class Subtraction(HasWCS):

    @declared_attr
    def reference_image_id(self):
        return sa.Column(sa.Integer, sa.ForeignKey('referenceimages.id',
                                                   ondelete='CASCADE'),
                         index=True)

    @declared_attr
    def reference_image(self):
        return relationship('ReferenceImage', cascade='all',
                            foreign_keys=[self.reference_image_id])

    @property
    def mjd(self):
        return self.target_image.mjd

    @classmethod
    def from_images(cls, sci, ref, data_product=False, tmpdir='/tmp',
                    **kwargs):

        from .hotpants import prepare_hotpants

        refined = kwargs.get('refined', True)

        directory = Path(tmpdir) / uuid.uuid4().hex
        directory.mkdir(exist_ok=True, parents=True)

        # we are gonna copy the inputs into the directory, then reload them
        # off of disk to keep things transactionally isolated
        shutil.copy(sci.local_path, directory)
        shutil.copy(sci.mask_image.local_path, directory)

        if hasattr(sci, '_rmsimg'):
            shutil.copy(sci.rms_image.local_path, directory)
        elif hasattr(sci, '_weightimg'):
            shutil.copy(sci.weight_image.local_path, directory)
        else:
            raise ValueError('Science image must have a weight map or '
                             'rms map defined prior to subtraction.')

        shutil.copy(ref.local_path, directory)
        shutil.copy(ref.mask_image.local_path, directory)
        shutil.copy(ref.weight_image.local_path, directory)

        sciname = os.path.join(directory, sci.basename)
        scimaskn = os.path.join(directory, sci.mask_image.basename)
        refname = os.path.join(directory, ref.basename)
        refmaskn = os.path.join(directory, ref.mask_image.basename)

        transact_sci = CalibratableImageBase.from_file(sciname)
        transact_scimask = MaskImageBase.from_file(scimaskn)
        transact_ref = CalibratableImageBase.from_file(refname)
        transact_refmask = MaskImageBase.from_file(refmaskn)

        transact_sci.mask_image = transact_scimask
        transact_ref.mask_image = transact_refmask

        # output goes into the temporary directory to keep things transactional
        outname = sub_name(transact_sci.local_path, transact_ref.local_path)
        outmask = outname.replace('.fits', '.mask.fits')

        # create the remapped ref, and remapped ref mask. the former will be
        # pixel-by-pixel subtracted from the science image. both will be written
        # to this subtraction's working directory (i.e., `directory`)

        remapped_ref = transact_ref.aligned_to(transact_sci, tmpdir=tmpdir)
        remapped_refmask = remapped_ref.mask_image

        remapped_refname = str(directory / remapped_ref.basename)
        remapped_refmaskname = remapped_refname.replace('.fits', '.mask.fits')

        # flush to the disk
        remapped_ref.map_to_local_file(remapped_refname)
        remapped_refmask.map_to_local_file(remapped_refmaskname)
        remapped_ref.save()
        remapped_refmask.save()
        remapped_ref.parent_image = transact_ref

        # create the mask

        # we truly want this to be a totally new copy of mask, with no
        # uncommitted changes from previous runs
        submask = MaskImageBase()

        submask.basename = os.path.basename(outmask)
        submask.field = sci.field
        submask.ccdid = sci.ccdid
        submask.qid = sci.qid
        submask.fid = sci.fid

        submask.map_to_local_file(outmask)
        badpix = remapped_refmask.data | sci.mask_image.data
        submask.data = badpix
        submask.header = sci.mask_image.header
        submask.header_comments = sci.mask_image.header_comments
        submask.save()

        submask.boolean.map_to_local_file(directory / submask.boolean.basename)
        submask.boolean.save()

        command = prepare_hotpants(transact_sci, remapped_ref, outname,
                                   submask.boolean, directory, tmpdir=tmpdir,
                                   refined=refined)

        final_dir = os.path.dirname(sci.local_path)
        final_out = os.path.join(final_dir, os.path.basename(outname))
        product_map = {
            outname: final_out,
            outname.replace('.fits', '.rms.fits'): final_out.replace('.fits',
                                                                     '.rms.fits'),
            outname.replace('.fits', '.mask.fits'): final_out.replace(
                '.fits', '.mask.fits')
        }

        # run HOTPANTS
        subprocess.check_call(command.split())

        # now modify the sub mask to include the stuff that's masked out from
        # hotpants

        with fits.open(outname) as hdul:
            sd = hdul[0].data

        hotbad = np.zeros_like(submask.data, dtype=int)
        hotbad[sd == 1e-30] = 2**17

        # flip the bits
        submask.data |= hotbad
        submask.header['BIT17'] = 17
        submask.header_comments['BIT17'] = 'MASKED BY HOTPANTS (1e-30) / DG'
        submask.save()

        # now copy the output files to the target directory, ending the
        # transaction

        for f in product_map:
            shutil.copy(f, product_map[f])

        # now read the final output products into database mapped records
        sub = cls.from_file(final_out, load_others=False)
        finalsubmask = MaskImage.from_file(final_out.replace('.fits',
                                                             '.mask.fits'))

        sub._rmsimg = FITSImage.from_file(final_out.replace('.fits',
                                                            '.rms.fits'))

        sub.header['FIELD'] = sub.field = sci.field
        sub.header['CCDID'] = sub.ccdid = sci.ccdid
        sub.header['QID'] = sub.qid = sci.qid
        sub.header['FID'] = sub.fid = sci.fid

        finalsubmask.header['FIELD'] = finalsubmask.field = sci.field
        finalsubmask.header['CCDID'] = finalsubmask.ccdid = sci.ccdid
        finalsubmask.header['QID'] = finalsubmask.qid = sci.qid
        finalsubmask.header['FID'] = finalsubmask.fid = sci.fid

        sub.mask_image = finalsubmask
        sub.reference_image = ref
        sub.target_image = sci

        sub.header['SEEING'] = sci.header['SEEING']
        sub.header_comments['SEEING'] = sci.header_comments['SEEING']


        if isinstance(sub, CalibratedImage):
            sub.header['MAGZP'] = sci.header['MAGZP']
            sub.header[APER_KEY] = sci.header[APER_KEY]
            sub.header_comments['MAGZP'] = sci.header_comments['MAGZP']
            sub.header_comments[APER_KEY] = sci.header_comments[APER_KEY]

        sub.save()
        sub.mask_image.save()

        if data_product:
            archive(sub)
            archive(sub.mask_image)

        shutil.rmtree(directory)

        return sub


class SingleEpochSubtraction(Subtraction, CalibratedImage):
    id = sa.Column(sa.Integer, sa.ForeignKey('calibratedimages.id',
                                             ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'sesub',
                       'inherit_condition': id == CalibratedImage.id}

    target_image_id = sa.Column(sa.Integer, sa.ForeignKey('scienceimages.id',
                                                          ondelete='CASCADE'),
                                index=True)
    target_image = relationship('ScienceImage', cascade='all',
                                foreign_keys=[target_image_id])


def overlapping_subtractions(sci, ref):

    from .joins import CoaddImage

    subq = DBSession().query(
        SingleEpochSubtraction
    ).join(
        ScienceImage.__table__, ScienceImage.id == SingleEpochSubtraction.target_image_id
    ).join(
        CoaddImage, CoaddImage.calibratableimage_id == ScienceImage.id
    ).filter(
        CoaddImage.coadd_id == sci.id,
        SingleEpochSubtraction.reference_image_id == ref.id
    )

    return subq.all()


class MultiEpochSubtraction(Subtraction, CalibratableImage):
    id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id',
                                             ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'mesub',
                       'inherit_condition': id == CalibratableImage.id}
    target_image_id = sa.Column(sa.Integer, sa.ForeignKey('sciencecoadds.id',
                                                          ondelete='CASCADE'),
                                index=True)
    target_image = relationship('ScienceCoadd', cascade='all',
                                foreign_keys=[target_image_id])

    input_images = relationship('SingleEpochSubtraction',
                                secondary='stackedsubtraction_frames',
                                cascade='all')


    @declared_attr
    def __table_args__(cls):
        return tuple()


    @classmethod
    def from_images(cls, sci, ref, data_product=False, tmpdir='/tmp',
                    **kwargs):

        nthreads = kwargs.get('nthreads', 1)
        force_map_subs = kwargs.get('force_map_subs', True)

        if not isinstance(sci, ScienceCoadd):
            raise TypeError(f'Input science image "{sci.basename}" must be '
                            f'an instance of ScienceCoadd, got {type(sci)}.')

        images = overlapping_subtractions(sci, ref)

        if force_map_subs:
            for image in images:
                image.basic_map()

        if len(images) != len(sci.input_images):
            raise ValueError('Number of single-epoch subtractions != number'
                             f' of stack inputs. Stack inputs: '
                             f'{[i.basename for i in sci.input_images]}, '
                             f'Single-epoch subtractions: '
                             f'{[i.basename for i in images]}')

        outfile_name = sub_name(sci.local_path, ref.local_path)

        coadd = _coadd_from_images(cls, images, outfile_name,
                                   nthreads=nthreads, addbkg=False,
                                   calculate_seeing=False)

        coadd.reference_image = ref
        coadd.target_image = sci
        coadd.header['SEEING'] = sci.header['SEEING']
        coadd.save()

        return coadd


