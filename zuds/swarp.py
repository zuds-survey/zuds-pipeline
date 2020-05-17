import os
import uuid
import shutil
from pathlib import Path
import subprocess
from astropy.wcs import WCS


from .utils import initialize_directory
from .constants import BKG_BOX_SIZE, GROUP_PROPERTIES, BKG_VAL
from .mask import MaskImageBase, MaskImage

__all__ = ['prepare_swarp_sci', 'prepare_swarp_mask', 'prepare_swarp_align',
           'run_coadd', 'run_align']


CONF_DIR = Path(__file__).parent / 'astromatic/makecoadd'
SCI_CONF = CONF_DIR / 'default.swarp'
MSK_CONF = CONF_DIR / 'mask.swarp'


def prepare_swarp_sci(images, outname, directory, swarp_kws=None,
                      swarp_zp_key='MAGZP'):

    conf = SCI_CONF
    initialize_directory(directory)

    impaths = [im.local_path for im in images]

    # normalize all images to the same zeropoint
    for im, path in zip(images, impaths):
        if swarp_zp_key in im.header:
            fluxscale = 10**(-0.4 * (im.header['MAGZP'] - 25.))
            im.header['FLXSCALE'] = fluxscale
            im.header_comments['FLXSCALE'] = 'Flux scale factor for coadd / DG'
            im.header['FLXSCLZP'] = 25.
            im.header_comments['FLXSCLZP'] = 'FLXSCALE equivalent ZP / DG'
            opath = im.local_path
            im.map_to_local_file(path)
            im.save()
            im.map_to_local_file(opath)

    # if weight images do not exist yet, write them to temporary
    # directory
    wgtpaths = []
    for image in images:
        if not image.weight_image.ismapped:
            wgtpath = f"{directory / image.basename.replace('.fits', '.weight.fits')}"
            image.weight_image.map_to_local_file(wgtpath)
            image.weight_image.save()
        else:
            wgtpath = image.weight_image.local_path
        wgtpaths.append(wgtpath)

    # need to write the images to a list, to avoid shell commands that are too
    # long and trigger SIGSEGV
    inlist = directory / 'images.in'
    with open(inlist, 'w') as f:
        for p in impaths:
            f.write(f'{p}\n')

    inweight = directory / 'weight.in'
    with open(inweight, 'w') as f:
        for p in wgtpaths:
            f.write(f'{p}\n')

    # get the output weight image in string form
    wgtout = outname.replace('.fits', '.weight.fits')

    syscall = f'swarp -c {conf} @{inlist} ' \
              f'-BACK_SIZE {BKG_BOX_SIZE} ' \
              f'-IMAGEOUT_NAME {outname} ' \
              f'-VMEM_DIR {directory} ' \
              f'-RESAMPLE_DIR {directory} ' \
              f'-WEIGHT_IMAGE @{inweight} ' \
              f'-WEIGHTOUT_NAME {wgtout} '

    if swarp_kws is not None:
        for kw in swarp_kws:
            syscall += f'-{kw.upper()} {swarp_kws[kw]} '

    return syscall


def prepare_swarp_mask(masks, outname, mskoutweightname, directory,
                       swarp_kws=None):


    conf = MSK_CONF
    initialize_directory(directory)

    # get the images in string form
    allims = ' '.join([c.local_path for c in masks])

    syscall = f'swarp -c {conf} {allims} ' \
              f'-SUBTRACT_BACK N ' \
              f'-IMAGEOUT_NAME {outname} ' \
              f'-VMEM_DIR {directory} ' \
              f'-RESAMPLE_DIR {directory} ' \
              f'-WEIGHTOUT_NAME {mskoutweightname} '

    if swarp_kws is not None:
        for kw in swarp_kws:
            syscall += f'-{kw.upper()} {swarp_kws[kw]} '

    return syscall


def prepare_swarp_align(image, other, directory, nthreads=1,
                        persist_aligned=False):
    conf = SCI_CONF
    shutil.copy(image.local_path, directory)
    impath = str(directory / image.basename)
    align_header = other.astropy_header

    # now get the WCS keys to align the header to
    head = WCS(align_header).to_header(relax=True)

    # and write the results to a file that swarp will read
    extension = f'_aligned_to_{other.basename[:-5]}.remap'

    if persist_aligned:
        outname = image.local_path.replace('.fits', f'{extension}.fits')
    else:
        outname = impath.replace('.fits', f'{extension}.fits')
    headpath = impath.replace('.fits', f'{extension}.head')

    with open(headpath, 'w') as f:
        for card in align_header.cards:
            if card.keyword.startswith('NAXIS'):
                f.write(f'{card.image}\n')
        for card in head.cards:
            f.write(f'{card.image}\n')

    # make a random file for the weightmap -> we dont want to use it
    weightname = directory / image.basename.replace(
        '.fits',
        f'{extension}.weight.fits'
    )

    combtype = 'OR' if isinstance(image, MaskImageBase) else 'CLIPPED'

    syscall = f'swarp -c {conf} {impath} ' \
              f'-BACK_SIZE {BKG_BOX_SIZE} ' \
              f'-IMAGEOUT_NAME {outname} ' \
              f'-NTHREADS {nthreads} ' \
              f'-VMEM_DIR {directory} ' \
              f'-RESAMPLE_DIR {directory} ' \
              f'-SUBTRACT_BACK N ' \
              f'-WEIGHTOUT_NAME {weightname} ' \
              f'-WEIGHT_TYPE NONE ' \
              f'-COMBINE_TYPE {combtype} '

    return syscall, outname, weightname


def run_align(image, other, tmpdir='/tmp',
              nthreads=1, persist_aligned=False):

    from .image import FITSImage

    directory = Path(tmpdir) / uuid.uuid4().hex
    directory.mkdir(exist_ok=True, parents=True)

    command, outname, outweight = prepare_swarp_align(
        image, other,
        directory,
        nthreads=nthreads,
        persist_aligned=persist_aligned
    )

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

    restype = MaskImageBase if isinstance(image, MaskImage) else FITSImage

    result = restype.from_file(outname)
    result.parent_image = image
    weightimage = FITSImage.from_file(outweight)

    if isinstance(image, MaskImage):
        result.update_from_weight_map(weightimage)

    # load everything into memory and unmap if the disk file is going to be
    # deleted
    if not persist_aligned:
        result.load()

        # unmap the object from disk, but preserve the loaded attrs.
        del result._path

    # clean up the swarp working dir
    shutil.rmtree(directory)

    return result


def run_coadd(cls, images, outname, mskoutname, addbkg=True,
              tmpdir='/tmp', sci_swarp_kws=None, mask_swarp_kws=None,
              solve_astrometry=False, swarp_zp_key='MAGZP', scamp_kws=None):
    """Run swarp on images `images`"""

    from .core import ZTFFile
    from .image import FITSImage, CalibratableImageBase
    from .catalog import PipelineFITSCatalog

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

        make_catalog = image.catalog is None
        if not make_catalog:
            shutil.copy(image.catalog.local_path, directory)

        # create the transaction elements
        transact_name = directory / image.basename
        transact_mask_name = directory / image.mask_image.basename
        transact_image = CalibratableImageBase.from_file(transact_name)
        transact_mask = MaskImageBase.from_file(transact_mask_name)
        transact_image.mask_image = transact_mask

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

    transact_outname = f'{directory / os.path.basename(outname))}'
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
    # (and mask) based on the input images
    if all([isinstance(i, ZTFFile) for i in images]):
        for prop in GROUP_PROPERTIES:
            for img in [coadd, coaddmask]:
                setattr(img, prop, getattr(images[0], prop))

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
    return coadd
