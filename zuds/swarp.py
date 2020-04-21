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



def prepare_swarp_sci(images, outname, directory, copy_inputs=False,
                      nthreads=1, swarp_kws=None):
    conf = SCI_CONF
    initialize_directory(directory)

    if copy_inputs:
        impaths = []
        for image in images:
            shutil.copy(image.local_path, directory)
            impaths.append(str(directory / image.basename))
    else:
        impaths = [im.local_path for im in images]

    # normalize all images to the same zeropoint
    for im, path in zip(images, impaths):
        if 'MAGZP' in im.header:
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
        if not image.weight_image.ismapped or copy_inputs:
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
              f'-WEIGHTOUT_NAME {wgtout} ' \
              f'-NTHREADS {nthreads} '

    if swarp_kws is not None:
        for kw in swarp_kws:
            syscall += f'-{kw.upper()} {swarp_kws[kw]} '

    return syscall


def prepare_swarp_mask(masks, outname, mskoutweightname, directory,
                       copy_inputs=False, nthreads=1):
    conf = MSK_CONF
    initialize_directory(directory)

    if copy_inputs:
        for image in masks:
            shutil.copy(image.local_path, directory)

    # get the images in string form
    allims = ' '.join([c.local_path for c in masks])

    syscall = f'swarp -c {conf} {allims} ' \
              f'-SUBTRACT_BACK N ' \
              f'-IMAGEOUT_NAME {outname} ' \
              f'-VMEM_DIR {directory} ' \
              f'-RESAMPLE_DIR {directory} ' \
              f'-WEIGHTOUT_NAME {mskoutweightname} ' \
              f'-NTHREADS {nthreads}'

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
              nthreads=1, tmpdir='/tmp', copy_inputs=False, swarp_kws=None):
    """Run swarp on images `images`"""

    from .image import FITSImage

    directory = Path(tmpdir) / uuid.uuid4().hex
    directory.mkdir(exist_ok=True, parents=True)

    command = prepare_swarp_sci(images, outname, directory,
                                copy_inputs=copy_inputs,
                                nthreads=nthreads,
                                swarp_kws=swarp_kws)

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
    masks = [image.mask_image for image in images]
    mskoutweightname = directory / Path(mskoutname.replace('.fits', '.weight.fits')).name
    command = prepare_swarp_mask(masks, mskoutname, mskoutweightname,
                                 directory, copy_inputs=False,
                                 nthreads=nthreads)

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


    # load the result
    coadd = cls.from_file(outname)
    coaddweightname = outname.replace('.fits', '.weight.fits')
    coadd._weightimg = FITSImage.from_file(coaddweightname)

    coaddmask = MaskImage.from_file(mskoutname)
    coaddmaskweight = FITSImage.from_file(mskoutweightname)
    coaddmask.update_from_weight_map(coaddmaskweight)

    # keep a record of the images that went into the coadd
    coadd.input_images = images.tolist()
    coadd.mask_image = coaddmask

    # set the ccdid, qid, field, fid for the coadd
    # (and mask) based on the input images

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
