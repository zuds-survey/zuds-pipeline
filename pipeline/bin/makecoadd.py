import os
import db
import numpy as np
import uuid
from astropy.time import Time
import shutil
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import paramiko
import subprocess
import tempfile

from utils import initialize_directory

import db

CONF_DIR = Path(__file__).parent.parent / 'astromatic/makecoadd'
REF_CONF = CONF_DIR / 'template.swarp'
SCI_CONF = CONF_DIR / 'default.swarp'
MSK_CONF = CONF_DIR / 'mask.swarp'
BKG_VAL = 150. # counts


def prepare_swarp_sci(images, outname, directory, copy_inputs=False,
                      reference=False, nthreads=1):
    conf = REF_CONF if reference else SCI_CONF
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

    # get the images in string form
    allwgts = ','.join(wgtpaths)
    allims = ' '.join(impaths)
    wgtout = outname.replace('.fits', '.weight.fits')

    syscall = f'swarp -c {conf} {allims} ' \
              f'-IMAGEOUT_NAME {outname} ' \
              f'-VMEM_DIR {directory} ' \
              f'-RESAMPLE_DIR {directory} ' \
              f'-WEIGHT_IMAGE {allwgts} ' \
              f'-WEIGHTOUT_NAME {wgtout} ' \
              f'-NTHREADS {nthreads}'

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
              f'-IMAGEOUT_NAME {outname} ' \
              f'-VMEM_DIR {directory} ' \
              f'-RESAMPLE_DIR {directory} ' \
              f'-WEIGHTOUT_NAME {mskoutweightname} ' \
              f'-NTHREADS {nthreads}'

    return syscall


def run_coadd(cls, images, outname, mskoutname, reference=False, addbkg=True,
              nthreads=1, tmpdir='/tmp', copy_inputs=False):
    """Run swarp on images `images`"""

    directory = Path(tmpdir) / uuid.uuid4().hex
    directory.mkdir(exist_ok=True, parents=True)

    command = prepare_swarp_sci(images, outname, directory,
                                reference=reference,
                                copy_inputs=copy_inputs,
                                nthreads=nthreads)

    # run swarp
    subprocess.check_call(command.split())

    # now swarp together the masks
    masks = [image.mask_image for image in images]
    mskoutweightname = directory / Path(mskoutname.replace('.fits', '.weight.fits')).name
    command = prepare_swarp_mask(masks, mskoutname, mskoutweightname,
                                 directory, copy_inputs=False,
                                 nthreads=nthreads)

    # run swarp
    subprocess.check_call(command.split())

    # load the result
    coadd = cls.from_file(outname)
    coaddmask = db.MaskImage.from_file(mskoutname)
    coaddweight = db.FloatingPointFITSImage.from_file(mskoutweightname)
    coaddmask.update_from_weight_map(coaddweight)

    # keep a record of the images that went into the coadd
    coadd.input_images = images.tolist()
    coadd.mask_image = coaddmask

    # set the ccdid, qid, field, fid for the coadd
    # (and mask) based on the input images

    for prop in db.GROUP_PROPERTIES:
        for img in [coadd, coaddmask]:
            setattr(img, prop, getattr(images[0], prop))

    if addbkg:
        coadd.data += BKG_VAL

    # save the coadd to disk
    coadd.save()

    # clean up -- this also deletes the mask weight map
    shutil.rmtree(directory)
    return coadd


def ensure_images_have_the_same_properties(images, properties):
    """Raise a ValueError if images have different fid, ccdid, qid, or field."""
    for prop in properties:
        vals = np.asarray([getattr(image, prop) for image in images])
        if not all(vals == vals[0]):
            raise ValueError(f'To be coadded, images must all have the same {prop}. '
                             f'These images had: {[(image.id, getattr(image, prop)) for image in images]}.')

