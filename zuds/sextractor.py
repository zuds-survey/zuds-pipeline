import numpy as np
import subprocess
from pathlib import Path
import shutil
import uuid

import os
from astropy.io import fits

from .core import ZTFFile
from .constants import BKG_BOX_SIZE, MASK_BORDER

__all__ = ['prepare_sextractor', 'run_sextractor']


SEX_CONF = Path(__file__).parent / 'astromatic/sextractor.conf'
PARAM_FILE = Path(__file__).parent / 'astromatic/sextractor.param'
NNW_FILE = Path(__file__).parent / 'astromatic/default.nnw'
CONV_FILE = Path(__file__).parent / 'astromatic/default.conv'


checkimage_map = {
    'rms': 'BACKGROUND_RMS',
    'segm': 'SEGMENTATION',
    'bkgsub': '-BACKGROUND',
    'bkg': 'BACKGROUND'
}


def prepare_sextractor(image, directory, checkimage_type=None,
                       catalog_type='FITS_LDAC', use_weightmap=True):
    """Set up the pipeline to do a run of source extractor."""
    from .mask import BAD_SUM

    conf = SEX_CONF
    valid_types = ['rms', 'segm', 'bkgsub', 'bkg']

    checkimage_type = checkimage_type or []
    checkimage_types = np.atleast_1d(checkimage_type).tolist()

    # poor man's atleast_1d
    if 'all' in checkimage_types:
        checkimage_types = valid_types

    for t in checkimage_types:
        if t not in valid_types:
            raise ValueError(f'Invalid CHECKIMAGE_TYPE "{t}". Must be one of '
                             f'{valid_types}.')

    # assume the newest catalog is always the one we want
    # TODO think about versioning /database collision issues

    impath = image.local_path
    outname = image.local_path.replace('.fits', '.cat')
    coutnames = [impath.replace('.fits', f'.{t}.fits') for t in
                 checkimage_types]

    if len(checkimage_types) > 0:
        ctypestr = ','.join([checkimage_map[t] for t in checkimage_types])
        cnamestr = ','.join(coutnames)
    else:
        ctypestr = cnamestr = 'None'


    syscall = f'sex -c {conf} {impath} ' \
              f'-CHECKIMAGE_TYPE {ctypestr} ' \
              f'-CHECKIMAGE_NAME {cnamestr} ' \
              f'-CATALOG_NAME {outname} ' \
              f'-CATALOG_TYPE {catalog_type} ' \
              f'-BACK_SIZE {BKG_BOX_SIZE} ' \
              f'-PARAMETERS_NAME {PARAM_FILE} ' \
              f'-STARNNW_NAME {NNW_FILE} ' \
              f'-FILTER_NAME {CONV_FILE} ' \
              f'-FLAG_IMAGE {image.mask_image.local_path} '

    if not use_weightmap:
        # make a false weightmap so that masked pixels are excluded from
        # background statistics

        imgbase = os.path.basename(impath)
        weightname = directory / imgbase.replace('.fits', '.false.weight.fits')
        falseweight = np.ones_like(image.mask_image.data)
        falseweight[(image.mask_image.data & BAD_SUM) > 0] = 0

        if image.basename.endswith('sciimg.fits'):
            falseweight[:MASK_BORDER] = 0
            falseweight[-MASK_BORDER:] = 0
            falseweight[:, :MASK_BORDER] = 0
            falseweight[:, -MASK_BORDER:] = 0

        fits.writeto(weightname, data=falseweight.astype('<f4'))
        syscall += f'-WEIGHT_IMAGE {weightname} -WEIGHT_TYPE MAP_WEIGHT'

    else:
        syscall += f'-WEIGHT_IMAGE {image.weight_image.local_path} ' \
                   f'-WEIGHT_TYPE MAP_WEIGHT'

    outnames = [outname] + coutnames

    return syscall, outnames


def run_sextractor(image, checkimage_type=None, catalog_type='FITS_LDAC',
                   tmpdir='/tmp', use_weightmap=True):
    """Run SExtractor on an image and produce the requested checkimages and
    catalogs, returning the results as ZUDS objects (potentially DB-backed)."""

    from .image import FITSImage
    from .catalog import PipelineFITSCatalog

    directory = Path(tmpdir) / uuid.uuid4().hex
    directory.mkdir(exist_ok=True, parents=True)

    command, outnames = prepare_sextractor(
        image, directory, checkimage_type=checkimage_type,
        catalog_type=catalog_type, use_weightmap=use_weightmap
    )

    # run it
    subprocess.check_call(command.split())

    # load up the results into objects

    result = []
    for name in outnames:
        if name.endswith('.cat'):
            product = PipelineFITSCatalog.from_file(name,
                                                    use_existing_record=True)
        else:
            product = FITSImage.from_file(
                name, use_existing_record=True
            )

        if isinstance(image, ZTFFile):
            product.field = image.field
            product.ccdid = image.ccdid
            product.qid = image.qid
            product.fid = image.fid
        result.append(product)

    shutil.rmtree(directory)
    return result
