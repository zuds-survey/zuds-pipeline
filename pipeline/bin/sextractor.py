import numpy as np
import subprocess
from pathlib import Path

from utils import initialize_directory

SEX_CONF = Path(__file__).parent.parent / 'astromatic/sextractor.conf'
PARAM_FILE = Path(__file__).parent.parent / 'astromatic/sextractor.param'
NNW_FILE = Path(__file__).parent.parent / 'astromatic/default.nnw'
CONV_FILE = Path(__file__).parent.parent / 'astromatic/default.conv'
BACKGROUND_BOXSIZE = 384  # pixels



checkimage_map = {
    'rms': 'BACKGROUND_RMS',
    'segm': 'SEGMENTATION',
    'bkgsub': '-BACKGROUND',
    'bkg': 'BACKGROUND'
}


def prepare_sextractor(image, checkimage_type=None, catalog_type='FITS_LDAC'):
    """Set up the pipeline to do a run of source extractor."""

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

    useweight = hasattr(image, '_rmsimg')

    syscall = f'sex -c {conf} {impath} ' \
              f'-CHECKIMAGE_TYPE {ctypestr} ' \
              f'-CHECKIMAGE_NAME {cnamestr} ' \
              f'-CATALOG_NAME {outname} ' \
              f'-CATALOG_TYPE {catalog_type} ' \
              f'-BACK_SIZE {BACKGROUND_BOXSIZE} ' \
              f'-PARAMETERS_NAME {PARAM_FILE} ' \
              f'-STARNNW_NAME {NNW_FILE} ' \
              f'-FILTER_NAME {CONV_FILE} ' \
              f'-FLAG_IMAGE {image.mask_image.local_path} ' \

    if useweight:
        image.weight_image.save()
        syscall += f'-WEIGHT_IMAGE {image.weight_image.local_path} ' \
                   f'-WEIGHT_TYPE MAP_WEIGHT'

    outnames = [outname] + coutnames

    return syscall, outnames


def run_sextractor(image, checkimage_type=None, catalog_type='FITS_LDAC'):
    """Run SExtractor on an image and produce the requested checkimages and
    catalogs, returning the results as ZUDS objects (potentially DB-backed)."""

    import db

    command, outnames = prepare_sextractor(image,
                                           checkimage_type=checkimage_type,
                                           catalog_type=catalog_type)
    # run it
    subprocess.check_call(command.split())

    # load up the results into objects

    result = []
    for name in outnames:
        if name.endswith('.cat'):
            product = db.PipelineFITSCatalog.from_file(name,
                                                       use_existing_record=True)
        else:
            product = db.IntegerFITSImage.from_file(
                name, use_existing_record=True
            )

        product.field = image.field
        product.ccdid = image.ccdid
        product.qid = image.qid
        product.fid = image.fid
        result.append(product)

    return result
