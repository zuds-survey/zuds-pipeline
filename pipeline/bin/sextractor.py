import uuid
import shutil
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


def prepare_sextractor(image, checkimage_types=None, catalog_type='FITS_LDAC'):
    """Set up the pipeline to do a run of source extractor."""

    conf = SEX_CONF
    valid_types = ['rms', 'segm', 'bkgsub', 'bkg']

    # poor man's atleast_1d
    if checkimage_types == 'all':
        checkimage_types = valid_types
    checkimage_types = checkimage_types or []

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
              f'-BACK_SIZE {BACKGROUND_BOXSIZE} ' \
              f'-PARAMETERS_NAME {PARAM_FILE} ' \
              f'-NNW_NAME {NNW_FILE} ' \
              f'-FILTER_NAME {CONV_FILE} ' \

    outnames = [outname] + coutnames

    return syscall, outnames


def run_sextractor(image, checkimage_types=None, catalog_type='FITS_LDAC'):
    """Run SExtractor on """

    import db

    command, outnames = prepare_sextractor(image,
                                           checkimage_types=checkimage_types,
                                           catalog_type=catalog_type)
    # run it
    subprocess.check_call(command.split())

    # load up the results into objects

    result = []
    for name in outnames:
        if name.endswith('.cat'):
            product = db.PipelineFITSCatalog.from_file(name)
        elif name.endswith('.segm.fits'):
            product = db.SegmentationImage.from_file(name)
        else:
            product = db.FloatingPointFITSImage.from_file(name)

        product.field = image.field
        product.ccdid = image.ccdid
        product.qid = image.qid
        product.fid = image.fid
        result.append(product)

    return result
