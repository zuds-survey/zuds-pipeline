import numpy as np
import subprocess
from pathlib import Path
import shutil
import uuid
import db

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



def prepare_sextractor(image, directory, checkimage_type=None,
                       catalog_type='FITS_LDAC'):

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

    hasweight = hasattr(image, '_rmsimg')

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

    delnames = []
    if hasweight:
        image.weight_image.save()
        syscall += f'-WEIGHT_IMAGE {image.weight_image.local_path} ' \
                   f'-WEIGHT_TYPE MAP_WEIGHT'
    else:
        # we will use the (inverted) bad pixel mask as the initial weight map
        #  if a weight map is not specified. this ensures that sextractor
        # does not use masked pixels in its estimates of the background, etc.

        # this means the weight map will have weight=0 for masked pixels and
        # weight = 1 for unmasked pixels

        bpmweight = db.FITSImage()
        bpmweight.data = (~image.mask_image.boolean.data).astype(float)
        bpmweight.header = image.mask_image.header
        bpmweight.header_comments= image.mask_image.header_comments
        bpmweight.basename = image.basename.replace('.fits', '.bpmweight.fits')
        bpmwpath = f'{(directory / bpmweight.basename).absolute()}'
        bpmweight.map_to_local_file(bpmwpath)
        bpmweight.save()

        syscall += f'-WEIGHT_IMAGE {bpmweight.local_path} ' \
                   f'-WEIGHT_TYPE MAP_WEIGHT'

    outnames = [outname] + coutnames

    return syscall, outnames


def run_sextractor(image, checkimage_type=None, catalog_type='FITS_LDAC',
                   tmpdir='/tmp'):
    """Run SExtractor on an image and produce the requested checkimages and
    catalogs, returning the results as ZUDS objects (potentially DB-backed)."""

    directory = Path(tmpdir) / uuid.uuid4().hex
    directory.mkdir(exist_ok=True, parents=True)

    command, outnames = prepare_sextractor(
        image, directory, checkimage_type=checkimage_type,
        catalog_type=catalog_type
    )

    # run it
    subprocess.check_call(command.split())

    # load up the results into objects

    result = []
    for name in outnames:
        if name.endswith('.cat'):
            product = db.PipelineFITSCatalog.from_file(name,
                                                       use_existing_record=True)
        else:
            product = db.FITSImage.from_file(
                name, use_existing_record=True
            )

        product.field = image.field
        product.ccdid = image.ccdid
        product.qid = image.qid
        product.fid = image.fid
        result.append(product)

    shutil.rmtree(directory)
    return result
