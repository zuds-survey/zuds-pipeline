from pathlib import Path
from uuid import uuid4
import shutil
import subprocess
from astropy.io import fits
import warnings
from .utils import initialize_directory
from .env import check_dependencies

SCAMP_CONF = Path(__file__).parent / 'astromatic/default.scamp'


def calibrate_astrometry(images, scamp_kws=None, inplace=False, tmpdir='/tmp'):
    """Derive astrometric solution for input images

    :param catalogs: list of mapped HasWCS --
    the images to astrometrically calibrate
    :param scamp_kws: dict -- configuration parameters to pass to scamp, e.g.,
    {'ASTREF_CATALOG': 'GAIA-DR1'}
    :param inplace: boolean - if true, replace the input image files on disk with
    new files that contain the new WCS information from the astrometric solution,
    and reload those files from disk into memory, else simply copy .head files
    to the image input directories
    :param tmpdir: directory in which to create the temporary directories for
    transactional isolation
    """

    from .image import CalibratableImageBase
    from .file import UnmappedFileError
    from .catalog import PipelineFITSCatalog

    # create a directory for the transaction
    directory = Path(tmpdir) / uuid4().hex
    initialize_directory(directory)

    # make sure all catalogs are mapped
    for image in images:
        if not hasattr(image, 'catalog') or image.catalog is None:
            _ = PipelineFITSCatalog.from_image(image)
        if not 'MJD-OBS' in image.header:
            warnings.warn(f'Image "{image.basename}" header does not contain'
                          f' MJD-OBS keyword, proper motions may not be used'
                          f' in deriving astrometric solution...')

    catalogs = [i.catalog if isinstance(i, CalibratableImageBase)
                else i.parent_image.catalog for i in images]

    catpaths = []
    for catalog in catalogs:
        if not catalog.ismapped:
            raise UnmappedFileError(f'Catalog "{catalog.basename}" '
                                    f'must be mapped.')

        # copy the catalogs to the directory so that .head files are not
        # left in the original directory prematurely
        shutil.copy(catalog.local_path, directory)
        catpaths.append(str(directory / catalog.basename))



    # create the scamp command
    command = f'scamp -c {SCAMP_CONF} '
    if scamp_kws is not None:
        for kw in scamp_kws:
            command += f' -{kw.upper()} {scamp_kws[kw]} '
    command += ' '.join(catpaths)

    # run scamp
    subprocess.check_call(command.split())

    # remove photometric keywords from the headers
    for c in catpaths:
        headpath = f'{c}'.replace('.cat', '.head')
        with open(headpath, 'r') as f:
            lines = f.readlines()

        out = []
        for line in lines:
            key = line[:8].strip()
            if key in ['FLXSCALE', 'MAGZEROP', 'PHOTIRMS',
                       'PHOTINST', 'PHOTLINK', 'COMMENT',
                       'HISTORY']:
                continue
            else:
                out.append(line)
        with open(headpath, 'w') as f:
            f.write(''.join(out))

    # write the result
    mskimgs = [i.mask_image for i in images]
    for imgs in [images, mskimgs]:
        for i, c in zip(imgs, catpaths):
            headpath = f'{c}'.replace('.cat', '.head')
            if inplace:
                with open(headpath, 'r') as f:
                    content = f.read()
                header = fits.Header.fromstring(content, sep='\n')

                for k in dict(header):
                    i.header[k] = header[k]
                    i.header_comments[k] = header.comments[k]

                i.save()
            else:
                shutil.copy(headpath,
                            Path(i.local_path).parent /
                            i.basename.replace('.fits', '.head'))

    shutil.rmtree(directory)
