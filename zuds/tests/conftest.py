import os
import warnings
import pytest
import pathlib
import requests
from sqlalchemy import exc as sa_exc
from zuds.tests.fixtures import (ScienceImageFactory,
                                 TriangulumScienceImageFactory,
                                 SourceFactory,
                                 TMP_DIR)
import uuid
import zuds
import shutil

fnames = ['ztf_20200531319167_000718_zr_c01_o_q1_sciimg_ra213.5613_dec38.2261_asec500.fits',
          'ztf_20200601248704_000718_zr_c01_o_q1_sciimg_ra213.5613_dec38.2261_asec500.fits',
          'ztf_20200604255023_000718_zr_c01_o_q1_sciimg_ra213.5613_dec38.2261_asec500.fits']
mnames = [f.replace('sciimg', 'mskimg') for f in fnames]

URLS = [f'https://portal.nersc.gov/cfs/astro250/zuds/{f}' for f in fnames]
MURLS = [u.replace('sciimg', 'mskimg') for u in URLS]


# initialize the database
print('Loading test configuration from test_config.yaml')
basedir = pathlib.Path(os.path.dirname(__file__))
config = basedir / '../config/test.conf.yaml'
target = pathlib.Path(TMP_DIR) / config.name
shutil.copy(config, target)
os.chmod(target, 0o700)
zuds.load_config(target)
zuds.create_database()

zuds.init_db()
# ensure the database is empty when the test suite starts
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=sa_exc.SAWarning)
    zuds.drop_tables()

zuds.create_tables()


def _get_mask(url):
    r = requests.get(url)
    outname = pathlib.Path(TMP_DIR) / url.split('/')[-1]
    outname = str(outname.absolute()).replace(
        '.fits', f'.{uuid.uuid4().hex}.fits'
    )
    with open(outname, 'wb') as f:
        f.write(r.content)

    return zuds.MaskImage.from_file(outname)


def _get_sci(url, mask):
    r = requests.get(url)
    outname = pathlib.Path(TMP_DIR) / url.split('/')[-1]
    outname = str(outname.absolute()).replace(
        '.fits', f'.{uuid.uuid4().hex}.fits'
    )
    mskpath = mask.local_path
    with open(outname, 'wb') as f:
        f.write(r.content)
    s = zuds.ScienceImage.from_file(outname, load_others=False)
    s.mask_image = mask
    mask.map_to_local_file(mskpath)
    return s


@pytest.fixture
def mask_image_data_20200531():
    return _get_mask(MURLS[0])


@pytest.fixture
def sci_image_data_20200531(mask_image_data_20200531):
    return _get_sci(URLS[0], mask_image_data_20200531)


@pytest.fixture
def mask_image_data_20200601():
    return _get_mask(MURLS[1])


@pytest.fixture
def sci_image_data_20200601(mask_image_data_20200601):
    return _get_sci(URLS[1], mask_image_data_20200601)


@pytest.fixture
def mask_image_data_20200604():
    return _get_mask(MURLS[2])


@pytest.fixture
def sci_image_data_20200604(mask_image_data_20200604):
    return _get_sci(URLS[2], mask_image_data_20200604)


@pytest.fixture
def refimg_data_first2_imgs(sci_image_data_20200531, sci_image_data_20200601):
    imgs = [sci_image_data_20200531, sci_image_data_20200601]
    out = os.path.join(TMP_DIR, f'ref-{uuid.uuid4().hex}.fits')
    ref = zuds.ReferenceImage.from_images(imgs, out)
    return ref


@pytest.fixture
def science_image():
    return ScienceImageFactory(basename=uuid.uuid4().hex)


@pytest.fixture
def triangulum_science_image():
    return TriangulumScienceImageFactory(basename=uuid.uuid4().hex)


@pytest.fixture
def source():
    return SourceFactory()

