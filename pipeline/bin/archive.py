import db
import os
import stat
import shutil
from pathlib import Path


fid_map = {
    1: 'zg',
    2: 'zr',
    3: 'zi'
}

perm = 0o755


def _mkdir_recursive(path):
    sub_path = os.path.dirname(path)
    if not os.path.exists(sub_path):
        _mkdir_recursive(sub_path)
    if not os.path.exists(path):
        os.mkdir(path, mode=perm)


def archive(product):
    """Publish a PipelineFITSProduct to the NERSC archive."""

    if not isinstance(product, db.PipelineProduct):
        raise ValueError(f'Cannot archive object "{product}", must be an instance of'
                         f'PipelineFITSProduct.')

    field = product.field
    qid = product.qid
    ccdid = product.ccdid
    fid = product.fid
    band = fid_map[fid]

    path = Path(db.NERSC_PREFIX) / f'{field:06d}/' \
                                   f'c{ccdid:02d}/' \
                                   f'q{qid}/' \
                                   f'{band}/' \
                                   f'{product.basename}'

    product.archive_path = f'{path.absolute()}'
    product.url = f'{path.absolute()}'.replace(db.NERSC_PREFIX, db.URL_PREFIX)

    if os.getenv('NERSC_HOST') == 'cori':
        if not path.parent.exists():
            _mkdir_recursive(path.parent)
        shutil.copy(product.local_path, path)
        os.chmod(path, perm)
    else:
        product.put()
