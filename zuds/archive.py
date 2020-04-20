import os
import shutil
from pathlib import Path

import requests

from .core import Base, ZTFFile, DBSession
from .secrets import get_secret
from .utils import fid_map

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy.dialects import postgresql as psql
from sqlalchemy.exc import UnboundExecutionError

perm = 0o755

NERSC_PREFIX = '/global/cfs/cdirs/m937/www/data'
URL_PREFIX = 'https://portal.nersc.gov/cfs/m937'
STAMP_PREFIX = '/global/cfs/cdirs/m937/www'

__all__ = ['ZTFFileCopy', 'HTTPArchiveCopy', 'TapeCopy',
           'archive']

class ZTFFileCopy(Base):
    """Record of a *permanent* (i.e., will not be deleted or modified,
    or will only be touched extremely rarely and by someone who can do it in
    concert with the database).

    A Copy is diferent from a File in that a copy cannot be mapped to a file
    on local disk, whereas a File is mappable. A copy is just a record of a
    file that lives in a permanent place somewhere."""

    __tablename__ = 'ztffilecopies'

    __mapper_args__ = {
        'polymorphic_on': 'type',
        'polymorphic_identity': 'copy'
    }

    type = sa.Column(sa.Text)
    product_id = sa.Column(sa.Integer, sa.ForeignKey('ztffiles.id',
                                                     ondelete='CASCADE'),
                           index=True)
    product = relationship('ZTFFile', back_populates='copies',
                           cascade='all')

    def get(self):
        """Pull the Copy to local disk and return the corresponding
        Product (File subclass) that the Copy is mapped to."""
        raise NotImplemented


class HTTPArchiveCopy(ZTFFileCopy):
    """Record of a copy of a ZTFFile that lives on the ZUDS disk
    archive at NERSC (on project) and is accessible via HTTP."""

    __mapper_args__ = {
        'polymorphic_identity': 'http'
    }

    __tablename__ = 'httparchivecopies'

    id = sa.Column(sa.Integer, sa.ForeignKey('ztffilecopies.id',
                                             ondelete='CASCADE'),
                   primary_key=True)

    url = sa.Column(sa.Text, index=True, unique=True)
    archive_path = sa.Column(sa.Text)

    def get(self):
        product = self.product
        with open(product.basename, 'wb') as f:
            r = requests.get(self.url)
            r.raise_for_status()
            f.write(r.content)
            product.map_to_local_file(product.basename)
        return product

    def put(self):
        archive(self)

    @classmethod
    def from_product(cls, product, check=True):
        if not isinstance(product, ZTFFile):
            raise ValueError(
                f'Cannot archive object "{product}", must be an instance of'
                f'PipelineFITSProduct.')

        field = product.field
        qid = product.qid
        ccdid = product.ccdid
        fid = product.fid
        band = fid_map[fid]

        path = Path(NERSC_PREFIX) / f'{field:06d}/' \
                                    f'c{ccdid:02d}/' \
                                    f'q{qid}/' \
                                    f'{band}/' \
                                    f'{product.basename}'

        archive_path = f'{path.absolute()}'
        url = f'{path.absolute()}'.replace(NERSC_PREFIX[:-5], URL_PREFIX)

        # check to see if a copy with this URL already exists.
        # if so return it


        if check:
            try:
                old = DBSession().query(cls).filter(
                    cls.url == url
                ).first()
            except UnboundExecutionError:
                # no database
                old = None

        else:
            old = None

        if old is None:
            copy = cls()
            copy.archive_path = archive_path
            copy.url = url
            copy.product = product
            return copy

        else:
            return old


class TapeCopy(ZTFFileCopy):
    """Record of a copy of a ZTFFile that lives inside a
    tape archive on HPSS."""

    __tablename__ = 'tapecopies'

    __mapper_args__ = {
        'polymorphic_identity': 'tape'
    }

    id = sa.Column(sa.Integer, sa.ForeignKey('ztffilecopies.id',
                                             ondelete='CASCADE'),
                   primary_key=True)

    archive_id = sa.Column(sa.Text, sa.ForeignKey('tapearchives.id',
                                                  ondelete='CASCADE'),
                           index=True)
    archive = relationship('TapeArchive', back_populates='contents')

    # The exact name of the TAR archive member
    member_name = sa.Column(sa.Text)


class TapeArchive(Base):
    """Record of a tape archive that contains copies of ZTFFiles."""
    id = sa.Column(sa.Text, primary_key=True)
    contents = relationship('TapeCopy', cascade='all')
    size = sa.Column(psql.BIGINT)  # size of the archive in bytes

    @classmethod
    def from_directories(cls, path):
        raise NotImplemented

    def get(self, extract=False, extract_kws=None):
        # use retrieve.retrieve for now
        raise NotImplemented


def _mkdir_recursive(path):
    sub_path = os.path.dirname(path)
    if not os.path.exists(sub_path):
        _mkdir_recursive(sub_path)
    if not os.path.exists(path):
        os.mkdir(path)
        os.chmod(path, perm)


def archive_copy_over_http(copy):
    product = copy.product
    product.save()
    # authenticate to nersc system
    target = 'https://newt.nersc.gov/newt/login'
    username = get_secret('nersc_username')
    password = get_secret('nersc_password')
    r = requests.post(target, data={
        'username': username,
        'password': password
    })
    r.raise_for_status()
    auth_cookie = r.cookies

    # prepare the destination directory to receive the file
    target = f'https://newt.nersc.gov/newt/command/cori'
    cmd = f'/usr/bin/mkdir -p {os.path.dirname(copy.archive_path)}'
    loginenv = False
    r = requests.post(target, data={
        'executable': cmd,
        'loginenv': loginenv
    }, cookies=auth_cookie)
    r.raise_for_status()

    # upload the file, delete leading "/" for newt
    target = f'https://newt.nersc.gov/newt/file/cori/' \
             f'{str(copy.archive_path)[1:]}'
    with open(product.local_path, 'rb') as f:
        contents = f.read()
    r = requests.put(target, data=contents, cookies=auth_cookie)
    resp = r.json()
    if not resp['status'] == 'OK':
        raise requests.RequestException(resp)


def archive(copy):
    """Publish a copy of a ZTFFile to the NERSC archive."""

    if not isinstance(copy, HTTPArchiveCopy):
        raise ValueError(
            f'Cannot archive object "{copy.basename}", must be an instance of'
            f'Copy.')

    path = Path(copy.archive_path)
    product = copy.product

    if os.getenv('NERSC_HOST') == 'cori':
        if not path.parent.exists():
            _mkdir_recursive(path.parent)
        shutil.copy(product.local_path, path)
        os.chmod(path, perm)
    else:
        archive_copy_over_http(copy)


