import db
import os
import stat
import shutil
from pathlib import Path
from secrets import get_secret
import requests


perm = 0o755


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

    if not isinstance(copy, db.HTTPArchiveCopy):
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


