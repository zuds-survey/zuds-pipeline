import time
import datetime
import stat
import os
from pathlib import Path

from zuds.secrets import get_secret

__all__ = ['ipac_authenticate', 'safe_download']


ipac_root = 'https://irsa.ipac.caltech.edu/'
ipac_username = get_secret('ipac_username')
ipac_password = get_secret('ipac_password')
ipac_cookie = None


def ipac_authenticate():
    import requests
    target = os.path.join(ipac_root, 'account', 'signon', 'login.do')

    r = requests.post(target, data={'josso_username':ipac_username,
                                    'josso_password':ipac_password,
                                    'josso_cmd': 'login'})

    if r.status_code != 200:
        raise ValueError('Unable to Authenticate')

    if r.cookies.get('JOSSO_SESSIONID') is None:
        raise ValueError('Unable to login to IPAC - bad credentials')

    return r.cookies


def download_file(target, destination, cookie):
    import requests
    path = Path(destination)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    dstart = time.time()
    while True:
        try:
            r = requests.get(target, cookies=cookie)
        except Exception as e:
            print(f'Received exception {e}, retrying...')
            time.sleep(1.)
        else:
            break
    dstop = time.time()

    if r.content.startswith(b'<!DOCTYPE'):
        raise requests.RequestException(f'Could not retrieve "{target}": {r.content}')
    else:
        print(f'{datetime.datetime.utcnow()}: Retrieved '
              f'"{target}" in {dstop - dstart:.2f} seconds')

    while True:
        try:
            with open(destination, 'wb') as f:
                f.write(r.content)

            # make group-writable
            st = os.stat(destination)
            os.chmod(destination, st.st_mode | stat.S_IWGRP)
        except OSError:
            continue
        else:
            break


def safe_download(target, destination, cookie, raise_exc=False):
    import requests
    try:
        download_file(target, destination, cookie)
    except requests.RequestException as e:
        print(f'File "{target}" does not exist. Continuing...')
        if raise_exc:
            raise e
