import os
import sys
import time
import requests
import logging
from pathlib import Path

from sqlalchemy import and_
from db import DBSession, Image, init_db

suffix_dict = {'sub': 'scimrefdiffimg.fits.fz',
               'sci': 'sciimg.fits',
               'mask': 'mskimg.fits',
               'psf': 'diffimgpsf.fits'}

ipac_username = os.getenv('IPAC_USERNAME')
ipac_password = os.getenv('IPAC_PASSWORD')
ipac_root = 'https://irsa.ipac.caltech.edu/'

# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []

CHUNK_SIZE = 1024


def ipac_authenticate():
    target = os.path.join(ipac_root, 'account', 'signon', 'login.do')

    while True:
        try:
            r = requests.post(target, data={'josso_username': ipac_username, 'josso_password': ipac_password,
                                            'josso_cmd': 'login'})
        except Exception as e:
            print(f'Got exception {e} trying to connect to IPAC, retrying...')
            time.sleep(1)
        else:
            break

    if r.status_code != 200:
        raise ValueError('Unable to Authenticate')

    if r.cookies.get('JOSSO_SESSIONID') is None:
        raise ValueError('Unable to login to IPAC - bad credentials')

    return r.cookies


class IPACQueryManager(object):

    def __init__(self, logger):
        self.logger = logger
        self.cookie = ipac_authenticate()
        self.start_time = time.time()

    def __call__(self, nchunks, mychunk, imagetypes=('psf', 'sub')):

        counter = 0
        for itype in imagetypes:

            hpss = getattr(Image, f'hpss_{itype}_path')
            disk = getattr(Image, f'disk_{itype}_path')

            images = DBSession().query(Image) \
                                .filter(and_(hpss == None, disk == None, Image.ipac_gid == 2)) \
                                .order_by(Image.field, Image.ccdid, Image.qid, Image.filtercode, Image.obsjd) \
                                .all()

            my_images = _split(images, nchunks)[mychunk - 1]
            suffix = suffix_dict[itype]

            for image in my_images:
                ipac_path = image.ipac_path(suffix)
                disk_path = image.disk_path(suffix)
                dp = Path(disk_path)
                dp.parent.mkdir(parents=True, exist_ok=True)

                while True:
                    now = time.time()
                    dt = now - self.start_time
                    if dt > 86400.:
                        self.start_time = time.time()
                        self.cookie = ipac_authenticate()
                    try:
                        r = requests.get(ipac_path, cookies=self.cookie)
                    except Exception as e:
                        self.logger.info(f'Received exception {e}, retrying...')
                        time.sleep(1.)
                    else:
                        break

                with open(disk_path, 'wb') as f:
                    f.write(r.content)
                    logger.info(f'Retrieved {ipac_path}')
                    counter += 1

                setattr(image, f'disk_{itype}_path', disk_path)

                DBSession().add(image)

                if counter == CHUNK_SIZE:
                    DBSession().commit()
                    counter = 0

            DBSession().commit()


if __name__ == '__main__':

    init_db()

    logger = logging.getLogger('poll')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    logger.addHandler(ch)

    hostname = os.getenv('HOSTNAME')

    formatter = logging.Formatter(f'[%(asctime)s - {hostname} - %(levelname)s] - %(message)s')
    ch.setFormatter(formatter)

    nchunks = 4
    mychunk = int(hostname.split('.')[0][-2:])
    manager = IPACQueryManager(logger)
    manager(nchunks, mychunk)
