import db
import os
import time
import stat
import requests
import logging
import sys
from uuid import uuid4
import subprocess
from pathlib import Path
from sqlalchemy.sql.expression import case

from secrets import get_secret

# read/write for group
os.umask(0o007)

# write images and masks to tape
# write difference images to disk

CHUNK_SIZE = 512
TAR_SIZE = 2048

# this script does not use shifter
# it is meant to be run on the data transfer nodes

ipac_root = 'https://irsa.ipac.caltech.edu/'
ipac_username = get_secret('ipac_username')
ipac_password = get_secret('ipac_password')


def ipac_authenticate():
    target = os.path.join(ipac_root, 'account', 'signon', 'login.do')

    r = requests.post(target, data={'josso_username':ipac_username,
                                    'josso_password':ipac_password,
                                    'josso_cmd': 'login'})

    if r.status_code != 200:
        raise ValueError('Unable to Authenticate')

    if r.cookies.get('JOSSO_SESSIONID') is None:
        raise ValueError('Unable to login to IPAC - bad credentials')

    return r.cookies


def submit_to_tape(tape_archive):

    tarname = tape_archive.id

    cmdlist = Path(get_secret("staging_cmddir")) / \
              f'{os.path.basename(tarname)}.cmd'
    cmdlist.parent.mkdir(parents=True, exist_ok=True)

    items = [copy.member_name for copy in archive.contents]
    with open(cmdlist, 'w') as f:
        for destination in items:
            f.write(f'{destination}\n')

    script = f"""#!/bin/bash
#SBATCH -q xfer
#SBATCH -N 1
#SBATCH -A {os.getenv("NERSC_ACCOUNT")}
#SBATCH -t 48:00:00
#SBATCH -L project,SCRATCH
#SBATCH -C haswell
#SBATCH -J {Path(tarname).name}

/usr/common/mss/bin/htar cvf {tarname} -L {cmdlist}
for f in `cat {cmdlist}`; do
    rm -v $f
done
"""
    shfile = f'{cmdlist.resolve()}.sh'
    with open(shfile, 'w') as f:
        f.write(script)

    subfile = shfile.replace('.sh', '.sub.sh')
    with open(subfile, 'w') as f:
        f.write(f'''#!/bin/bash
        module load esslurm
        sbatch {Path(shfile).resolve()}'''
    )

    subprocess.check_call(f'bash {subfile}'.split())


def reset_tarball():
    archive = db.TapeArchive(
        id=f'/nersc/projects/ptf/ztf/{uuid4().hex}.tar'
    )
    return list(), archive


def download_file(target, destination, cookie):

    path = Path(destination)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    while True:
        try:
            r = requests.get(target, cookies=cookie)
        except Exception as e:
            print(f'Received exception {e}, retrying...')
            time.sleep(1.)
        else:
            break

    if r.content.startswith(b'<!DOCTYPE'):
        raise requests.RequestException(f'Could not retrieve "{target}": {r.content}')
    else:
        print(f'Retrieved "{target}"')

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


def safe_download(target, destination, cookie, logger):
    try:
        download_file(target, destination, cookie)
    except requests.RequestException as e:
        logger.warning(f'File "{target}" does not exist. Continuing...')


ZUDS_FIELDS = [523,524,574,575,576,623,624,625,626,
               627,670,671,672,673,674,675,714,715,
               716,717,718,719,754,755,756,757,758,
               759,789,790,791,792,793,819,820,821,
               822,823,843,844,845,846,861,862,863]


fmap = {
    1: 'zg',
    2: 'zr',
    3: 'zi'
}

if __name__ == '__main__':

    hostname = os.getenv('HOSTNAME')
    current_tarball, archive = reset_tarball()

    tstart = time.time()
    icookie = ipac_authenticate()

    FORMAT = '[%(asctime)-15s]: %(message)s'
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fmter = logging.Formatter(fmt=FORMAT)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmter)
    logger.addHandler(handler)

    # download the partnership images first, caltech second, public last
    _gid_priorities = {2: 1, 3: 2, 1: 3}
    sort_order = case(value=db.ScienceImage.ipac_gid, whens=_gid_priorities)

    while True:

        # check if the cookies need to be reset
        tnow = time.time()
        if tnow - tstart > 86400.:  # assume the cookies expire after 1 day
            tstart = time.time()
            icookie = ipac_authenticate()

        idownload_q = db.DBSession().query(db.ZTFFile).outerjoin(
            db.TapeCopy, db.ZTFFile.id == db.TapeCopy.product_id
        ).outerjoin(
            db.HTTPArchiveCopy, db.ZTFFile.id == db.HTTPArchiveCopy.product_id
        ).filter(
            db.sa.or_(
                db.ZTFFile.basename.ilike('ztf%sciimg.fits'),
                db.ZTFFile.basename.ilike('ztf%mskimg.fits'),
            ),
            db.ZTFFile.field.in_(ZUDS_FIELDS),
            db.HTTPArchiveCopy.product_id == None,
            db.TapeCopy.product_id == None
        ).with_for_update(skip_locked=True, of=db.ZTFFile).order_by(
            db.ZTFFile.field,
            db.ZTFFile.ccdid,
            db.ZTFFile.qid,
            db.ZTFFile.fid
        ).limit(CHUNK_SIZE)

        to_download = idownload_q.all()

        if len(to_download) == 0:
            time.sleep(180.)  # sleep for 3 minutes
            continue

        for image in to_download:

            if image.type == 'sci':
                target = image.ipac_path('sciimg.fits')
            else:
                # it's a mask
                target = image.parent_image.ipac_path('mskimg.fits')

            # ensure this has 12 components so that it can be used with
            # retrieve
            destination_base = Path(
                f'/global/cscratch1/sd/dgold/zuds/data/xfer/{hostname}/'
                f'{image.field:06d}/'
                f'c{image.ccdid:02d}/'
                f'q{image.qid}/'
                f'{fmap[image.fid]}'
            )

            destination = f'{destination_base / image.basename}'
            safe_download(target, destination, icookie, logger)
            image.map_to_local_file(destination)

            # associate it with the tape archive
            tcopy = db.TapeCopy(
                member_name=destination,
                archive=archive,
                product=image
            )

            # and archive the file to disk
            acopy = db.HTTPArchiveCopy.from_product(image)
            acopy.put()
            db.DBSession().add(acopy)

            if len(current_tarball) >= TAR_SIZE:
                # write the archive to the database
                db.DBSession().add(archive)

                size = 0
                for file in archive.contents:
                    size += os.path.getsize(file.local_path)

                archive.size = size

                # submit it to tape
                submit_to_tape(archive)
                current_tarball, tar_name = reset_tarball()

        db.DBSession().add_all(to_download)
        db.DBSession().commit()
