import db
import os
import time
import requests
import logging
import tempfile
import sys
from uuid import uuid4
from libztf import ipac_authenticate
import paramiko
from pathlib import Path

# write images and masks to tape
# write difference images to disk

CHUNK_SIZE = 5000
TAR_SIZE = 2024


# this script does not use shifter
# it is meant to be run on the data transfer nodes


def submit_to_tape(items, tarname):

    cmdlist = Path(os.getenv("STAGING_CMDDIR")) / f'{os.path.basename(tarname)}.cmd'
    cmdlist.parent.mkdir(parents=True, exist_ok=True)

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
        rm $f
    done
    """

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    with tempfile.NamedTemporaryFile() as f:
        f.write(script.encode('ASCII'))
        f.seek(0)

        submit_command = f'PATH="/global/common/cori/software/hypnotoad:/opt/esslurm/bin:$PATH" ' \
                         f'LD_LIBRARY_PATH="/opt/esslurm/lib64" sbatch {f.name}'

        stdin, stdout, stderr = ssh_client.exec_command(submit_command)
        print(stdout.readlines())
        print(stderr.readlines())
        ssh_client.close()


def reset_tarball():
    return list(), f'/nersc/projects/ptf/ztf/{uuid4().hex}.tar'


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
        except OSError:
            continue
        else:
            break


def safe_download(target, destination, cookie, logger):
    try:
        download_file(target, destination, cookie)
    except requests.RequestException as e:
        logger.warning(f'File "{target}" does not exist. Continuing...')


if __name__ == '__main__':

    env, cfg = db.load_env()
    db.init_db(**cfg['database'])

    hostname = os.getenv('HOSTNAME')
    current_tarball, tar_name = reset_tarball()

    tstart = time.time()
    icookie = ipac_authenticate()

    FORMAT = '[%(asctime)-15s]: %(message)s'
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fmter = logging.Formatter(fmt=FORMAT)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmter)
    logger.addHandler(handler)

    while True:

        # check if the cookies need to be reset
        tnow = time.time()
        if tnow - tstart > 86400.:  # assume the cookies expire after 1 day
            tstart = time.time()
            icookie = ipac_authenticate()

        to_download = db.DBSession().query(db.Image).filter(db.sa.or_(
            db.Image.hpss_sci_path == None,
            db.Image.hpss_mask_path == None,
            db.sa.and_(
                db.Image.disk_sub_path == None,
                db.Image.subtraction_exists != False
                ),
            db.sa.and_(
                db.Image.disk_psf_path == None,
                db.Image.subtraction_exists != False
            )
        )
        ).with_for_update(skip_locked=True).order_by(db.Image.field,
                                                     db.Image.ccdid,
                                                     db.Image.qid,
                                                     db.Image.filtercode,
                                                     db.Image.obsjd).limit(CHUNK_SIZE).all()

        if len(to_download) == 0:
            time.sleep(1800.)  # sleep for half an hour
            continue

        for image in to_download:

            if image.hpss_sci_path is None:
                target = image.ipac_path('sciimg.fits')
                destination = image.hpss_staging_path('sciimg.fits')
                safe_download(target, destination, icookie, logger)
                image.hpss_sci_path = tar_name
                current_tarball.append(destination)

            if image.hpss_mask_path is None:
                target = image.ipac_path('mskimg.fits')
                destination = image.hpss_staging_path('mskimg.fits')
                safe_download(target, destination, icookie, logger)
                image.hpss_mask_path = tar_name
                current_tarball.append(destination)

            if image.disk_sub_path is None and image.subtraction_exists != False:
                target = image.ipac_path('scimrefdiffimg.fits.fz')
                destination = image.disk_path('scimrefdiffimg.fits.fz')

                try:
                    download_file(target, destination, icookie)
                except requests.RequestException as e:
                    image.subtraction_exists = False
                    logger.warning(f'File "{target}" does not exist. Continuing...')
                else:
                    image.subtraction_exists = True
                    image.disk_sub_path = destination

            if image.disk_psf_path is None and image.subtraction_exists != False:
                target = image.ipac_path('diffimgpsf.fits')
                destination = image.disk_path('diffimgpsf.fits')
                safe_download(target, destination, icookie, logger)
                image.disk_psf_path = destination

            if len(current_tarball) == TAR_SIZE:
                submit_to_tape(current_tarball, tar_name)
                current_tarball, tar_name = reset_tarball()

        db.DBSession().add_all(to_download)
        db.DBSession().commit()
