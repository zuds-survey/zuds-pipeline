import os
import db
import numpy as np
import uuid
import time
from astropy.io import fits
from astropy.wcs import WCS
import galsim
from astropy.time import Time
import logging
from galsim import des
from astropy.convolution import convolve
import shutil
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import paramiko
import subprocess
import tempfile
import archive
import photutils

from seeing import estimate_seeing

def submit_template(variance_dependencies, metatable, nimages=100, start_date=datetime(2017, 12, 10),
                    end_date=datetime(2018, 4, 1), template_science_minsep_days=0, template_destination='.',
                    log_destination='.', job_script_destination=None, task_name=None):

    nersc_account = os.getenv('NERSC_ACCOUNT')
    nersc_username = os.getenv('NERSC_USERNAME')
    nersc_password = os.getenv('NERSC_PASSWORD')
    nersc_host = os.getenv('NERSC_HOST')
    shifter_image = os.getenv('SHIFTER_IMAGE')
    volumes = os.getenv('VOLUMES')

    template_metatable = []
    template_destination = Path(template_destination)
    dependency_dict = {}

    remaining_images = metatable.copy()

    for (field, quadrant, band, ccdnum), group in metatable.groupby(['field',
                                                                     'qid',
                                                                     'filtercode',
                                                                     'ccdid']):
        pass

    template_rows = group.copy()

    if end_date is not None:
        template_rows = template_rows[template_rows['obsdate'] < end_date]

    if start_date is not None:
        template_rows = template_rows[template_rows['obsdate'] > start_date]

    # make cuts on seeing
    template_rows = template_rows[template_rows['seeing'] > 1.8]
    template_rows = template_rows[template_rows['seeing'] < 2.3]

    if len(template_rows) < nimages:
        raise ValueError(f'Not enough images to create requested template (Requested {nimages}, '
                         f'got {len(template_rows)}).')


    template_rows = template_rows.sort_values(by='maglimit', ascending=False)
    template_rows = template_rows.iloc[:nimages]
    template_rows = template_rows.sort_values(by='obsjd')

    if len(variance_dependencies) > 0:
        dependency_list = list(set([variance_dependencies[frame] for frame in template_rows['path']]))
    else:
        dependency_list = []

    jobname = f'ref.{task_name}.{field}.{quadrant}.{band}.{ccdnum}'
    dependency_string = ':'.join(list(map(str, dependency_list)))

    minjd = template_rows.iloc[0]['obsjd']
    maxjd = template_rows.iloc[-1]['obsjd']

    mintime = Time(minjd, format='jd', scale='utc')
    maxtime = Time(maxjd, format='jd', scale='utc')

    mindatestr = mintime.iso.split()[0].replace('-', '')
    maxdatestr = maxtime.iso.split()[0].replace('-', '')


    template_basename = f'{field:06d}_c{ccdnum:02d}_{quadrant:d}_' \
                        f'{band:s}_{mindatestr:s}_{maxdatestr:s}_ztf_deepref.fits'

    template_name = template_destination / template_basename
    template_metatable.append([field, quadrant, band, ccdnum, f'{template_name}'])

    incatstr = ' '.join([p.replace('fits', 'cat') for p in template_rows['full_path']])
    inframestr = ' '.join(template_rows['full_path'])

    ref = db.Reference(field=int(field), filtercode=str(band), qid=int(quadrant), ccdid=int(ccdnum),
                       disk_path=f'{template_name}')
    db.DBSession().add(ref)
    db.DBSession().commit()

    for imid in template_rows['id']:
        im = db.DBSession().query(db.Image).get(int(imid))
        refim = db.ReferenceImage(reference_id=ref.id, imag_id=im.id)
        db.DBSession().add(refim)
    db.DBSession().commit()

    estring = os.getenv("ESTRING").replace(r"\x27", "'")

    jobstr = f'''#!/bin/bash
#SBATCH -N 1
#SBATCH -J {jobname}
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -A {nersc_account}
#SBATCH --partition=realtime
#SBATCH --image={shifter_image}
#SBATCH --dependency=afterok:{dependency_string}
#SBATCH -C haswell
#SBATCH --exclusive
#SBATCH --volume="{volumes}"
#SBATCH -o {Path(log_destination).resolve() / jobname}.out

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

shifter {estring} python /pipeline/bin/makecoadd.py --outfile-path {template_name}  --input-catalogs {incatstr}  --input-frames {inframestr} --template

if [ $? -eq 0 ]; then 

shifter {estring} python /pipeline/bin/log_image.py {template_name} {ref.id} Reference

fi

'''

    if len(dependency_list) == 0:
        jobstr = jobstr.replace('#SBATCH --dependency=afterok:\n', '')

    if job_script_destination is None:
        jobscript = tempfile.NamedTemporaryFile()
    else:
        jobscript = open(Path(job_script_destination / f'{jobname}.sh').resolve(), 'w')

    jobscript.write(jobstr)
    jobscript.seek(0)

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=nersc_host, password=nersc_password, username=nersc_username)

    command = f'sbatch {jobscript.name}'
    stdin, stdout, stderr = ssh_client.exec_command(command)

    if stdout.channel.recv_exit_status() != 0:
        raise RuntimeError(f'SSH Command returned nonzero exit status: {command}')

    out = stdout.read()
    err = stderr.read()

    print(out, flush=True)
    print(err, flush=True)

    jobscript.close()
    ssh_client.close()

    jobid = int(out.strip().split()[-1])

    dependency_dict[f'{template_name}'] = jobid

    indices_left = remaining_images.index.difference(template_rows.index)
    remaining_images = remaining_images.loc[indices_left, :]

    early_enough = remaining_images['obsdate'] < start_date - timedelta(days=template_science_minsep_days)
    late_enough = remaining_images['obsdate'] > end_date + timedelta(days=template_science_minsep_days)

    remaining_images = remaining_images[early_enough | late_enough]
    template_metatable = pd.DataFrame(template_metatable, columns=['field', 'qid', 'filtercode', 'ccdid', 'path'])
    return dependency_dict, remaining_images, ref


import db

CONF_DIR = Path(__file__).parent.parent / 'astromatic/makecoadd'
REF_CONF = CONF_DIR / 'template.swarp'
SCI_CONF = CONF_DIR / 'default.swarp'
MSK_CONF = CONF_DIR / 'mask.swarp'
BKG_VAL = 150. # counts


def initialize_directory(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)


def prepare_swarp_sci(images, outname, directory, copy_inputs=False,
                      reference=False, nthreads=1):
    conf = REF_CONF if reference else SCI_CONF
    initialize_directory(directory)

    if copy_inputs:
        impaths = []
        for image in images:
            shutil.copy(image.local_path, directory)
            impaths.append(str(directory / image.basename))
    else:
        impaths = [im.local_path for im in images]

    # normalize all images to the same zeropoint
    for im, path in zip(images, impaths):
        if 'MAGZP' in im.header:
            fluxscale = 10**(-0.4 * (im.header['MAGZP'] - 25.))
            im.header['FLXSCALE'] = fluxscale
            im.header_comments['FLXSCALE'] = 'Flux scale factor for coadd / DG'
            im.header['FLXSCLZP'] = 25.
            im.header_comments['FLXSCLZP'] = 'FLXSCALE equivalent ZP / DG'
            opath = im.local_path
            im.map_to_local_file(path)
            im.save()
            im.map_to_local_file(opath)

    # if weight images do not exist yet, write them to temporary
    # directory
    wgtpaths = []
    for image in images:
        if not image.weight_image.ismapped or copy_inputs:
            wgtpath = f"{directory / image.basename.replace('.fits', '.weight.fits')}"
            image.weight_image.map_to_local_file(wgtpath)
            image.weight_image.save()
        else:
            wgtpath = image.weight_image.local_path
        wgtpaths.append(wgtpath)

    # get the images in string form
    allwgts = ','.join(wgtpaths)
    allims = ' '.join(impaths)
    wgtout = outname.replace('.fits', '.weight.fits')

    syscall = f'swarp -c {conf} {allims} ' \
              f'-IMAGEOUT_NAME {outname} ' \
              f'-VMEM_DIR {directory} ' \
              f'-RESAMPLE_DIR {directory} ' \
              f'-WEIGHT_IMAGE {allwgts} ' \
              f'-WEIGHTOUT_NAME {wgtout} ' \
              f'-NTHREADS {nthreads}'

    return syscall


def prepare_swarp_mask(masks, outname, mskoutweightname, directory,
                       copy_inputs=False, nthreads=1):
    conf = MSK_CONF
    initialize_directory(directory)

    if copy_inputs:
        for image in masks:
            shutil.copy(image.local_path, directory)

    # get the images in string form
    allims = ' '.join([c.local_path for c in masks])

    syscall = f'swarp -c {conf} {allims} ' \
              f'-IMAGEOUT_NAME {outname} ' \
              f'-VMEM_DIR {directory} ' \
              f'-RESAMPLE_DIR {directory} ' \
              f'-WEIGHTOUT_NAME {mskoutweightname} ' \
              f'-NTHREADS {nthreads}'

    return syscall


def run_coadd(cls, images, outname, mskoutname, reference=False, addbkg=True,
              nthreads=1, tmpdir='/tmp', copy_inputs=False):
    """Run swarp on images `images`"""

    directory = Path(tmpdir) / uuid.uuid4().hex
    directory.mkdir(exist_ok=True, parents=True)

    command = prepare_swarp_sci(images, outname, directory,
                                reference=reference,
                                copy_inputs=copy_inputs,
                                nthreads=nthreads)

    # run swarp
    subprocess.check_call(command.split())

    # now swarp together the masks
    masks = [image.mask_image for image in images]
    mskoutweightname = directory / Path(mskoutname.replace('.fits', '.weight.fits')).name
    command = prepare_swarp_mask(masks, mskoutname, mskoutweightname,
                                 directory, copy_inputs=False,
                                 nthreads=nthreads)

    # run swarp
    subprocess.check_call(command.split())

    # load the result
    coadd = cls.from_file(outname)
    coaddmask = db.MaskImage.from_file(mskoutname)
    coaddweight = db.FloatingPointFITSImage.from_file(mskoutweightname)
    coaddmask.update_from_weight_map(coaddweight)

    # keep a record of the images that went into the coadd
    coadd.input_images = images.tolist()
    coadd.mask_image = coaddmask

    # set the ccdid, qid, field, fid for the coadd
    # (and mask) based on the input images

    for prop in db.GROUP_PROPERTIES:
        for img in [coadd, coaddmask]:
            setattr(img, prop, getattr(images[0], prop))

    if addbkg:
        coadd.data += BKG_VAL

    # save the coadd to disk
    coadd.save()

    # clean up -- this also deletes the mask weight map
    shutil.rmtree(directory)
    return coadd


def ensure_images_have_the_same_properties(images, properties):
    """Raise a ValueError if images have different fid, ccdid, qid, or field."""
    for prop in properties:
        vals = np.asarray([getattr(image, prop) for image in images])
        if not all(vals == vals[0]):
            raise ValueError(f'To be coadded, images must all have the same {prop}. '
                             f'These images had: {[(image.id, getattr(image, prop)) for image in images]}.')

