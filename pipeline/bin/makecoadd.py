import os
import db
import numpy as np
import libztf
import uuid
import time
from astropy.io import fits
from astropy.wcs import WCS
import galsim
from makevariance import make_variance
from astropy.time import Time
import logging
from galsim import des
from astropy.convolution import convolve
import shutil
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import paramiko
import tempfile


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
    template_rows = template_rows[template_rows['seeing'] < 3.0]

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

shifter python /pipeline/bin/makecoadd.py --outfile-path "{template_name}"  --input-catalogs "{incatstr}"  --input-frames "{inframestr}" --template

shifter python /pipeline/bin/log_image.py {template_name} {ref.id} Reference

'''

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


if __name__ == '__main__':

    import argparse

    # set up the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile-path', dest='outfile_path', required=True,
                        help='Basename of output coadd.', nargs=1)
    parser.add_argument('--input-catalogs', dest='cats', required=True,
                        help='List of catalogs to use for astrometric alignment.', nargs='+')
    parser.add_argument('--input-frames', dest='frames', nargs='+', required=True,
                        help='List of frames to coadd.')
    parser.add_argument('--template', help='Turn on 64 threads for template jobs.',
                        action='store_true', default=False)

    args = parser.parse_args()

    # distribute the work to each processor
    if args.frames[0].startswith('@'):
        frames = np.genfromtxt(args.frames[0][1:], dtype=None, encoding='ascii')
        frames = np.atleast_1d(frames)
    else:
        frames = args.frames[0].split()

    if args.cats[0].startswith('@'):
        cats = np.genfromtxt(args.cats[0][1:], dtype=None, encoding='ascii')
        cats = np.atleast_1d(cats)
    else:
        cats = args.cats[0].split()

    # now set up a few pointers to auxiliary files read by sextractor
    wd = os.path.dirname(__file__)
    confdir = os.path.join(wd, '..', 'astromatic', 'makecoadd')
    sexconf = os.path.join(confdir, 'scamp.sex')
    scampparam = os.path.join(confdir, 'scamp.param')
    filtname = os.path.join(confdir, 'default.conv')
    nnwname = os.path.join(confdir, 'default.nnw')
    scampconf = os.path.join(confdir, 'scamp.conf')
    swarpconf = os.path.join(confdir, 'default.swarp') if not args.template else os.path.join(confdir, 'template.swarp')
    psfconf = os.path.join(confdir, 'psfex.conf')

    clargs = '-PARAMETERS_NAME %s -FILTER_NAME %s -STARNNW_NAME %s' % (scampparam, filtname, nnwname)

    allims = ' '.join(frames)
    out = args.outfile_path[0]
    oweight = out.replace('.fits', '.weight.fits')

    # put all swarp temp files into a random dir
    swarp_rundir = f'/tmp/{uuid.uuid4().hex}'
    os.makedirs(swarp_rundir)

    syscall = 'swarp -c %s %s -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s' % (swarpconf, allims, out, oweight)
    syscall += f' -VMEM_DIR {swarp_rundir} -RESAMPLE_DIR {swarp_rundir}'
    libztf.execute(syscall, capture=False)

    # Now postprocess it a little bit
    with fits.open(frames[0]) as f:
        h0 = f[0].header
        band = h0['FILTER']
        field = h0['FIELDID']
        ccdid = h0['CCDID']
        qid = h0['QID']

    mjds = []
    for frame in frames:
        with fits.open(frame) as f:
            mjds.append(f[0].header['OBSMJD'])

    with fits.open(out, mode='update') as f:
        header = f[0].header

        if 'r' in band.lower():
            header['FILTER'] = 'r'
        elif 'g' in band.lower():
            header['FILTER'] = 'g'
        elif 'i' in band.lower():
            header['FILTER'] = 'i'
        else:
            raise ValueError('Invalid filter "%s."' % band)

        # add in the basic stuff
        header['FILTERCODE'] = 'z' + header['FILTER']
        header['FIELD'] = field
        header['CCDID'] = ccdid
        header['QID'] = qid


        # TODO make this more general
        header['PIXSCALE'] = 1.01
        header['MJDEFF'] = np.median(mjds)

        # Add the sky back in as a constant
        f[0].data += 150.

    # Make a new catalog
    outcat = out.replace('.fits', '.cat')
    noise = out.replace('.fits', '.noise.fits')
    bkgsub =out.replace('.fits', '.bkgsub.fits')
    syscall = f'sex -c {sexconf} -CATALOG_NAME {outcat} -CHECKIMAGE_TYPE BACKGROUND_RMS,-BACKGROUND ' \
              f'-CHECKIMAGE_NAME {noise},{bkgsub} -MAG_ZEROPOINT 27.5 {out}'
    syscall = ' '.join([syscall, clargs])
    libztf.execute(syscall, capture=False)

    # now model the PSF
    syscall = f'psfex -c {psfconf} {outcat}'
    libztf.execute(syscall, capture=False)
    psf = out.replace('.fits', '.psf')

    # and save it as a fits model
    gsmod = des.DES_PSFEx(psf)
    with fits.open(psf) as f:
        xcen = f[1].header['POLZERO1']
        ycen = f[1].header['POLZERO2']
        psfsamp = f[1].header['PSF_SAMP']

    cpos = galsim.PositionD(xcen, ycen)
    psfmod = gsmod.getPSF(cpos)
    psfimg = psfmod.drawImage(scale=1., nx=25, ny=25, method='real_space')

    # clear wcs and rotate array to be in same orientation as coadded images (north=up and east=left)
    psfimg.wcs = None
    psfimg = galsim.Image(np.fliplr(psfimg.array))

    psfimpath = f'{psf}.fits'
    # save it to the D
    psfimg.write(psfimpath)

    # And zeropoint the coadd, putting results in the header
    libztf.solve_zeropoint(out, psfimpath, outcat, bkgsub)

    # Now retrieve the zeropoint
    with fits.open(out) as f:
        zp = f[0].header['MAGZP']

    # redo sextractor
    syscall = 'sex -c %s -CATALOG_NAME %s -CHECKIMAGE_NAME %s -MAG_ZEROPOINT %f %s'
    syscall = syscall % (sexconf, outcat, noise, zp, out)
    syscall = ' '.join([syscall, clargs])
    libztf.execute(syscall, capture=False)
    libztf.make_rms(out, oweight)
    libztf.medg(out)
