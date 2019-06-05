import os
import numpy as np
import liblg
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


def determine_and_submit_template_jobs(variance_corrids, metatable, options):

    # check to see if new templates are needed
    template_corrids = {}

    batch = []

    for (field, quadrant, band, ccdnum), group in metatable.groupby(['field',
                                                                     'qid',
                                                                     'filtercode',
                                                                     'ccdid']):

        ofield, oquadrant, oband, occdnum = field, quadrant, band, ccdnum

        # convert into proper values
        field = int(field)
        quadrant = int(quadrant)
        band = band[1:]
        ccdnum = int(ccdnum)

        # check if a template is needed
        if not self.needs_template(field, ccdnum, quadrant, band):
            continue

        tmplids, tmplims, jds, hasvar = self.create_template_image_list(field, ccdnum, quadrant, band)

        if len(tmplids) < self.pipeline_schema['template_minimages']:
            # not enough images to make a template -- try again some other time
            continue

        minjd = np.min(jds)
        maxjd = np.max(jds)

        mintime = Time(minjd, format='jd', scale='utc')
        maxtime = Time(maxjd, format='jd', scale='utc')

        mindatestr = mintime.iso.split()[0].replace('-', '')
        maxdatestr = maxtime.iso.split()[0].replace('-', '')

        # see what jobs need to finish before this one can run
        dependencies = []
        remake_variance = []
        for path, hv in zip(tmplims, hasvar):
            if not hv:
                if path in variance_corrids:
                    varcorrid = variance_corrids[path]
                    dependencies.append(varcorrid)
                else:
                    remake_variance.append(path)

        if len(remake_variance) > 0:
            moredeps = self.determine_and_relay_variance_jobs(remake_variance)
            dependencies.extend(moredeps.values())

        dependencies = list(set(dependencies))

        # what will this template be called?
        tmpbase = tmp_basename_form.format(paddedfield=field,
                                           qid=oquadrant,
                                           paddedccdid=ccdnum,
                                           filtercode=oband,
                                           mindate=mindatestr,
                                           maxdate=maxdatestr)

        outfile_name = nersc_tmpform.format(fname=tmpbase)

        # now that we have the dependencies we can relay the coadd job for submission
        tmpl_data = {'dependencies':dependencies, 'jobtype':'template', 'images':tmplims,
                     'outfile_name': outfile_name, 'imids': tmplids, 'quadrant': quadrant,
                     'field': field, 'ccdnum': ccdnum, 'mindate':mindatestr,
                     'maxdate':maxdatestr, 'filter': band,
                     'pipeline_schema_id': self.pipeline_schema['schema_id']}

        batch.append(tmpl_data)

        if len(batch) == template_batchsize:
            payload = {'jobtype': 'template', 'jobs': batch}
            body = json.dumps(payload)
            tmpl_corrid = self.relay_job(body)
            for d in batch:
                template_corrids[(d['field'], d['quadrant'], d['filter'], d['ccdnum'])] = (tmpl_corrid, d)
            batch = []

    if len(batch) > 0:
        payload = {'jobtype': 'template', 'jobs': batch}
        body = json.dumps(payload)
        tmpl_corrid = self.relay_job(body)
        for d in batch:
            template_corrids[(d['field'], d['quadrant'], d['filter'], d['ccdnum'])] = (tmpl_corrid, d)

    return template_corrids

def make_coadd_bins(self, field, ccdnum, quadrant, filter, maxdate=None):

    self._refresh_connections()

    if maxdate is None:
        query = 'SELECT MAXDATE FROM TEMPLATE WHERE FIELD=%s AND CCDNUM=%s AND QUADRANT=%s AND FILTER=%s ' \
                'AND PIPELINE_SCHEMA_ID=%s'
        self.cursor.execute(query, (field, ccdnum, quadrant, filter, self.pipeline_schema['schema_id']))
        result = self.cursor.fetchall()

        if len(result) == 0:
            raise ValueError('No template queued or on disk -- can\'t make coadd bins')
        date = result[0][0]

    else:
        date = maxdate

    startsci = pd.to_datetime(date) + pd.Timedelta(self.pipeline_schema['template_science_minsep_days'],
                                                   unit='d')

    if self.pipeline_schema['rolling']:
        dates = pd.date_range(startsci, pd.to_datetime(datetime.date.today()), freq='1D')
        bins = []
        for i, date in enumerate(dates):
            if i + self.pipeline_schema['scicoadd_window_size'] >= len(dates):
                break
            bins.append((date, dates[i + self.pipeline_schema['scicoadd_window_size']]))

    else:
        binedges = pd.date_range(startsci, pd.to_datetime(datetime.datetime.today()),
                                 freq=f'{self.pipeline_schema["scicoadd_window_size"]}D')

        bins = []
        for i, lbin in enumerate(binedges[:-1]):
            bins.append((lbin, binedges[i + 1]))

    return bins

if __name__ == '__main__':

    import argparse

    # set up the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-basename', dest='output_basename', required=True,
                        help='Basename of output coadd.', nargs=1)
    parser.add_argument('--input-catalogs', dest='cats', required=True,
                        help='List of catalogs to use for astrometric alignment.', nargs='+')
    parser.add_argument('--input-frames', dest='frames', nargs='+', required=True,
                        help='List of frames to coadd.')
    parser.add_argument('--nothreads', dest='nothreads', action='store_true', default=False,
                        help='Run astromatic software with only one thread.')
    parser.add_argument('--add-fakes', dest='nfakes', type=int, default=0,
                        help='Number of fakes to add. Default 0.')
    parser.add_argument('--convolve', dest='convolve', action='store_true', default=False,
                        help='Convolve image with PSF to artificially degrade its quality.')

    args = parser.parse_args()

    # distribute the work to each processor
    if args.frames[0].startswith('@'):
        frames = np.genfromtxt(args.frames[0][1:], dtype=None, encoding='ascii')
        frames = np.atleast_1d(frames)
    else:
        frames = args.frames

    if args.cats[0].startswith('@'):
        cats = np.genfromtxt(args.cats[0][1:], dtype=None, encoding='ascii')
        cats = np.atleast_1d(cats)
    else:
        cats = args.cats

    # now set up a few pointers to auxiliary files read by sextractor
    wd = os.path.dirname(__file__)
    confdir = os.path.join(wd, '..', 'astromatic', 'makecoadd')
    sexconf = os.path.join(confdir, 'scamp.sex')
    scampparam = os.path.join(confdir, 'scamp.param')
    filtname = os.path.join(confdir, 'default.conv')
    nnwname = os.path.join(confdir, 'default.nnw')
    scampconf = os.path.join(confdir, 'scamp.conf')
    swarpconf = os.path.join(confdir, 'default.swarp')
    psfconf = os.path.join(confdir, 'psfex.conf')

    clargs = '-PARAMETERS_NAME %s -FILTER_NAME %s -STARNNW_NAME %s' % (scampparam, filtname, nnwname)




    allims = ' '.join(frames)
    out = args.output_basename[0] + '.fits'
    oweight = args.output_basename[0] + '.weight.fits'

    # put all swarp temp files into a random dir
    swarp_rundir = f'/tmp/{uuid.uuid4().hex}'
    os.makedirs(swarp_rundir)

    syscall = 'swarp -c %s %s -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s' % (swarpconf, allims, out, oweight)
    syscall += f' -VMEM_DIR {swarp_rundir} -RESAMPLE_DIR {swarp_rundir}'
    if args.nothreads:
        syscall += ' -NTHREADS 2'
    liblg.execute(syscall, capture=False)

    # Now postprocess it a little bit
    with fits.open(frames[0]) as f:
        h0 = f[0].header
        band = h0['FILTER']

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

        # TODO make this more general
        header['PIXSCALE'] = 1.0
        header['MJDEFF'] = np.median(mjds)

        # Add the sky back in as a constant
        f[0].data += 150.

    # Make a new catalog
    outcat = args.output_basename[0] + '.cat'
    noise = args.output_basename[0] + '.noise.fits'
    bkgsub = args.output_basename[0] + '.bkgsub.fits'
    syscall = f'sex -c {sexconf} -CATALOG_NAME {outcat} -CHECKIMAGE_TYPE BACKGROUND_RMS,-BACKGROUND ' \
              f'-CHECKIMAGE_NAME {noise},{bkgsub} -MAG_ZEROPOINT 27.5 {out}'
    syscall = ' '.join([syscall, clargs])
    liblg.execute(syscall, capture=False)

    # now model the PSF
    syscall = f'psfex -c {psfconf} {outcat}'
    liblg.execute(syscall, capture=False)
    psf = args.output_basename[0] + '.psf'

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

    if args.convolve:
        with fits.open(out, mode='update') as f, fits.open(psfimpath) as pf:
            kernel = pf[0].data
            idata = f[0].data
            convolved = convolve(idata, kernel)
            f[0].data = convolved

        # Now remake the PSF model
        # Make a new catalog
        outcat = args.output_basename[0] + '.cat'
        noise = args.output_basename[0] + '.noise.fits'
        bkgsub = args.output_basename[0] + '.bkgsub.fits'
        syscall = f'sex -c {sexconf} -CATALOG_NAME {outcat} -CHECKIMAGE_TYPE BACKGROUND_RMS,-BACKGROUND ' \
                  f'-CHECKIMAGE_NAME {noise},{bkgsub} -MAG_ZEROPOINT 27.5 {out}'
        syscall = ' '.join([syscall, clargs])
        liblg.execute(syscall, capture=False)

        # now model the PSF
        syscall = f'psfex -c {psfconf} {outcat}'
        liblg.execute(syscall, capture=False)
        psf = args.output_basename[0] + '.psf'

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
    liblg.solve_zeropoint(out, psfimpath, outcat, bkgsub)

    # Now retrieve the zeropoint
    with fits.open(out) as f:
        zp = f[0].header['MAGZP']

    # redo sextractor
    syscall = 'sex -c %s -CATALOG_NAME %s -CHECKIMAGE_NAME %s -MAG_ZEROPOINT %f %s'
    syscall = syscall % (sexconf, outcat, noise, zp, out)
    syscall = ' '.join([syscall, clargs])
    liblg.execute(syscall, capture=False)
    liblg.make_rms(out, oweight)
    liblg.medg(out)

    if args.nfakes > 0:
        with fits.open(out, mode='update') as f, fits.open(frames[0]) as ff, open(out.replace('fits', 'reg'), 'w') as o:
            cards = [c for c in ff[0].header.cards if ('FAKE' in c.keyword and
                                                       ('RA' in c.keyword or
                                                        'DC' in c.keyword or
                                                        'MG' in c.keyword or
                                                        'FL' in c.keyword))]
            wcs = WCS(f[0].header)
            f[0].header.update(cards)

            # make the region file
            hdr = f[0].header

            o.write("""# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 de\
lete=1 include=1 source=1
physical
""")

            for i, fake in enumerate(fakes):
                x, y = fake.xy(wcs)
                hdr[f'FAKE{i:02d}X'], hdr[f'FAKE{i:02d}Y'] = x, y
                o.write(f'circle({x},{y},10) # width=2 color=red\n')

