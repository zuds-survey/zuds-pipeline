import os
import numpy as np
import liblg
import uuid
import time
from astropy.io import fits
from astropy.wcs import WCS
import galsim
from makevariance import make_variance
import logging
from galsim import des
from astropy.convolution import convolve
import shutil

#TODO: Delete this
SEED = 1234


class Fake(object):

    def __init__(self, ra, dec, mag=20.):
        self.ra = ra
        self.dec = dec
        self.mag = mag

    def xy(self, wcs):
        # Return xy pixel coordinates of fake on image
        return wcs.wcs_world2pix([(self.ra, self.dec)], 1)[0]

    def galsim_object(self, sigma, magzpt):
        flux = 10**(-0.4 * (self.mag - magzpt))
        obj = galsim.Gaussian(sigma=sigma).withFlux(flux)
        return obj


def add_fakes_to_image(inim, outim, fakes, seed=None):

    if seed is None:
        seed = time.time()

    rng = galsim.BaseDeviate(seed)

    im = galsim.fits.read(inim)
    with fits.open(inim) as f:
        seeing = f[0].header['SEEING']
        sigma = seeing / 2.355
        zp = f[0].header['MAGZP']
        wcs = WCS(f[0].header)

    for fake in fakes:
        obj = fake.galsim_object(sigma, zp)
        img = obj.drawImage()
        noise = galsim.PoissonNoise(rng)
        img.addNoise(noise)
        img.setCenter(fake.xy(wcs))
        bounds = img.bounds
        im[bounds] = im[bounds] + img

    galsim.fits.write(im, file_name=outim)

    with fits.open(outim, mode='update') as f, fits.open(inim) as ff:
        hdr = f[0].header
        inhdr = ff[0].header
        hdr.update(inhdr.cards)
        for i, fake in enumerate(fakes):
            hdr[f'FAKE{i:02d}RA'] = fake.ra
            hdr[f'FAKE{i:02d}DC'] = fake.dec
            hdr[f'FAKE{i:02d}X'], hdr[f'FAKE{i:02d}Y'] = fake.xy(wcs)
            hdr[f'FAKE{i:02d}MG'] = fake.mag


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


    # TODO: Delete this
    rng = np.random.RandomState(SEED)

    # First stamp everything together so that fakes are put down at the right place
    mycats = ' '.join(cats)

    # First scamp everything
    # make a random dir for the output catalogs
    scamp_outpath = f'/tmp/{uuid.uuid4().hex}'
    os.makedirs(scamp_outpath)

    syscall = 'scamp -c %s %s' % (scampconf, mycats)
    syscall += f' -REFOUT_CATPATH {scamp_outpath}'
    if args.nothreads:
        syscall += ' -NTHREADS 2'
    liblg.execute(syscall, capture=False)


    # First check to see if the fakes should be added
    if args.nfakes > 0:

        # get the range of ra and dec over which fakes can be implanted

        radec = []

        for frame in frames:

            # read in the fits header from scamp

            head = frame.replace('.fits', '.head')
            
            with open(head) as f:
                h = fits.Header()
                for text in f:
                    h.append(fits.Card.fromstring(text.strip()))
                with fits.open(frame) as hdul:
                    hh = hdul[0].header
                    h['SIMPLE'] = hh['SIMPLE']
                    h['NAXIS'] = hh['NAXIS']
                    h['BITPIX'] = hh['BITPIX']
                    h['NAXIS1'] = hh['NAXIS1']
                    h['NAXIS2'] = hh['NAXIS2']
                    
                wcs = WCS(h)
                
                # get x and y coords
                world_corners = wcs.calc_footprint()
                radec.append(world_corners)

        radec = np.vstack(radec)
        minra, mindec = radec.min(axis=0)
        maxra, maxdec = radec.max(axis=0)

        rarange = maxra - minra
        decrange = maxdec - mindec

        fakeminra = minra + 0.15 * rarange
        fakemaxra = minra + 0.85 * rarange

        fakemindec = mindec + 0.15 * decrange
        fakemaxdec = mindec + 0.85 * decrange

        fakes = []

        for i in range(args.nfakes):

            ra = rng.uniform(fakeminra, fakemaxra)
            dec = rng.uniform(fakemaxdec, fakemindec)
            mag = rng.uniform(19.5, 24)
            fake = Fake(ra, dec, mag=mag)
            fakes.append(fake)

        for frame in frames:
            outim = frame.replace('.fits', '.fake.fits')
            add_fakes_to_image(frame, outim, fakes, seed=SEED)

        masks = [f.replace('sciimg','mskimg') for f in frames]
        for f in frames:
            orighead = f.replace('.fits', '.head')
            newhead = orighead.replace('.head', '.fake.head')
            origcat = f.replace('.fits', '.cat')
            newcat = origcat.replace('.cat', '.fake.cat')
            shutil.copy(orighead, newhead)
            shutil.copy(origcat, newcat)

        frames = [f.replace('.fits', '.fake.fits') for f in frames]
        logger = logging.getLogger('fakevar')
        logger.setLevel(logging.DEBUG)
        make_variance(frames, masks, logger)


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
    syscall = 'sex -c %s -CATALOG_NAME %s -CHECKIMAGE_NAME %s -MAG_ZEROPOINT 27.5 %s'
    syscall = syscall % (sexconf, outcat, noise, out)
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
    liblg.solve_zeropoint(out, psfimpath, outcat)

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
                                                        'MG' in c.keyword))]
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

    if args.convolve:
        with fits.open(out, mode='update') as f, fits.open(psfimpath, mode='update') as pf:
            kernel = pf[0].data
            idata = f[0].data
            convolved = convolve(idata, kernel)
            f[0].data = convolved
            newpsf = convolve(kernel, kernel)
            pf[0].data = newpsf


