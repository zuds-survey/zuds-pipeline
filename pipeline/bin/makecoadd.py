import os
import numpy as np
import liblg
import uuid
import time
from astropy.io import fits
from astropy.wcs import WCS
import galsim


#TODO: Delete this
SEED = 1234


class Fake(object):

    def __init__(self, ra, dec, mag=20.):
        self.ra = ra
        self.dec = dec
        self.mag = mag

    def xy(self, impath):
        # Return xy pixel coordinates of fake on image
        with fits.open(impath) as hdul:
            wcs = WCS(hdul[0].header)

        return wcs.wcs_world2pix((self.ra, self.dec), 1)

    def galsim_image(self, rng, sigma, magzpt, wcs_image=None):

        flux = 10**(-0.4 * (self.mag - magzpt))
        object = galsim.Gaussian(sigma=sigma).withFlux(flux)
        noise = galsim.PoissonNoise(rng)
        image = object.drawImage()
        image.addNoise(noise)

        if wcs_image is not None:

            image.setCenter(self.xy(wcs_image))
            with fits.open(wcs_image) as hdul:
                gwcs = galsim.AstropyWCS(header=hdul[0].header)
                image.wcs = gwcs

        return image


def add_fakes_to_image(inim, outim, fakes, seed=None):

    if seed is None:
        seed = time.time()

    rng = galsim.BaseDeviate(seed)

    im = galsim.fits.read(inim)
    with fits.open(inim) as f:
        seeing = f[0].header['SEEING']
        sigma = seeing / 2.355
        zp = f[0].header['MAGZP']

    for fake in fakes:
        im += fake.galsim_image(rng, sigma, zp, wcs_image=inim)

    galsim.fits.write(im, file_name=outim)

    with fits.open(outim, mode='update') as f, fits.open(inim) as ff:
        hdr = f[0].header
        inhdr = ff[0].header
        hdr.update(inhdr.cards)
        for i, fake in enumerate(fakes):
            hdr[f'FAKE{i:02d}RA'] = fake.ra
            hdr[f'FAKE{i:02d}DC'] = fake.dec
            hdr[f'FAKE{i:02d}X'], hdr[f'FAKE{i:02d}Y'] = fake.xy(outim)
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
    confdir = os.path.join(wd, '..', 'config', 'makecoadd')
    sexconf = os.path.join(confdir, 'scamp.sex')
    scampparam = os.path.join(confdir, 'scamp.param')
    filtname = os.path.join(confdir, 'default.conv')
    nnwname = os.path.join(confdir, 'default.nnw')
    scampconf = os.path.join(confdir, 'scamp.conf')
    swarpconf = os.path.join(confdir, 'default.swarp')

    clargs = '-PARAMETERS_NAME %s -FILTER_NAME %s -STARNNW_NAME %s' % (scampparam, filtname, nnwname)

    mycats = ' '.join(cats)

    # TODO: Delete this
    rng = np.random.RandomState(SEED)

    # First check to see if the fakes should be added
    if args.nfakes > 0:

        # get the range of ra and dec over which fakes can be implanted

        radec = []

        for frame in frames:
            with fits.open(frame) as f:

                wcs = WCS(f[0].header)
                im = f[0].data
                n1, n2 = im.shape

                # get x and y coords
                pixcrd = np.asarray([[1, 1], [n1, 1], [1, n2], [n1, n2]])
                world_corners = wcs.wcs_pix2world(pixcrd, 1)
                radec.append(world_corners)

        radec = np.vstack(radec)
        minra, mindec = radec.min(axis=0)
        maxra, maxdec = radec.max(axis=0)

        rarange = maxra - minra
        decrange = maxdec - mindec

        fakeminra = minra + 0.15 * rarange
        fakemaxra = minra + 0.85 * rarange

        fakemindec = mindec + 0.15 * rarange
        fakemaxdec = mindec + 0.85 * rarange

        fakes = []

        for i in range(args.nfakes):

            ra = rng.uniform(fakeminra, fakemaxra)
            dec = rng.uniform(fakemaxdec, fakemindec)
            mag = rng.uniform(17, 24)
            fake = Fake(ra, dec, mag=mag)
            fakes.append(fake)

        for frame in frames:
            outim = frame.replace('.fits', '.fake.fits')
            add_fakes_to_image(frame, outim, fakes, seed=SEED)

        frames = [f.replace('.fits', '.fake.fits') for f in frames]


    # First scamp everything
    # make a random dir for the output catalogs
    scamp_outpath = f'/tmp/{uuid.uuid4().hex}'
    os.makedirs(scamp_outpath)

    syscall = 'scamp -c %s %s' % (scampconf, mycats)
    syscall += f' -REFOUT_CATPATH {scamp_outpath}'
    if args.nothreads:
        syscall += ' -NTHREADS 2'
    liblg.execute(syscall, capture=False)

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

        # Add the sky back in as a constant
        f[0].data += 150.

    # Make a new catalog
    outcat = args.output_basename[0] + '.cat'
    noise = args.output_basename[0] + '.noise.fits'
    syscall = 'sex -c %s -CATALOG_NAME %s -CHECKIMAGE_NAME %s -MAG_ZEROPOINT 27.5 %s'
    syscall = syscall % (sexconf, outcat, noise, out)
    syscall = ' '.join([syscall, clargs])
    liblg.execute(syscall, capture=False)

    # And zeropoint the coadd, putting results in the header
    liblg.solve_zeropoint(out, outcat)

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
