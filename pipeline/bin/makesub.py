import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from liblg import make_rms, cmbmask, execute, cmbrms
import uuid
import shutil

from liblg import yao_photometry_single

from filterobjects import filter_sexcat
from makecoadd import Fake
from publish import load_catalog

# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


def make_sub(myframes, mytemplates, publish=True):

    # now set up a few pointers to auxiliary files read by sextractor
    wd = os.path.dirname(__file__)
    confdir = os.path.join(wd, '..', 'astromatic', 'makesub')
    scampconfcat = os.path.join(confdir, 'scamp.conf.cat')
    defswarp = os.path.join(confdir, 'default.swarp')
    defsexref = os.path.join(confdir, 'default.sex.ref')
    defsexaper = os.path.join(confdir, 'default.sex.aper')
    defsexsub = os.path.join(confdir, 'default.sex.sub')
    defparref = os.path.join(confdir, 'default.param.ref')
    defparaper = os.path.join(confdir, 'default.param.aper')
    defparsub = os.path.join(confdir, 'default.param.sub')
    defconv = os.path.join(confdir, 'default.conv')
    defnnw = os.path.join(confdir, 'default.nnw')

    for frame, template in zip(myframes, mytemplates):

        # read some header keywords from the template
        with fits.open(template) as f:
            header = f[0].header
            seeref = header['SEEING']
            refskybkg = header['MEDSKY']
            trefskysig = header['SKYSIG']
            tu = header['SATURATE']

        refp = os.path.basename(template)[:-5]
        newp = os.path.basename(frame)[:-5]

        refdir = os.path.dirname(template)
        outdir = os.path.dirname(frame)

        subp = '_'.join([newp, refp])

        refweight = os.path.join(refdir, refp + '.weight.fits')
        refmask = os.path.join(refdir, refp + '.mask.fits')
        refcat = os.path.join(refdir, refp + '.cat')

        newweight = os.path.join(outdir, newp + '.weight.fits')
        newmask = os.path.join(outdir, newp + '.bpm.fits')
        newnoise = os.path.join(outdir, newp + '.rms.fits')
        newcat = os.path.join(outdir, newp + '.cat')
        newhead = os.path.join(outdir, newp + '.head')

        refremap = os.path.join(outdir, 'ref.%s.remap.fits' % subp)
        refremapweight = os.path.join(outdir, 'ref.%s.remap.weight.fits' % subp)
        refremapmask = os.path.join(outdir, 'ref.%s.remap.bpm.fits' % subp)
        refremapnoise = os.path.join(outdir, 'ref.%s.remap.rms.fits' % subp)
        refremaphead = os.path.join(outdir, 'ref.%s.remap.head' % subp)
        refremapcat = os.path.join(outdir, 'ref.%s.remap.cat' % subp)

        subcat = os.path.join(outdir, 'sub.%s.cat' % subp)
        sublist = os.path.join(outdir, 'sub.%s.list' % subp)
        submask = os.path.join(outdir, 'sub.%s.bpm.fits' % subp)
        apercat = os.path.join(outdir, 'ref.%s.remap.ap.cat' % subp)
        badpix = os.path.join(outdir, 'sub.%s.bpix' % subp)
        subrms = os.path.join(outdir, 'sub.%s.rms.fits' % subp)

        sub = os.path.join(outdir, 'sub.%s.fits' % subp)
        tmpnew = os.path.join(outdir, 'new.%s.fits' % subp)

        hotlog = os.path.join(outdir, 'hot.%s.log' % subp)
        hotpar = os.path.join(outdir, 'hot.%s.par' % subp)

        hotlogger = logging.getLogger('hotlog')
        hotparlogger = logging.getLogger('hotpar')

        fh = logging.FileHandler(hotlog)
        fhp = logging.FileHandler(hotpar)

        fh.setLevel(logging.DEBUG)
        fhp.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

        fh.setFormatter(formatter)
        fhp.setFormatter(formatter)

        hotparlogger.info(sub)
        hotparlogger.info(template)
        hotparlogger.info(frame)

        # read some keywords from the fits headers
        with fits.open(frame) as f:
            header = f[0].header
            seenew = header['SEEING']
            newskybkg = header['MEDSKY']
            tnewskysig = header['SKYSIG']
            iu = header['SATURATE']
            naxis1 = header['NAXIS1']
            naxis2 = header['NAXIS2']
            naxis = header['NAXIS']
            refzp = header['MAGZP']

            # make the naxis card images
            hstr = []
            for card in header.cards:
                if 'NAXIS' in card.keyword:
                    hstr.append(card.image)
            hstr = '\n'.join(hstr) + '\n'

        # Make a catalog from the reference for astrometric matching
        syscall = 'scamp -c %s -ASTREFCAT_NAME %s %s'
        syscall = syscall % (scampconfcat, refcat, newcat)
        print(syscall)
        execute(syscall, capture=False)

        # Merge header files
        with open(refremaphead, 'w') as f:
            f.write(hstr)
            with open(newhead, 'r') as nh:
                f.write(nh.read())

        # put all swarp temp files into a random dir
        swarp_rundir = f'/tmp/{uuid.uuid4().hex}'
        os.makedirs(swarp_rundir)

        # Make the remapped ref
        syscall = 'swarp -c %s %s -SUBTRACT_BACK N -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s'
        syscall = syscall % (defswarp, template, refremap, refremapweight)
        syscall += f' -VMEM_DIR {swarp_rundir} -RESAMPLE_DIR {swarp_rundir}'

        execute(syscall, capture=False)

        # Make the noise and bpm images
        make_rms(refremap, refremapweight)

        # Add the masks together to make the supermask
        cmbmask(refremapmask, newmask, submask)

        # Add the sub and new rms together in quadrature
        cmbrms(refremapnoise, newnoise, subrms)

        # Create new and reference noise images
        ntst = seenew > seeref

        seeing = seenew if ntst else seeref

        pixscal = 1.01  # TODO make this more general
        seepix = pixscal * seeing
        r = 2.5 * seepix
        rss = 6. * seepix

        hotlogger.info('r and rss %f %f' % (r, rss))

        gain = 1.0  # TODO check this assumption

        newskysig = tnewskysig * 1.48 / gain
        refskysig = trefskysig * 1.48 / gain

        il = newskybkg - 10. * newskysig
        tl = refskybkg - 10. * refskysig

        hotlogger.info('tl and il %f %f' % (tl, il))
        hotlogger.info('refskybkg and newskybkg %f %f' % (refskybkg, newskybkg))
        hotlogger.info('refskysig and newskysig %f %f' % (refskysig, newskysig))

        nsx = naxis1 / 100.
        nsy = naxis2 / 100.

        hotlogger.info('nsx nsy %f %f' % (nsx, nsy))
        hotparlogger.info(str(il))
        hotparlogger.info(str(iu))
        hotparlogger.info(str(tl))
        hotparlogger.info(str(tu))
        hotparlogger.info(str(r))
        hotparlogger.info(str(rss))
        hotparlogger.info(str(nsx))
        hotparlogger.info(str(nsy))

        convnew = seeref > seenew

        convolve_target = 'i' if convnew else 't'
        syscall = f'hotpants -inim %s -hki -n i -c {convolve_target} -tmplim %s -outim %s -tu %f -iu %f  -tl %f -il %f -r %f ' \
                  f'-rss %f -tni %s -ini %s -imi %s -nsx %f -nsy %f'
        syscall = syscall % (frame, refremap, sub, tu, iu, tl, il, r, rss, refremapnoise, newnoise,
                             submask, nsx, nsy)

        print(syscall)
        execute(syscall, capture=False)

        # Calibrate the subtraction

        with fits.open(sub, mode='update') as f, fits.open(frame) as fr:
            header = f[0].header
            subzp = fr[0].header['MAGZP']  # normalized to photometric system of new image
            header['MAGZP'] = subzp
            subpix = f[0].data
            wcs = WCS(header)

        subpixvar = (0.5 * (np.percentile(subpix, 84) - np.percentile(subpix, 16.)))**2

        # estimate the variance in psf fit fluxes
        subcorners = wcs.calc_footprint()
        subminra = subcorners[:, 0].min()
        submaxra = subcorners[:, 0].max()
        dra = submaxra - subminra
        submindec = subcorners[:, 1].min()
        submaxdec = subcorners[:, 1].max()
        ddec = submaxdec - submindec

        subminra += 0.1 * dra
        submaxra -= 0.1 * dra
        submindec += 0.1 * ddec
        submaxdec -= 0.1 * ddec

        refpsf = template.replace('.fits', '.psf')
        newpsf = frame.replace('.fits', '.psf')
        psf = newpsf if convnew else refpsf
        subpsf = sub.replace('.fits', '.psf')
        shutil.copy(psf, subpsf)

        fluxes = []
        for i in range(1000):
            ra = np.random.uniform(subminra, submaxra)
            dec = np.random.uniform(submindec, submaxdec)
            pobj = yao_photometry_single(sub, subpsf, ra, dec)
            fluxes.append(pobj.Fpsf)

        subpsffluxvar = (0.5 * (np.percentile(fluxes, 84) - np.percentile(fluxes, 16.)))**2
        beta = subpsffluxvar / subpixvar

        with fits.open(sub, mode='update') as f:
            f[0].header['BETA'] = beta

        # Make the subtraction catalogs
        clargs = ' -PARAMETERS_NAME %%s -FILTER_NAME %s -STARNNW_NAME %s' % (defconv, defnnw)

        # Reference catalog
        syscall = 'sex -c %s -MAG_ZEROPOINT %f -CATALOG_NAME %s -VERBOSE_TYPE QUIET %s'
        syscall = syscall % (defsexref, refzp, refremapcat, refremap)
        syscall += clargs % defparref
        execute(syscall, capture=False)

        # Subtraction catalog
        syscall = 'sex -c %s -MAG_ZEROPOINT %f -CATALOG_NAME %s -ASSOC_NAME %s -VERBOSE_TYPE QUIET %s -WEIGHT_IMAGE %s -WEIGHT_TYPE MAP_RMS'
        syscall = syscall % (defsexsub, subzp, subcat, refremapcat, sub, subrms)
        syscall += clargs % defparsub
        syscall += f' -FLAG_IMAGE {submask}'
        print(syscall)
        execute(syscall, capture=False)

        # Aperture catalog
        syscall = 'sex -c %s -MAG_ZEROPOINT %f -CATALOG_NAME %s -VERBOSE_TYPE QUIET %s,%s'
        syscall = syscall % (defsexaper, refzp, apercat, sub, refremap)
        syscall += clargs % defparaper
        execute(syscall, capture=False)

        # now filter objects
        filter_sexcat(subcat)

        # publish to marshal

        if publish:
            goodcat = subcat.replace('cat', 'cat.out.fits')
            load_catalog(goodcat, refremap, frame, sub)

        # put fake info in header and region file if there are any fakes
        with fits.open(frame) as f, fits.open(sub, mode='update') as fsub:
            hdr = f[0].header
            wcs = WCS(hdr)
            fakecards = [c for c in hdr.cards if 'FAKE' in c.keyword and 'X' not in c.keyword and 'Y' not in c.keyword]
            fakekws = [c.keyword for c in fakecards]

            try:
                maxn = max(set(map(int, [kw[4:-2] for kw in fakekws]))) + 1
            except ValueError:
                maxn = 0

            if maxn > 0:
                subheader = fsub[0].header
                subheader.update(fakecards)

                with open(sub.replace('fits', 'fake.reg'), 'w') as o:

                    # make the region files
                    o.write("""# Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal" \
    select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 \
    delete=1 include=1 source=1
    physical
    """)

                    for i in range(maxn):
                        ra = hdr[f'FAKE{i:02d}RA']
                        dec = hdr[f'FAKE{i:02d}DC']
                        mag = hdr[f'FAKE{i:02d}MG']
                        fake = Fake(ra, dec, mag)
                        x, y = fake.xy(wcs)
                        subheader[f'FAKE{i:02d}X'] = x
                        subheader[f'FAKE{i:02d}Y'] = y

                        o.write(f'circle({x},{y},10) # width=2 color=red\n')
                        o.write(f'text({x},{y+8}  # text={{mag={mag:.2f}}}\n')



if __name__ == '__main__':

    import argparse
    import logging

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # set up the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--science-frames', dest='sciimg', required=True,
                        help='Input frames. Prefix with "@" to read from a list.', nargs='+')
    parser.add_argument('--templates', dest='template', nargs='+', required=True,
                        help='Templates to subtract. Prefix with "@" to read from a list.')
    parser.add_argument('--no-publish', dest='publish', action='store_false', default=True,
                        help='Dont publish results to the marshal.')
    args = parser.parse_args()

    # distribute the work to each processor

    if args.sciimg[0].startswith('@'):
        framelist = args.sciimg[0][1:]

        if rank == 0:
            frames = np.genfromtxt(framelist, dtype=None, encoding='ascii')
            frames = np.atleast_1d(frames).tolist()
        else:
            frames = None

    else:
        frames = args.sciimg

    if args.template[0].startswith('@'):
        templist = args.template[0][1:]

        if rank == 0:
            templates = np.genfromtxt(templist, dtype=None, encoding='ascii')
            templates = np.atleast_1d(templates).tolist()
        else:
            templates = None
    else:
        templates = args.template

    myframes = _split(frames, size)[rank]
    mytemplates = _split(templates, size)[rank]

    make_sub(myframes, mytemplates, publish=args.publish)




