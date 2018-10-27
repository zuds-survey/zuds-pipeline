import os
import numpy as np
from astropy.io import fits
from liblg import make_rms, cmbmask, execute

# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


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
                        help='List of new images to make subtractions with.', nargs=1)
    parser.add_argument('--templates', dest='template', nargs=1, required=True,
                        help='Template to subtract.')
    args = parser.parse_args()

    # distribute the work to each processor

    framelist = args.sciimg[0]
    templist = args.template[0]

    if rank == 0:
        frames = np.genfromtxt(framelist, dtype=None, encoding='ascii')
        templates = np.genfromtxt(templist, dtype=None, encoding='ascii')
        frames = np.atleast_1d(frames).tolist()
        templates = np.atleast_1d(templates).tolist()
    else:
        frames = None
        templates = None

    frames = comm.bcast(frames, root=0)
    templates = comm.bcast(templates, root=0)

    myframes = _split(frames, size)[rank]
    mytemplates = _split(templates, size)[rank]

    # now set up a few pointers to auxiliary files read by sextractor
    wd = os.path.dirname(__file__)
    confdir = os.path.join(wd, '..', 'config', 'makesub')
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
        execute(syscall, capture=False)

        # Merge header files
        with open(refremaphead, 'w') as f:
            f.write(hstr)
            with open(newhead, 'r') as nh:
                f.write(nh.read())

        # Make the remapped ref
        syscall = 'swarp -c %s %s -SUBTRACT_BACK N -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s'
        syscall = syscall % (defswarp, template, refremap, refremapweight)
        execute(syscall, capture=False)

        # Make the noise and bpm images
        make_rms(refremap, refremapweight)

        # Add the masks together to make the supermask
        cmbmask(refremapmask, newmask, submask)

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

        syscall  = 'hotpants -inim %s -hki -n i -c t -tmplim %s -outim %s -tu %f -iu %f  -tl %f -il %f -r %f ' \
                   '-rss %f -tni %s -ini %s -imi %s -nsx %f -nsy %f'
        syscall = syscall % (frame, refremap, sub, tu, iu, tl, il, r, rss, refremapnoise, newnoise,
                             submask, nsx, nsy)
        execute(syscall, capture=False)

        # Calibrate the subtraction

        with fits.open(sub, mode='update') as f:
            header = f[0].header
            frat = float(header['KSUM00'])
            subzp = 2.5 * np.log10(frat) + refzp
            header['MAGZP'] = subzp

        # Make the subtraction catalogs
        clargs = ' -PARAMETERS_NAME %%s -FILTER_NAME %s -STARNNW_NAME %s' % (defconv, defnnw)

        # Reference catalog
        syscall = 'sex -c %s -MAG_ZEROPOINT %f -CATALOG_NAME %s -VERBOSE_TYPE QUIET %s'
        syscall = syscall % (defsexref, refzp, refremapcat, refremap)
        syscall += clargs % defparref
        execute(syscall, capture=False)

        # Subtraction catalog
        syscall = 'sex -c %s -MAG_ZEROPOINT %f -CATALOG_NAME %s -ASSOC_NAME %s -VERBOSE_TYPE QUIET %s'
        syscall = syscall % (defsexsub, subzp, subcat, refremapcat, sub)
        syscall += clargs % defparsub
        execute(syscall, capture=False)

        # Aperture catalog
        syscall = 'sex -c %s -MAG_ZEROPOINT %f -CATALOG_NAME %s -VERBOSE_TYPE QUIET %s,%s'
        syscall = syscall % (defsexaper, refzp, apercat, sub, refremap)
        syscall += clargs % defparaper
        execute(syscall, capture=False)
