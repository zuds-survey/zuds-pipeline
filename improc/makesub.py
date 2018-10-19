import os
import numpy as np
from imlib import fits
from imlib import make_rms, cmbmask

# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


if __name__ == '__main__':

    import argparse
    import logging
    from mpi4py import MPI

    # set up the inter-rank communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # set up the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--new-frames', dest='frames', required=True,
                        help='List of new images to make subtractions with.', nargs=1)
    parser.add_argument('--template', dest='template', nargs=1, required=True,
                        help='Template to subtract.')
    args = parser.parse_args()

    # distribute the work to each processor
    if rank == 0:
        frames = np.genfromtxt(args.frames[0], dtype=None, encoding='ascii')
    else:
        frames = None

    frames = comm.bcast(frames, root=0)
    frames = _split(frames, size)[rank]

    # now set up a few pointers to auxiliary files read by sextractor
    wd = os.path.dirname(__file__)
    confdir = os.path.join(wd, 'config', 'makesub')
    scampconfcat = os.path.join(confdir, 'scamp.conf.cat')
    defswarp = os.path.join(confdir, 'default.swarp')
    defsexref = os.path.join(confdir, 'default.sex.ref')
    defsexaper = os.path.join(confdir, 'default.sex.aper')
    defsexsub = os.path.join(confdir, 'default.sex.sub')

    for frame in frames:
        refp = os.path.basename(args.template)[:-5]
        newp = os.path.basename(frame)[:-5]

        refdir = os.path.dirname(args.template)
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
        hotparlogger.info(args.template)
        hotparlogger.info(frame)

        # Make a catalog from the reference for astrometric matching
        syscall = 'scamp -c %s -ASTREFCAT_NAME %s %s >> %s 2>&1'
        syscall = syscall % (scampconfcat, refcat, newcat, hotlog)
        os.system(syscall)

        # Merge header files
        with open(refremaphead, 'w') as f:
            f.writelines([fits.read_header_int(frame, 'NAXIS')])
            with open(newhead, 'r') as nh:
                f.write(nh.read())

        # Make the remapped ref
        syscall = 'SWarp -c %s %s -SUBTRACT_BACK N -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s > %s 2&>1'
        syscall = syscall % (defswarp, args.template, refremap, refremapweight, hotlog)
        os.system(syscall)

        # Make the noise and bpm images
        make_rms(refremap, refremapweight)

        # Add the masks together to make the supermask
        cmbmask(refremapmask, newmask, submask)

        # Create new and reference noise images
        seenew = fits.read_header_float(frame, 'SEEING')
        seeref = fits.read_header_float(args.template, 'SEEING')
        ntst = seenew > seeref

        seeing = seenew if ntst else seeref

        pixscal = 1.01  # TODO make this more general
        seepix = pixscal * seeing
        r = 2.5 * seepix
        rss = 6. * seepix

        hotlogger.info('r and rss %f %f' % (r, rss))

        newskybkg = fits.read_header_float(frame, 'MEDSKY')
        refskybkg = fits.read_header_float(args.template, 'MEDSKY')

        tnewskysig = fits.read_header_float(frame, 'SKYSIG')
        trefskysig = fits.read_header_float(args.template, 'SKYSIG')

        tu = fits.read_header_float(args.template, 'SATURATE')
        iu = fits.read_header_float(frame, 'SATURATE')

        gain = 1.0  # TODO check this assumption

        newskysig = tnewskysig * 1.48 / gain
        refskysig = trefskysig * 1.48 / gain

        il = newskybkg - 10. * newskysig
        tl = refskybkg - 10. * refskysig

        hotlogger.info('tl and il %f %f' % (tl, il))
        hotlogger.info('refskybkg and newskybkg %f %f' % (refskybkg, newskybkg))
        hotlogger.info('refskysig and newskysig %f %f' % (refskysig, newskysig))

        naxis1 = fits.read_header_int(frame, 'NAXIS1')
        naxis2 = fits.read_header_int(frame, 'NAXIS2')

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
                   '-rss %f -tni %s -ini %s -imi %s -nsx %f -nsy %f >> %s 2>&1'
        syscall = syscall % (frame, refremap, sub, tu, iu, tl, il, r, rss, refremapnoise, newnoise,
                             submask, nsx, nsy, hotlog)
        os.system(syscall)

        # Calibrate the subtraction
        refzp = fits.read_header_float(sub, 'MAGZP')
        frat = fits.read_header_float(sub, 'KSUM00')
        subzp = 2.5 * np.log10(frat) + refzp

        fits.update_header(sub, 'SUBZP', subzp)

        # Make the subtraction catalogs

        # Reference catalog
        syscall = 'sextractor -c %s -MAG_ZEROPOINT %f -CATALOG_NAME %f -VERBOSE_TYPE QUIET %s'
        syscall = syscall % (defsexref, refzp, refremapcat, refremap)
        os.system(syscall)

        # Subtraction catalog
        syscall = 'sextractor -c %s -MAG_ZEROPOINT %f -CATALOG_NAME %f -ASSOC_NAME %s -VERBOSE_TYPE QUIET %s'
        syscall = syscall % (defsexsub, subzp, subcat, refremapcat, sub)
        os.system(syscall)

        # Aperture catalog
        syscall = 'sextractor -c %s -MAG_ZEROPOINT %f -CATALOG_NAME %f -VERBOSE_TYPE QUIET %s,%s'
        syscall = syscall % (defsexaper, refzp, apercat, sub, refremap)
        os.system(syscall)
