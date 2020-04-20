from .utils import initialize_directory, quick_background_estimate
from .seeing import estimate_seeing
from .constants import BIG_RMS

__all__ = ['prepare_hotpants']


def chunk(iterable, chunksize):
    isize = len(iterable)
    nchunks = isize // chunksize if isize % chunksize == 0 else isize // chunksize + 1
    for i in range(nchunks):
        yield i, iterable[i * chunksize : (i + 1) * chunksize]


def prepare_hotpants(sci, ref, outname, submask, directory,  tmpdir='/tmp',
                     refined=False):

    from .sextractor import run_sextractor
    from .swarp import BKG_VAL

    initialize_directory(directory)
    # this both creates and unmaps the background subtracted image

    scimbkg = run_sextractor(sci, checkimage_type=['bkgsub'])[1]
    scimbkg.data += BKG_VAL
    scimbkg.save()

    # if requested, copy the input images to a temporary working directory
    impaths = [im.local_path for im in [scimbkg, ref]]
    scipath, refpath = impaths

    if 'SEEING' not in sci.header:
        estimate_seeing(sci)
        sci.save()

    seepix = sci.header['SEEING']  # header seeing is FWHM in pixels
    r = 2.5 * seepix
    rss = 6. * seepix

    nsx = sci.header['NAXIS1'] / 100.
    nsy = sci.header['NAXIS2'] / 100.

    # get the background for the input images
    scirms = sci.rms_image
    refrms = ref.parent_image.rms_image.aligned_to(scirms, tmpdir=tmpdir)

    # save temporary copies of rms images if necessary
    if not scirms.ismapped:
        scirms_tmpnam = str((directory / scirms.basename).absolute())
        scirms.map_to_local_file(scirms_tmpnam)
        scirms.save()

    if not refrms.ismapped:
        refrms_tmpnam = str((directory / refrms.basename).absolute())
        refrms.map_to_local_file(refrms_tmpnam)
        refrms.save()

    # we only need a quick estimate of the bkg.
    scibkg, scibkgstd = quick_background_estimate(scimbkg,
                                                  mask_image=sci.mask_image)
    refbkg, refbkgstd = quick_background_estimate(ref)

    # output the RMS image from hotpants
    subrms = outname.replace('.fits', '.rms.fits')

    il = scibkg - 10 * scibkgstd
    tl = refbkg - 10 * refbkgstd

    satlev = 5e3  # not perfect, but close enough.

    syscall = f'hotpants -inim {scipath} -hki -n i -c t ' \
              f'-tmplim {ref.local_path} -outim {outname} ' \
              f'-tu {satlev} -iu {satlev}  -tl {tl} -il {il} -r {r} ' \
              f'-rss {rss} -tni {refrms.local_path} ' \
              f'-ini {scirms.local_path} ' \
              f'-imi {submask.local_path}  -v 0 -oni {subrms} ' \
              f'-fin {BIG_RMS} '
    if not refined:
        syscall += f'-nsx {nsx} -nsy {nsy}'
    else:
        syscall += f'-nsx {nsx / 3} -nsy {nsy / 3} ' \
                   f' -ko 4 -bgo 0 -nrx 3 -nry 3'

    return syscall



