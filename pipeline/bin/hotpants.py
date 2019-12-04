import os
import db
import numpy as np
import shutil
import pandas as pd

from utils import initialize_directory, quick_background_estimate
from seeing import estimate_seeing


# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


def chunk(iterable, chunksize):
    isize = len(iterable)
    nchunks = isize // chunksize if isize % chunksize == 0 else isize // chunksize + 1
    for i in range(nchunks):
        yield i, iterable[i * chunksize : (i + 1) * chunksize]


def prepare_hotpants(sci, ref, outname, submask, directory,
                     copy_inputs=False, tmpdir='/tmp'):

    initialize_directory(directory)
    # this both creates and unmaps the background subtracted image
    sci.background_subtracted_image.load()
    old = sci.background_subtracted_image.local_path
    os.remove(old)
    bn = sci.background_subtracted_image.basename
    sci.background_subtracted_image.map_to_local_file(directory / bn)
    sci.background_subtracted_image.save()
    sci.background_subtracted_image.data += 100
    scimbkg = sci.background_subtracted_image

    # if requested, copy the input images to a temporary working directory
    if copy_inputs:
        impaths = []
        for image in [scimbkg, ref]:
            shutil.copy(image.local_path, directory)
            impaths.append(str(directory / image.basename))
    else:
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
    if not scirms.ismapped or copy_inputs:
        scirms_tmpnam = str((directory / scirms.basename).absolute())
        scirms.map_to_local_file(scirms_tmpnam)
        scirms.save()

    if not refrms.ismapped or copy_inputs:
        refrms_tmpnam = str((directory / refrms.basename).absolute())
        refrms.map_to_local_file(refrms_tmpnam)
        refrms.save()

    # we only need a quick estimate of the bkg.
    scibkg, scibkgstd = quick_background_estimate(scimbkg,
                                                  mask_image=sci.mask_image)
    refbkg, refbkgstd = quick_background_estimate(ref)

    il = scibkg - 10 * scibkgstd
    tl = refbkg - 10 * refbkgstd

    satlev = 5e4  # not perfect, but close enough.

    syscall = f'hotpants -inim {scipath} -hki -n i -c t ' \
              f'-tmplim {ref.local_path} -outim {outname} ' \
              f'-tu {satlev} -iu {satlev}  -tl {tl} -il {il} -r {r} ' \
              f'-rss {rss} -tni {refrms.local_path} ' \
              f'-ini {scirms.local_path} ' \
              f'-imi {submask.local_path} ' \
              f'-nsx {nsx} -nsy {nsy}'

    return syscall



