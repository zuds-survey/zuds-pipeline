import numpy as np
from pathlib import Path

__all__ = ['initialize_directory', 'quick_background_estimate',
           'fid_map', '_split', 'print_time',
           'ensure_images_have_the_same_properties']


def initialize_directory(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)


def quick_background_estimate(image, nsamp=None, mask_image=None):
    # only requires that image.data and image.mask_image be defined
    # we only need a quick estimate of the bkg.
    # so we mask out any pixels where the MASK value is non zero.

    if mask_image is None:
        mask_image = image.mask_image

    bkgpix = image.data[mask_image.data == 0]

    if nsamp is not None:
        bkgpix = np.random.choice(bkgpix, size=nsamp)

    bkg = np.median(bkgpix)

    # use MAD as it is less sensitive to outliers (bright pixels)

    # some bright pixels may not be masked and can totally throw off
    # the bkgrms calculation if np.std is used
    bkgstd = 1.4826 * np.median(np.abs(bkgpix - bkg))

    return bkg, bkgstd


fid_map = {
    1: 'zg',
    2: 'zr',
    3: 'zi'
}


# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


def print_time(start, stop, detection, step):
    print(f'took {stop-start:.2f} sec to do {step} for {detection.id}',
          flush=True)


def ensure_images_have_the_same_properties(images, properties):
    """Raise a ValueError if images have different fid, ccdid, qid, or field."""
    for prop in properties:
        vals = np.asarray([getattr(image, prop) for image in images])
        if not all(vals == vals[0]):
            raise ValueError(f'To be coadded, images must all have the same {prop}. '
                             f'These images had: {[(image.id, getattr(image, prop)) for image in images]}.')


