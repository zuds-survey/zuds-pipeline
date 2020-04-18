from pathlib import Path
import numpy as np


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
