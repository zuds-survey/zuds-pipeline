from pathlib import Path
import numpy as np


def initialize_directory(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)


def quick_background_estimate(image, nsamp=10000):
    # only requires that image.data and image.mask_image be defined
    # we only need a quick estimate of the bkg.
    # so we mask out any pixels where the MASK value is non zero.

    bkgpix = image.data[image.mask_image.data == 0]
    bkgpix = np.random.choice(bkgpix, size=nsamp)

    bkg = np.median(bkgpix)
    bkgstd = np.std(bkgpix)

    return bkg, bkgstd
