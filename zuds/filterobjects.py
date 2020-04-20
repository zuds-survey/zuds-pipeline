import numpy as np
from astropy.table import Column
from photutils import CircularAperture
from photutils import aperture_photometry
from astropy.table import Table
import time

import os
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord

from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import normalize as tf_norm

from .seeing import estimate_seeing
from .constants import BRAAI_MODEL, RB_CUT, BAD_SUM


__all__ = ['filter_sexcat']


CUTSIZE = 11 # pixels
old_norm = int(BRAAI_MODEL.split('d6_m')[1]) <= 7


def load_model_helper(path, model_base_name):
    """
        Build keras model using json-file with architecture and hdf5-file with weights
    """
    with open(os.path.join(path, f'{model_base_name}.architecture.json'), 'r') as json_file:
        loaded_model_json = json_file.read()
    m = model_from_json(loaded_model_json)
    m.load_weights(os.path.join(path, f'{model_base_name}.weights.h5'))

    return m

def _read_clargs(val):
    if val[0].startswith('@'):
        # then its a list
        val = np.genfromtxt(val[0][1:], dtype=None, encoding='ascii')
        val = np.atleast_1d(val)
    return np.asarray(val)


def make_triplet_for_braai(ra, dec, new_aligned, ref_aligned, sub_aligned,
                           old_norm=False):
    # aligned images are db.CalibratableImages that have north up, east left

    triplet = np.zeros((63, 63, 3))
    coord = SkyCoord(ra, dec, unit='deg')
    for i, img in enumerate([new_aligned, ref_aligned, sub_aligned]):
        cutout = Cutout2D(
            img.data, coord, size=63, mode='partial',
            fill_value=0., wcs=img.wcs
        )
        if old_norm:
            triplet[:, :, i] = tf_norm(cutout.data)
        else:
            triplet[:, :, i] = cutout.data / np.linalg.norm(cutout.data)
    return triplet


def filter_sexcat(cat):
    """Read in sextractor catalog `incat` and filter it using Peter's technique.
    Write the results to sextractor catalog `outcat`."""

    from .image import ScienceImage
    from .subtraction import SingleEpochSubtraction

    """python ./badpix.py sub*.cat"""

    if 'GOODCUT' in cat.data.dtype.names:
        return cat


    image = cat.image
    rms = image.rms_image
    bpm = image.mask_image.boolean
    table = Table(cat.data)

    rms = rms.data
    bpm = bpm.data

    med = np.median(rms[~bpm])
    medcut = med * 1.1

    last = table['X_IMAGE'].size
    print('Total number of candidates: ', last)

    pos = np.vstack((table['X_IMAGE'], table['Y_IMAGE'])).T
    positions = pos.tolist()

    if not 'SEEING' in image.astropy_header:
        start = time.time()
        estimate_seeing(image)
        stop = time.time()
        print(f'filter: {stop - start:.2f} sec to estimate '
              f'seeing for {cat.basename}', flush=True)
    see = image.astropy_header['SEEING']

    good = np.ones(last, dtype='uint8')

    good_cut = Column(good)
    table.add_column(good_cut, name='GOODCUT')

    area = np.pi * (6.0) ** 2

    apertures = CircularAperture(positions, r=6.0)

    rms_table = aperture_photometry(rms, apertures)
    bpm_table = aperture_photometry(bpm, apertures)

    rmsbig = rms_table['aperture_sum']
    bpmbig = bpm_table['aperture_sum']

    bpm_cut = Column(bpmbig)
    table.add_column(bpm_cut, name='BPMCUT')

    rms_cut = Column(rmsbig / area)
    table.add_column(rms_cut, name='RMSCUT')

    # the following bits are disqualifying:


    start = time.time()

    table['GOODCUT'][np.where(table['IMAFLAGS_ISO'] & BAD_SUM > 0)] = 0
    print('Number of candidates after external flag cut: ', np.sum(table['GOODCUT']))

    table['GOODCUT'][np.where(table['FLAGS'] > 2)] = 0
    print('Number of candidates after internal flag cut: ', np.sum(table['GOODCUT']))

    table['GOODCUT'][np.where(table['A_IMAGE'] / table['B_IMAGE'] > 2.0)] = 0
    print('Number of candidates after elipticity cuts: ', np.sum(table['GOODCUT']))

    table['GOODCUT'][np.where(table['FWHM_IMAGE'] / see > 2.0)] = 0
    print('Number of candidates after fwhm cuts: ', np.sum(table['GOODCUT']))

    table['GOODCUT'][np.where(table['FWHM_IMAGE'] < 0.8 * see)] = 0
    print('Number of candidates after sharp cuts: ', np.sum(table['GOODCUT']))

    table['GOODCUT'][np.where(table['BPMCUT'] > 0)] = 0
    print('Number of candidates after bpm cuts: ', np.sum(table['GOODCUT']))

    table['GOODCUT'][np.where(table['RMSCUT'] > medcut)] = 0
    print('Number of candidates after rms cuts: ', np.sum(table['GOODCUT']))

    table['GOODCUT'][np.where(table['FLUX_APER'] / table['FLUXERR_APER'] < 5)]\
        = 0
    print('Number of candidates after s/n > 5 cut: ', np.sum(table['GOODCUT']))

    stop = time.time()
    print(f'filter: {stop - start:.2f} sec to do initial cuts '
          f'for {cat.basename}', flush=True)

    start = time.time()

    # cut on anything with more than 3 10 sigma negative pixels in a 10x10 box
    imdata = image.data
    imsig = 1.48 * np.median(np.abs(imdata - np.median(imdata)))
    immed = np.median(imdata)
    for row in table:

        if row['GOODCUT'] > 0.0:

            xsex = np.round(row['X_IMAGE']).astype(int)
            ysex = np.round(row['Y_IMAGE']).astype(int)

            xsex += -1
            ysex += -1

            yslice = slice(ysex - CUTSIZE // 2, ysex + CUTSIZE // 2 + 1)
            xslice = slice(xsex - CUTSIZE // 2, xsex + CUTSIZE // 2 + 1)

            ybig = slice(ysex - CUTSIZE // 2 - 1, ysex + CUTSIZE // 2 + 2)
            xbig = slice(xsex - CUTSIZE // 2 - 1, xsex + CUTSIZE // 2 + 2)

            imcutout = imdata[yslice, xslice]
            bigcut = imdata[ybig, xbig]

            sigim = (imcutout - immed) / imsig
            sigbig = (bigcut - immed) / imsig

            neg5 = np.argwhere(sigim < -5.)
            for r, c in neg5:
                yneg = r + 1
                xneg = c + 1
                cutaround = sigbig[yneg - 1:yneg + 2, xneg - 1:xneg + 2]
                if (cutaround > 5).any():
                    row['GOODCUT'] = 0.
                    break

    stop = time.time()

    print('Number of candidates after negpix cut: ', np.sum(table['GOODCUT']))
    print(f'filter: {stop - start:.2f} sec to negpix cut for {cat.basename}')

    # machine learning

    start = time.time()
    new_aligned = None
    ref_aligned = image.input_images[0].reference_image
    sub_aligned = None
    ml_model = None

    table['rb'] = -99.

    for row in table:
        if row['GOODCUT'] > 0:

            # cache the images for stamp making
            if new_aligned is None:
                if isinstance(image.target_image, ScienceImage):
                    new_aligned = image.target_image.aligned_to(
                        image.reference_image)
                else:
                    new_aligned = image.target_image

            if sub_aligned is None:
                if isinstance(image, SingleEpochSubtraction):
                    sub_aligned = image.aligned_to(image.reference_image)
                else:
                    sub_aligned = image

            if ml_model is None:
                mydir = os.path.dirname(__file__)
                ml_model = load_model_helper(f'{mydir}/../ml', BRAAI_MODEL)

            # put it through machine learning
            triplet = make_triplet_for_braai(
                row['X_WORLD'], row['Y_WORLD'], new_aligned,
                ref_aligned, sub_aligned, old_norm=old_norm
            )

            rb = ml_model.predict(np.expand_dims(triplet, axis=0))
            row['rb'] = rb[0, 0]
            if row['rb'] < RB_CUT[image.fid]:
                row['GOODCUT'] = 0

    stop = time.time()
    print('Number of candidates after ML cut: ', np.sum(table['GOODCUT']))
    print(f'filter: {stop - start:.2f} sec to ML cut for {cat.basename}')

    start = time.time()
    cat.data = table.to_pandas().to_records(index=False)
    cat.save()
    stop = time.time()
    print(f'filter: {stop - start:.2f} sec to save {cat.basename} to disk')

    # make the region file
    #db.PipelineRegionFile.from_catalog(cat)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('cat', help='the catalog to read in')
    args = parser.parse_args()
    filter_sexcat(args.cat)
