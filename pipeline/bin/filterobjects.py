import numpy as np
from astropy.table import Column
from photutils import CircularAperture
from photutils import aperture_photometry
from astropy.table import Table
import time
from seeing import estimate_seeing

import db

from scipy.optimize import minimize

CUTSIZE = 11 # pixels


# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


def _read_clargs(val):
    if val[0].startswith('@'):
        # then its a list
        val = np.genfromtxt(val[0][1:], dtype=None, encoding='ascii')
        val = np.atleast_1d(val)
    return np.asarray(val)


def filter_sexcat(cat):
    """Read in sextractor catalog `incat` and filter it using Peter's technique.
    Write the results to sextractor catalog `outcat`."""

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
    medcut = med * 1.18

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
    bad_bits = np.asarray([0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 16, 17])
    bad_bits = int(np.sum(2**bad_bits))

    start = time.time()

    table['GOODCUT'][np.where(table['IMAFLAGS_ISO'] & bad_bits > 0)] = 0
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

    start = time.time()
    cat.data = table.to_pandas().to_records(index=False)
    cat.save()
    stop = time.time()
    print(f'filter: {stop - start:.2f} sec to save {cat.basename} to disk')

    # make the region file
    db.PipelineRegionFile.from_catalog(cat)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('cat', help='the catalog to read in')
    args = parser.parse_args()
    filter_sexcat(args.cat)
