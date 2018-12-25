import sys, subprocess
import numpy as np
import argparse, sys, subprocess, os, re, glob, time

from astropy.io import fits
from astropy.io.ascii import SExtractor
from astropy.table import Column
from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import aperture_photometry

from scipy.optimize import minimize

CUTSIZE = 51 # pixels


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

    img = re.sub('cat', 'fits', cat)
    rms = re.sub('cat', 'rms.fits', cat)
    bpm = re.sub('cat', 'bpm.fits', cat)
    reg = re.sub('cat', 'reg', cat)

    hdu_img = fits.open(img)
    hdu_rms = fits.open(rms)
    hdu_bpm = fits.open(bpm)
    flx = hdu_img[0].data
    rms = hdu_rms[0].data
    bpm = hdu_bpm[0].data

    bpm[(flx == 1e-30)] = 256

    med = np.median(rms)
    medcut = med * 1.25

    print('Working on:', cat)
    print('Median: ', med)

    s = SExtractor()
    table = s.read(cat)

    last = table['X_IMAGE'].size
    print('Total number of candidates: ', last)

    pos = np.vstack((table['X_IMAGE'], table['Y_IMAGE'])).T
    positions = pos.tolist()

    see = hdu_img[0].header['SEEING']
    ubflx = 50000.

    good = np.ones(last, dtype=int)

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

    table['GOODCUT'][np.where(table['IMAFLAGS_ISO'] > 0)] = 0
    print('Number of candidates after external flag cut: ', np.sum(table['GOODCUT']))

    table['GOODCUT'][np.where(table['FLAGS'] > 2)] = 0
    print('Number of candidates after internal flag cut: ', np.sum(table['GOODCUT']))

    table['GOODCUT'][np.where(table['A_IMAGE'] / table['B_IMAGE'] > 2.0)] = 0
    print('Number of candidates after elipticity cuts: ', np.sum(table['GOODCUT']))

    table['GOODCUT'][np.where(table['FWHM_IMAGE'] / see > 2.0)] = 0
    print('Number of candidates after fwhm cuts: ', np.sum(table['GOODCUT']))

    table['GOODCUT'][np.where(table['BPMCUT'] > 0)] = 0
    print('Number of candidates after bpm cuts: ', np.sum(table['GOODCUT']))

    table['GOODCUT'][np.where(table['RMSCUT'] > medcut)] = 0
    print('Number of candidates after rms cuts: ', np.sum(table['GOODCUT']))

    """

    chisqs = []
    values = []

    imdata = hdu_img[0].data
    rmsdata = hdu_rms[0].data

    for row in table:

        if row['GOODCUT'] > 0.0 :

            xsex = np.round(row['X_IMAGE']).astype(int)
            ysex = np.round(row['Y_IMAGE']).astype(int)


            xsex += -1
            ysex += -1

            yslice = slice(ysex - CUTSIZE // 2, ysex + CUTSIZE // 2)
            xslice = slice(xsex - CUTSIZE // 2, xsex + CUTSIZE // 2)

            imcutout = imdata[yslice, xslice]
            rmscutout = rmsdata[yslice, xslice]

            lbflx = 1.5*rmscutout.min()
            gsflx = imcutout.max()

            y, x = np.meshgrid(np.arange(ysex - CUTSIZE // 2, ysex + CUTSIZE // 2),
                               np.arange(xsex - CUTSIZE // 2, xsex + CUTSIZE // 2))

            def chi_squared(parameters):

                A, x0, y0, sigx = parameters
                yarg = (y - y0) * (y - y0)
                xarg = (x - x0) * (x - x0)
                arg = (yarg + xarg)/sigx**2
                yth = A * np.exp(-arg)
                ressig = (yth - imcutout) / rmscutout
                chisq = np.sum(ressig * ressig)

                return chisq

            guess = [gsflx,  row['X_IMAGE'], row['Y_IMAGE'], see/2.355]

            fitres = minimize(chi_squared, tuple(guess), method='L-BFGS-B',
                              bounds=[
                                  (lbflx, ubflx),
                                  (guess[1] - 0.5, guess[1] + 0.5),
                                  (guess[2] - 0.5, guess[2] + 0.5),
                                  (guess[3] * 0.70, guess[3] * 2.00),
                              ])
            chisq_opt = fitres.fun
            optimal_values = fitres.x
            chisqs.append(chisq_opt)
            values.append(optimal_values)

        else:

            chisqs.append(1000.0*(CUTSIZE**2 - 1))
            values.append([0.0, 0.0, 0.0, 0.0])

    redchisq = Column(np.asarray(chisqs) / (CUTSIZE**2 - 4))
    table.add_column(redchisq, name='REDCHISQ')

    #table['GOODCUT'][np.where(table['REDCHISQ'] > 1.2)] = 0
    #print('Number of candidates after chisq cut: ', np.sum(table['GOODCUT']))

    for i, value  in enumerate(['A', 'x0', 'y0', 'sigx']):
         table[value] = [ov[i] for ov in values]
    """

    table.write(cat.replace('cat', 'cat.out.fits'), format='fits', overwrite=True)

    f = open(reg, "w+")
    f.write("# Region file format: DS9 version 4.1\n")
    f.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
    f.write("image\n")

    for i in range(0, last):
        (x,y) = pos[i]
        if table['GOODCUT'][i] > 0.0 :
           f.write("circle(%s,%s,10) # width=2 color=green\n" % (x,y))
    #       print(table['NUMBER'][i], pos[i], redchisq[i]  )
        else:
           f.write("circle(%s,%s,10) # width=2 color=red\n" % (x,y))


if __name__ == '__main__':

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('cat', help='the catalog to read in')
    args = parser.parse_args()
    filter_sexcat(args.cat)
