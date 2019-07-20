import sep
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np


def phot_sex_auto(img_meas, stack_detection):

    with fits.open(img_meas) as hdul:
        hd = hdul[0].header
        meas_pix = hdul[0].data
        meas_wcs = WCS(hd)
        gain = hd['GAIN']

    meas_x, meas_y = meas_wcs.all_world2pix([[stack_detection.ra, stack_detection.dec]], 1)[0]

    # shape parameters
    det_a = stack_detection.a_image
    det_b = stack_detection.b_image
    det_theta = stack_detection.theta_image

    # do the photometry (FLUX_AUTO equivalent)
    kronrad, krflag = sep.kron_radius(meas_pix, meas_x, meas_y, det_a, det_b, det_theta, 6.0)
    flux, fluxerr, flag = sep.sum_ellipse(meas_pix, meas_x, meas_y, det_a, det_b, det_theta,
                                          2.5*kronrad, subpix=1, gain=gain)

    flag |= krflag  # combine flags into 'flag'

    r_min = 1.75  # minimum diameter = 3.5
    use_circle = kronrad * np.sqrt(det_a * det_b) < r_min
    cflux, cfluxerr, cflag = sep.sum_circle(meas_pix, meas_x[use_circle], meas_y[use_circle], r_min, subpix=1,
                                            gain=gain)
    flux[use_circle] = cflux
    fluxerr[use_circle] = cfluxerr
    flag[use_circle] = cflag

    # convert results to scalars
    flux = np.squeeze(flux)[()]
    fluxerr = np.squeeze(fluxerr)[()]

    return flux, fluxerr

