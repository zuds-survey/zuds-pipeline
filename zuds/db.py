import os
import sncosmo
import requests
import numpy as np
from numpy.lib import recfunctions
from pathlib import Path
import shutil

import time

from datetime import datetime

from .secrets import get_secret
from .photometry import aperture_photometry, APER_KEY
from .swarp import ensure_images_have_the_same_properties, run_coadd, run_align
from . import archive
from .hotpants import prepare_hotpants
from .filterobjects import filter_sexcat, BRAAI_MODEL
from .seeing import estimate_seeing

from . import sextractor

import requests

import fitsio
import subprocess
import uuid
import warnings
import pandas as pd

import photutils
from astropy.io import fits
from astropy.wcs import WCS
from astropy import convolution
from astropy.coordinates import SkyCoord

import publish




TABLE_COLUMNS = ['id', 'xcentroid', 'ycentroid', 'sky_centroid',
                 'sky_centroid_icrs', 'source_sum', 'source_sum_err',
                 'orientation', 'eccentricity', 'semimajor_axis_sigma',
                 'semiminor_axis_sigma']


SEXTRACTOR_EQUIVALENTS = ['NUMBER', 'XWIN_IMAGE', 'YWIN_IMAGE', 'X_WORLD',
                          'Y_WORLD', 'FLUX_APER', 'FLUXERR_APER',
                          'THETA_WORLD', 'ELLIPTICITY', 'A_IMAGE', 'B_IMAGE']

















# Detections & Photometry #####################################################




class FilterRun(models.Base):
    tstart = sa.Column(sa.DateTime)
    tend = sa.Column(sa.DateTime)
    status = sa.Column(sa.Boolean, default=None)
    reason = sa.Column(sa.Text, nullable=True)


class Fit(models.Base):
    success = sa.Column(sa.Boolean)
    message = sa.Column(sa.Text)
    ncall = sa.Column(sa.Integer)
    chisq = sa.Column(sa.Float)
    ndof = sa.Column(sa.Integer)
    param_names = sa.Column(psql.ARRAY(sa.Text))
    parameters = sa.Column(models.NumpyArray)
    vparam_names = sa.Column(psql.ARRAY(sa.Text))
    covariance = sa.Column(models.NumpyArray)
    errors = sa.Column(psql.JSONB)
    nfit = sa.Column(sa.Integer)
    data_mask = sa.Column(psql.ARRAY(sa.Boolean))
    source_id = sa.Column(sa.Text,
                          sa.ForeignKey('sources.id', ondelete='SET NULL'))
    source = relationship('Source')

    @property
    def model(self):
        mod = sncosmo.Model(source='salt2-extended')
        for p, n in zip(self.parameters, self.param_names):
            mod[n] = p
        return mod


models.Source.fits = relationship('Fit', cascade='all')




from sqlalchemy import event


@event.listens_for(DBSession(), 'before_flush')
def bump_modified(session, flush_context, instances):
    for object in session.dirty:
        if isinstance(object, models.Base) and session.is_modified(object):
            object.modified = sa.func.now()

from copy import deepcopy
