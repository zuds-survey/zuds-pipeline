import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as psql
import numpy as np
from sqlalchemy.orm import relationship, column_property
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.ext.declarative import declared_attr

from sqlalchemy import Index
from sqlalchemy import func
import os
from skyportal import models
from skyportal.models import (join_model, DBSession, ACL,
                              Role, User, Token, Group, init_db as idb)
from skyportal.model_util import create_tables, drop_tables
from sqlalchemy.ext.hybrid import hybrid_property
import sncosmo
from photometry import aperture_photometry
from astropy.io import fits

from astropy.coordinates import SkyCoord
from astropy.table import Table
import requests
from secrets import get_secret
import photutils

from astropy.wcs import WCS

BKG_BOX_SIZE = 64


def init_db():
    hpss_dbhost = get_secret('hpss_dbhost')
    hpss_dbport = get_secret('hpss_dbport')
    hpss_dbusername = get_secret('hpss_dbusername')
    hpss_dbname = get_secret('hpss_dbname')
    hpss_dbpassword = get_secret('hpss_dbpassword')
    return idb(hpss_dbusername, hpss_dbname, hpss_dbpassword, hpss_dbhost, hpss_dbport)


class SpatiallyIndexed(object):
    ra = sa.Column(psql.DOUBLE_PRECISION)
    dec = sa.Column(psql.DOUBLE_PRECISION)

    @declared_attr
    def __table_args__(cls):
        tn = cls.__tablename__
        return sa.Index(f'{tn}_q3c_ang2ipix_idx', sa.func.q3c_ang2ipix(cls.ra, cls.dec)),


class UnmappedFileError(FileNotFoundError):
    pass


class File(object):
    """Abstract representation of a file"""
    basename = sa.Column(sa.Text)


class MappableToLocalFilesystem(File):

    @property
    def local_path(self):
        try:
            return self._path
        except AttributeError:
            errormsg = f'File "{self.basename}" is not mapped to the local file system. ' \
                       f'Identify the file corresponding to this object on the local file system, ' \
                       f'then call the `map_to_local_file` to identify the path.'
            raise UnmappedFileError(errormsg)

    def map_to_local_file(self, path):
        self._path = path


class HasFITSHeader(File):
    header = sa.Column(psql.JSONB)

    @classmethod
    def from_file(cls, f):
        obj = cls()
        with fits.open(f) as hdul:
            hd = dict(hdul[0].header)
        hd2 = hd.copy()
        for k in hd:
            if not isinstance(hd[k], (int, str, bool, float)):
                del hd2[k]
        obj.header = hd2
        obj.basename = os.path.basename(f)
        return obj

    @property
    def astropy_header(self):
        header = fits.Header()
        header.update(self.header)
        return header


class CanResideOnTape(File):
    tarball = sa.Column(sa.Text)


class FileServedViaHTTP(MappableToLocalFilesystem):
    url = sa.Column(sa.Text)

    def get(self):
        with open(self.basename, 'wb') as f:
            r = requests.get(self.url)
            r.raise_for_status()
            f.write(r.content)


class HasPoly(object):
    ra1 = sa.Column(psql.DOUBLE_PRECISION)
    dec1 = sa.Column(psql.DOUBLE_PRECISION)
    ra2 = sa.Column(psql.DOUBLE_PRECISION)
    dec2 = sa.Column(psql.DOUBLE_PRECISION)
    ra3 = sa.Column(psql.DOUBLE_PRECISION)
    dec3 = sa.Column(psql.DOUBLE_PRECISION)
    ra4 = sa.Column(psql.DOUBLE_PRECISION)
    dec4 = sa.Column(psql.DOUBLE_PRECISION)

    @hybrid_property
    def poly(self):
        return array((self.ra1, self.dec1, self.ra2, self.dec2,
                      self.ra3, self.dec3, self.ra4, self.dec4))


class HasWCS(HasFITSHeader, HasPoly, SpatiallyIndexed):

    @property
    def wcs(self):
        try:
            return self._wcs
        except AttributeError:
            self._wcs = WCS(self.astropy_header)
        return self._wcs

    @classmethod
    def from_file(cls, fname):
        self = super(HasWCS, cls).from_file(fname)
        corners = self.wcs.calc_footprint()
        for i, row in enumerate(corners):
            setattr(self, f'ra{i+1}', row[0])
            setattr(self, f'dec{i+1}', row[1])
        naxis1 = self.header['NAXIS1']
        naxis2 = self.header['NAXIS2']
        self.ra, self.dec = self.wcs.all_pix2world([[naxis1 / 2, naxis2 / 2]], 1)[0]
        return self

    @property
    def sources_contained(self):
        return DBSession().query(models.Source) \
            .filter(func.q3c_poly_query(models.Source.ra, models.Source.dec, self.poly))


class IPACRecord(models.Base, SpatiallyIndexed, HasPoly):
    __tablename__ = 'ipacrecords'

    created_at = sa.Column(sa.DateTime(), nullable=True, default=func.now())
    path = sa.Column(sa.Text, unique=True)
    filtercode = sa.Column(sa.CHAR(2))
    qid = sa.Column(sa.Integer)
    field = sa.Column(sa.Integer)
    ccdid = sa.Column(sa.Integer)
    obsjd = sa.Column(psql.DOUBLE_PRECISION)
    good = sa.Column(sa.Boolean)
    hasvariance = sa.Column(sa.Boolean)

    infobits = sa.Column(sa.Integer)
    fid = sa.Column(sa.Integer)
    rcid = sa.Column(sa.Integer)
    pid = sa.Column(psql.BIGINT)
    nid = sa.Column(sa.Integer)
    expid = sa.Column(sa.Integer)
    itid = sa.Column(sa.Integer)
    obsdate = sa.Column(sa.DateTime)
    seeing = sa.Column(sa.Float)
    airmass = sa.Column(sa.Float)
    moonillf = sa.Column(sa.Float)
    moonesb = sa.Column(sa.Float)
    maglimit = sa.Column(sa.Float)
    crpix1 = sa.Column(sa.Float)
    crpix2 = sa.Column(sa.Float)
    crval1 = sa.Column(sa.Float)
    crval2 = sa.Column(sa.Float)
    cd11 = sa.Column(sa.Float)
    cd12 = sa.Column(sa.Float)
    cd21 = sa.Column(sa.Float)
    cd22 = sa.Column(sa.Float)
    ipac_pub_date = sa.Column(sa.DateTime)
    ipac_gid = sa.Column(sa.Integer)
    imgtypecode = sa.Column(sa.CHAR(1))
    exptime = sa.Column(sa.Float)
    filefracday = sa.Column(psql.BIGINT)
    fcqfo = Index("image_field_ccdid_qid_filtercode_obsjd_idx", field, ccdid, qid, filtercode, obsjd)

    science_image = relationship('ScienceImage', cascade='all')

    @hybrid_property
    def obsmjd(self):
        return self.obsjd - 2400000.5

    @hybrid_property
    def filter(self):
        return 'ztf' + self.filtercode[-1]

    def ipac_path(self, suffix):
        sffd = str(self.filefracday)
        return f'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{sffd[:4]}/{sffd[4:8]}/{sffd[8:]}/' \
               f'ztf_{sffd}_{self.field:06d}_{self.filtercode}_c{self.ccdid:02d}_' \
               f'{self.imgtypecode}_q{self.qid}_{suffix}'


class PipelineFITSProduct(models.Base, HasFITSHeader, FileServedViaHTTP):
    # this is the polymorphic column
    type = sa.Column(sa.Text)

    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'fitsproduct'

    }


class MaskImage(PipelineFITSProduct):
    id = sa.Column(sa.Integer, sa.ForeignKey('pipelinefitsproducts.id'), primary_key=True)
    __mapper_args__ = {
        'polymorphic_identity': 'mask',
        'inherit_condition': id == PipelineFITSProduct.id
    }
    parent_image_id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id', ondelete='CASCADE'))
    parent_image = relationship('CalibratableImage', cascade='all', foreign_keys=[parent_image_id])


class RMSImage(PipelineFITSProduct):
    id = sa.Column(sa.Integer, sa.ForeignKey('pipelinefitsproducts.id'), primary_key=True)
    __mapper_args__ = {
        'polymorphic_identity': 'rms',
        'inherit_condition': id == PipelineFITSProduct.id
    }
    parent_image_id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id', ondelete='CASCADE'))
    parent_image = relationship('CalibratableImage', cascade='all', foreign_keys=[parent_image_id])


class CalibratableImage(PipelineFITSProduct, HasWCS):

    id = sa.Column(sa.Integer, sa.ForeignKey('pipelinefitsproducts.id'), primary_key=True)

    __mapper_args__ = {'polymorphic_identity': 'calibratableimage',
                       'inherit_condition': id == PipelineFITSProduct.id}

    detections = relationship('Detection', cascade='all')
    objects = relationship('ObjectWithFlux', cascade='all')

    rms_image_id = sa.Column(sa.Integer, sa.ForeignKey('rmsimages.id', ondelete='RESTRICT'),
                             nullable=True)
    rms_image = relationship('RMSImage', foreign_keys=[rms_image_id])

    mask_image_id = sa.Column(sa.Integer, sa.ForeignKey('maskimages.id', ondelete='RESTRICT'),
                              nullable=True)
    mask_image = relationship('MaskImage', foreign_keys=[mask_image_id])

    @property
    def bkg(self):
        try:
            return self._background
        except AttributeError:
            # calculate the background
            with fits.open(self.rms_image.local_path) as hdul:
                bg = photutils.Background2D(hdul[0].data, box_size=BKG_BOX_SIZE)
            self._background = bg
        return self._background

    @property
    def mask(self):
        try:
            return self._mask
        except AttributeError:
            # load the mask into memory
            with fits.open(self.mask_image.local_path) as hdul:
                maskpix = hdul[0].data
            self._mask = maskpix
        return self._mask

    @property
    def data(self):
        try:
            return self._data
        except AttributeError:
            # load the data into memory
            with fits.open(self.local_path) as hdul:
                data = hdul[0].data
            self._data = data
        return self._data


class CalibratedImage(CalibratableImage):
    """An image on which photometry can be performed."""

    id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id'), primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'calibratedimage',
                       'inherit_condition': id == CalibratableImage.id}

    forced_photometry = relationship('ForcedPhotometry', cascade='all')

    def force_photometry(self, sources):
        """Force aperture photometry at the locations of `sources`.
        Assumes that calibration has already been done.

        """

        # ensure sources is at least 1d
        sources = np.atleast_1d(sources)

        ra = [source.ra for source in sources]
        dec = [source.dec for source in sources]

        result = aperture_photometry(self, ra, dec, apply_calibration=True)

        photometry = []
        for row, source in zip(result, sources):
            phot = ForcedPhotometry(flux=row['flux'],
                                    fluxerr=row['fluxerr'],
                                    status=row['status'],
                                    reason=row['reason'],
                                    image=self,
                                    source=source)
            photometry.append(phot)

        return photometry

    @declared_attr
    def __table_args__(cls):
        # override spatial index - only need it on one table (calibratable images)
        return tuple()


class ScienceImage(CalibratedImage):

    id = sa.Column(sa.Integer, sa.ForeignKey('calibratedimages.id'), primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'sci',
                       'inherit_condition': id == CalibratedImage.id}

    ipac_record_id = sa.Column(sa.Integer, sa.ForeignKey('ipacrecords.id'), index=True)
    ipac_record = relationship('IPACRecord', back_populates='science_image')


# Coadds #################################################################################################

class Coadd(CalibratableImage):
    id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id'), primary_key=True)
    __mapper_args__ = {
        'polymorphic_identity': 'coadd',
        'inherit_condition': id == CalibratableImage.id
    }

    science_images = relationship('ScienceImage', secondary='coadd_scienceimages', cascade='all')

    @declared_attr
    def __table_args__(cls):
        return tuple()


CoaddScienceImages = join_model('coadd_scienceimages', Coadd, ScienceImage)


class ReferenceImage(Coadd):
    id = sa.Column(sa.Integer, sa.ForeignKey('coadds.id'), primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'ref',
                       'inherit_condition': id == Coadd.id}

    field = sa.Column(sa.Integer)
    fid = sa.Column(sa.Integer)
    qid = sa.Column(sa.Integer)
    ccdid = sa.Column(sa.Integer)
    idx = sa.Index('reference_field_ccdid_qid_fid', field, ccdid, qid, fid)

    single_epoch_subtractions = relationship('SingleEpochSubtraction', cascade='all')
    multi_epoch_subtractions = relationship('MultiEpochSubtraction', cascade='all')


class ScienceCoadd(Coadd):
    id = sa.Column(sa.Integer, sa.ForeignKey('coadds.id'), primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'scicoadd',
                       'inherit_condition': id == Coadd.id}
    subtraction = relationship('MultiEpochSubtraction', uselist=False, cascade='all')


# Subtractions #############################################################################################

class SingleEpochSubtraction(CalibratedImage):
    id = sa.Column(sa.Integer, sa.ForeignKey('calibratedimages.id'), primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'sesub',
                       'inherit_condition': id == CalibratedImage.id}

    science_image_id = sa.Column(sa.Integer, sa.ForeignKey('scienceimages.id', ondelete='CASCADE'), index=True)
    science_image = relationship('ScienceImage', cascade='all')

    reference_image_id = sa.Column(sa.Integer, sa.ForeignKey('referenceimages.id', ondelete='CASCADE'), index=True)
    reference_image = relationship('ReferenceImage', cascade='all')


class MultiEpochSubtraction(CalibratableImage):
    id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id'), primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'mesub',
                       'inherit_condition': id == CalibratableImage.id}
    science_coadd_id = sa.Column(sa.Integer, sa.ForeignKey('sciencecoadds.id', ondelete='CASCADE'), index=True)
    science_coadd = relationship('ScienceCoadd', cascade='all')

    reference_image_id = sa.Column(sa.Integer, sa.ForeignKey('referenceimages.id', ondelete='CASCADE'), index=True)
    reference_image = relationship('ReferenceImage', cascade='all')

    @declared_attr
    def __table_args__(cls):
        return tuple()



# Detections & Photometry ###################################################################################

class ObjectWithFlux(models.Base):
    type = sa.Column(sa.Text)

    __tablename__ = 'objectswithflux'
    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'base'
    }

    image_id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id', ondelete='CASCADE'), index=True)
    image = relationship('CalibratableImage', back_populates='objects', cascade='all')


    #thumbnails = relationship('Thumbnail', cascade='all')

    source_id = sa.Column(sa.Text, sa.ForeignKey('sources.id', ondelete='CASCADE'), index=True)
    source = relationship('Source', cascade='all')

    flux = sa.Column(sa.Float)
    fluxerr = sa.Column(sa.Float)

    @hybrid_property
    def snr(self):
        return self.flux / self.fluxerr


class Detection(ObjectWithFlux, SpatiallyIndexed):
    id = sa.Column(sa.Integer, sa.ForeignKey('objectswithflux.id'), primary_key=True)
    __tablename__ = 'detections'
    __mapper_args__ = {'polymorphic_identity': 'detection',
                       'inherit_condition': id == ObjectWithFlux.id}


class ForcedPhotometry(ObjectWithFlux):
    id = sa.Column(sa.Integer, sa.ForeignKey('objectswithflux.id'), primary_key=True)
    __tablename__ = 'forcedphotometry'
    __mapper_args__ = {'polymorphic_identity': 'photometry',
                       'inherit_condition': id == ObjectWithFlux.id}

    status = sa.Column(sa.Boolean)
    reason = sa.Column(sa.Text)

    @property
    def mag(self):
        return -2.5 * np.log10(self.flux) + self.image.header['MAGZP']

    @property
    def magerr(self):
        return 1.08573620476 * self.fluxerr / self.flux



class HPSSJob(models.Base):
    user = sa.Column(sa.Text)
    status = sa.Column(sa.Boolean, default=False)
    reason = sa.Column(sa.Text)


def images(self):
    candidates = DBSession().query(IPACRecord).filter(func.q3c_radial_query(IPACRecord.ra, IPACRecord.dec, self.ra, self.dec, 0.64))\
                                         .filter(func.q3c_poly_query(self.ra, self.dec, IPACRecord.poly))
    return candidates.all()


# keep track of the images that the photometry came from
"""
models.Photometry.image_id = sa.Column(sa.Integer, sa.ForeignKey('image.id', ondelete='CASCADE'), index=True)
models.Photometry.image = relationship('Image', back_populates='photometry')
models.Photometry.provenance = sa.Column(sa.Text)
models.Photometry.method = sa.Column(sa.Text)



models.Source.images = property(images)
models.Source.q3c = Index(f'sources_q3c_ang2ipix_idx', func.q3c_ang2ipix(models.Source.ra, models.Source.dec))
models.Source.stack_detections = relationship('StackDetection', cascade='all')

def best_stack_detection(self):
    sds = self.stack_detections

    # only keep stack detections that have shape parameters
    sds = [s for s in sds if (s.a_image is not None and s.b_image is not None and s.theta_image is not None)]
    return max(sds, key=lambda sd: sd.flux / sd.fluxerr)


models.Source.best_stack_detection = property(best_stack_detection)


def light_curve(self):
    photometry = self.photometry
    lc_raw = []

    for photpoint in photometry:
        photd = {'mjd': photpoint.mjd,
                 'filter': photpoint.filter,
                 'zp': photpoint.zp,
                 'zpsys': photpoint.zpsys,
                 'flux': photpoint.flux,
                 'fluxerr': photpoint.fluxerr}
        lc_raw.append(photd)

    return Table(lc_raw)

models.Source.light_curve = light_curve
"""

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
    source_id = sa.Column(sa.Text, sa.ForeignKey('sources.id', ondelete='SET NULL'))
    source = relationship('Source')


    @property
    def model(self):
        mod = sncosmo.Model(source='salt2-extended')
        for p, n in zip(self.parameters, self.param_names):
            mod[n] = p
        return mod


models.Source.fits = relationship('Fit', cascade='all')


class DR8(models.Base, SpatiallyIndexed):

    hemisphere = sa.Column(sa.Text)

    __tablename__ = 'dr8'
    __mapper_args__ = {
        'polymorphic_on': hemisphere,
        'polymorphic_identity': 'base'
    }

    def __repr__(self):
        attr_list = [f"{c.name.lower()}={getattr(self, c.name.lower())}"
                     for c in self.__table__.columns]
        return f"<{type(self).__name__}({', '.join(attr_list)})>"


    release = sa.Column('RELEASE', sa.Integer)
    brickid = sa.Column('BRICKID', sa.Integer)
    brickname = sa.Column('BRICKNAME', sa.Text)
    objid = sa.Column('OBJID', sa.Integer)
    type = sa.Column('TYPE', sa.Text)
    ra = sa.Column('RA', psql.DOUBLE_PRECISION)
    dec = sa.Column('DEC', psql.DOUBLE_PRECISION)
    ra_ivar = sa.Column('RA_IVAR', psql.DOUBLE_PRECISION)
    dec_ivar = sa.Column('DEC_IVAR', psql.DOUBLE_PRECISION)
    ebv = sa.Column('EBV', psql.DOUBLE_PRECISION)
    flux_g = sa.Column('FLUX_G', psql.DOUBLE_PRECISION)
    flux_r = sa.Column('FLUX_R', psql.DOUBLE_PRECISION)
    flux_z = sa.Column('FLUX_Z', psql.DOUBLE_PRECISION)
    flux_w1 = sa.Column('FLUX_W1', psql.DOUBLE_PRECISION)
    flux_w2 = sa.Column('FLUX_W2', psql.DOUBLE_PRECISION)
    flux_w3 = sa.Column('FLUX_W3', psql.DOUBLE_PRECISION)
    flux_w4 = sa.Column('FLUX_W4', psql.DOUBLE_PRECISION)
    flux_ivar_g = sa.Column('FLUX_IVAR_G', psql.DOUBLE_PRECISION)
    flux_ivar_r = sa.Column('FLUX_IVAR_R', psql.DOUBLE_PRECISION)
    flux_ivar_z = sa.Column('FLUX_IVAR_Z', psql.DOUBLE_PRECISION)
    flux_ivar_w1 = sa.Column('FLUX_IVAR_W1', psql.DOUBLE_PRECISION)
    flux_ivar_w2 = sa.Column('FLUX_IVAR_W2', psql.DOUBLE_PRECISION)
    flux_ivar_w3 = sa.Column('FLUX_IVAR_W3', psql.DOUBLE_PRECISION)
    flux_ivar_w4 = sa.Column('FLUX_IVAR_W4', psql.DOUBLE_PRECISION)
    mw_transmission_g = sa.Column('MW_TRANSMISSION_G', psql.DOUBLE_PRECISION)
    mw_transmission_r = sa.Column('MW_TRANSMISSION_R', psql.DOUBLE_PRECISION)
    mw_transmission_z = sa.Column('MW_TRANSMISSION_Z', psql.DOUBLE_PRECISION)
    mw_transmission_w1 = sa.Column('MW_TRANSMISSION_W1', psql.DOUBLE_PRECISION)
    mw_transmission_w2 = sa.Column('MW_TRANSMISSION_W2', psql.DOUBLE_PRECISION)
    mw_transmission_w3 = sa.Column('MW_TRANSMISSION_W3', psql.DOUBLE_PRECISION)
    mw_transmission_w4 = sa.Column('MW_TRANSMISSION_W4', psql.DOUBLE_PRECISION)
    nobs_g = sa.Column('NOBS_G', sa.Integer)
    nobs_r = sa.Column('NOBS_R', sa.Integer)
    nobs_z = sa.Column('NOBS_Z', sa.Integer)
    nobs_w1 = sa.Column('NOBS_W1', sa.Integer)
    nobs_w2 = sa.Column('NOBS_W2', sa.Integer)
    nobs_w3 = sa.Column('NOBS_W3', sa.Integer)
    nobs_w4 = sa.Column('NOBS_W4', sa.Integer)
    rchisq_g = sa.Column('RCHISQ_G', psql.DOUBLE_PRECISION)
    rchisq_r = sa.Column('RCHISQ_R', psql.DOUBLE_PRECISION)
    rchisq_z = sa.Column('RCHISQ_Z', psql.DOUBLE_PRECISION)
    rchisq_w1 = sa.Column('RCHISQ_W1', psql.DOUBLE_PRECISION)
    rchisq_w2 = sa.Column('RCHISQ_W2', psql.DOUBLE_PRECISION)
    rchisq_w3 = sa.Column('RCHISQ_W3', psql.DOUBLE_PRECISION)
    rchisq_w4 = sa.Column('RCHISQ_W4', psql.DOUBLE_PRECISION)
    fracflux_g = sa.Column('FRACFLUX_G', psql.DOUBLE_PRECISION)
    fracflux_r = sa.Column('FRACFLUX_R', psql.DOUBLE_PRECISION)
    fracflux_z = sa.Column('FRACFLUX_Z', psql.DOUBLE_PRECISION)
    fracflux_w1 = sa.Column('FRACFLUX_W1', psql.DOUBLE_PRECISION)
    fracflux_w2 = sa.Column('FRACFLUX_W2', psql.DOUBLE_PRECISION)
    fracflux_w3 = sa.Column('FRACFLUX_W3', psql.DOUBLE_PRECISION)
    fracflux_w4 = sa.Column('FRACFLUX_W4', psql.DOUBLE_PRECISION)
    fracmasked_g = sa.Column('FRACMASKED_G', psql.DOUBLE_PRECISION)
    fracmasked_r = sa.Column('FRACMASKED_R', psql.DOUBLE_PRECISION)
    fracmasked_z = sa.Column('FRACMASKED_Z', psql.DOUBLE_PRECISION)
    fracin_g = sa.Column('FRACIN_G', psql.DOUBLE_PRECISION)
    fracin_r = sa.Column('FRACIN_R', psql.DOUBLE_PRECISION)
    fracin_z = sa.Column('FRACIN_Z', psql.DOUBLE_PRECISION)
    anymask_g = sa.Column('ANYMASK_G', sa.Integer)
    anymask_r = sa.Column('ANYMASK_R', sa.Integer)
    anymask_z = sa.Column('ANYMASK_Z', sa.Integer)
    allmask_g = sa.Column('ALLMASK_G', sa.Integer)
    allmask_r = sa.Column('ALLMASK_R', sa.Integer)
    allmask_z = sa.Column('ALLMASK_Z', sa.Integer)
    wisemask_w1 = sa.Column('WISEMASK_W1', sa.Integer)
    wisemask_w2 = sa.Column('WISEMASK_W2', sa.Integer)
    psfsize_g = sa.Column('PSFSIZE_G', psql.DOUBLE_PRECISION)
    psfsize_r = sa.Column('PSFSIZE_R', psql.DOUBLE_PRECISION)
    psfsize_z = sa.Column('PSFSIZE_Z', psql.DOUBLE_PRECISION)
    psfdepth_g = sa.Column('PSFDEPTH_G', psql.DOUBLE_PRECISION)
    psfdepth_r = sa.Column('PSFDEPTH_R', psql.DOUBLE_PRECISION)
    psfdepth_z = sa.Column('PSFDEPTH_Z', psql.DOUBLE_PRECISION)
    galdepth_g = sa.Column('GALDEPTH_G', psql.DOUBLE_PRECISION)
    galdepth_r = sa.Column('GALDEPTH_R', psql.DOUBLE_PRECISION)
    galdepth_z = sa.Column('GALDEPTH_Z', psql.DOUBLE_PRECISION)
    psfdepth_w1 = sa.Column('PSFDEPTH_W1', psql.DOUBLE_PRECISION)
    psfdepth_w2 = sa.Column('PSFDEPTH_W2', psql.DOUBLE_PRECISION)
    wise_coadd_id = sa.Column('WISE_COADD_ID', sa.Text)
    fracdev = sa.Column('FRACDEV', psql.DOUBLE_PRECISION)
    fracdev_ivar = sa.Column('FRACDEV_IVAR', psql.DOUBLE_PRECISION)
    shapedev_r = sa.Column('SHAPEDEV_R', psql.DOUBLE_PRECISION)
    shapedev_r_ivar = sa.Column('SHAPEDEV_R_IVAR', psql.DOUBLE_PRECISION)
    shapedev_e1 = sa.Column('SHAPEDEV_E1', psql.DOUBLE_PRECISION)
    shapedev_e1_ivar = sa.Column('SHAPEDEV_E1_IVAR', psql.DOUBLE_PRECISION)
    shapedev_e2 = sa.Column('SHAPEDEV_E2', psql.DOUBLE_PRECISION)
    shapedev_e2_ivar = sa.Column('SHAPEDEV_E2_IVAR', psql.DOUBLE_PRECISION)
    shapeexp_r = sa.Column('SHAPEEXP_R', psql.DOUBLE_PRECISION)
    shapeexp_r_ivar = sa.Column('SHAPEEXP_R_IVAR', psql.DOUBLE_PRECISION)
    shapeexp_e1 = sa.Column('SHAPEEXP_E1', psql.DOUBLE_PRECISION)
    shapeexp_e1_ivar = sa.Column('SHAPEEXP_E1_IVAR', psql.DOUBLE_PRECISION)
    shapeexp_e2 = sa.Column('SHAPEEXP_E2', psql.DOUBLE_PRECISION)
    shapeexp_e2_ivar = sa.Column('SHAPEEXP_E2_IVAR', psql.DOUBLE_PRECISION)
    fiberflux_g = sa.Column('FIBERFLUX_G', psql.DOUBLE_PRECISION)
    fiberflux_r = sa.Column('FIBERFLUX_R', psql.DOUBLE_PRECISION)
    fiberflux_z = sa.Column('FIBERFLUX_Z', psql.DOUBLE_PRECISION)
    fibertotflux_g = sa.Column('FIBERTOTFLUX_G', psql.DOUBLE_PRECISION)
    fibertotflux_r = sa.Column('FIBERTOTFLUX_R', psql.DOUBLE_PRECISION)
    fibertotflux_z = sa.Column('FIBERTOTFLUX_Z', psql.DOUBLE_PRECISION)
    ref_cat = sa.Column('REF_CAT', sa.Text)
    ref_id = sa.Column('REF_ID', sa.Integer)
    ref_epoch = sa.Column('REF_EPOCH', psql.DOUBLE_PRECISION)
    gaia_phot_g_mean_mag = sa.Column('GAIA_PHOT_G_MEAN_MAG', psql.DOUBLE_PRECISION)
    gaia_phot_g_mean_flux_over_error = sa.Column('GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR', psql.DOUBLE_PRECISION)
    gaia_phot_bp_mean_mag = sa.Column('GAIA_PHOT_BP_MEAN_MAG', psql.DOUBLE_PRECISION)
    gaia_phot_bp_mean_flux_over_error = sa.Column('GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR', psql.DOUBLE_PRECISION)
    gaia_phot_rp_mean_mag = sa.Column('GAIA_PHOT_RP_MEAN_MAG', psql.DOUBLE_PRECISION)
    gaia_phot_rp_mean_flux_over_error = sa.Column('GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR', psql.DOUBLE_PRECISION)
    gaia_astrometric_excess_noise = sa.Column('GAIA_ASTROMETRIC_EXCESS_NOISE', psql.DOUBLE_PRECISION)
    gaia_duplicated_source = sa.Column('GAIA_DUPLICATED_SOURCE', sa.Boolean)
    gaia_phot_bp_rp_excess_factor = sa.Column('GAIA_PHOT_BP_RP_EXCESS_FACTOR', psql.DOUBLE_PRECISION)
    gaia_astrometric_sigma5d_max = sa.Column('GAIA_ASTROMETRIC_SIGMA5D_MAX', psql.DOUBLE_PRECISION)
    gaia_astrometric_params_solved = sa.Column('GAIA_ASTROMETRIC_PARAMS_SOLVED', sa.Integer)
    parallax = sa.Column('PARALLAX', psql.DOUBLE_PRECISION)
    parallax_ivar = sa.Column('PARALLAX_IVAR', psql.DOUBLE_PRECISION)
    pmra = sa.Column('PMRA', psql.DOUBLE_PRECISION)
    pmra_ivar = sa.Column('PMRA_IVAR', psql.DOUBLE_PRECISION)
    pmdec = sa.Column('PMDEC', psql.DOUBLE_PRECISION)
    pmdec_ivar = sa.Column('PMDEC_IVAR', psql.DOUBLE_PRECISION)
    maskbits = sa.Column('MASKBITS', sa.Integer)
    z_phot_mean = sa.Column('z_phot_mean', psql.DOUBLE_PRECISION)
    z_phot_median = sa.Column('z_phot_median', psql.DOUBLE_PRECISION)
    z_phot_std = sa.Column('z_phot_std', psql.DOUBLE_PRECISION)
    z_phot_l68 = sa.Column('z_phot_l68', psql.DOUBLE_PRECISION)
    z_phot_u68 = sa.Column('z_phot_u68', psql.DOUBLE_PRECISION)
    z_phot_l95 = sa.Column('z_phot_l95', psql.DOUBLE_PRECISION)
    z_phot_u95 = sa.Column('z_phot_u95', psql.DOUBLE_PRECISION)
    z_spec = sa.Column('z_spec', psql.DOUBLE_PRECISION)
    survey = sa.Column('survey', sa.Text)
    training = sa.Column('training', sa.Boolean)

    @hybrid_property
    def gmag(self):
        return -2.5 * np.log10(self.flux_g) + 22.5

    @gmag.expression
    def gmag(self):
        return -2.5 * sa.func.log(self.flux_g) + 22.5

    @hybrid_property
    def rmag(self):
        return -2.5 * np.log10(self.flux_r) + 22.5

    @rmag.expression
    def rmag(self):
        return -2.5 * sa.func.log(self.flux_r) + 22.5

    @hybrid_property
    def w1mag(self):
        return -2.5 * np.log10(self.flux_w1) + 22.5

    @w1mag.expression
    def w1mag(self):
        return -2.5 * sa.func.log(self.flux_w1) + 22.5


class DR8North(DR8):
    id = sa.Column(sa.Integer, sa.ForeignKey('dr8.id'), primary_key=True)
    __tablename__ = 'dr8_north'
    __mapper_args__ = {'polymorphic_identity': 'n'}

    @declared_attr
    def __table_args__(cls):
        return tuple()


class DR8South(DR8):
    id = sa.Column(sa.Integer, sa.ForeignKey('dr8.id'), primary_key=True)
    __tablename__ = 'dr8_south'
    __mapper_args__ = {'polymorphic_identity': 's'}

    @declared_attr
    def __table_args__(cls):
        return tuple()
