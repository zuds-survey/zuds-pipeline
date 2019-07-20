import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as psql
import numpy as np

from sqlalchemy.orm import relationship, column_property
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.ext.declarative import declared_attr


from sqlalchemy import Index
from sqlalchemy import func

from pathlib import Path
import os
from skyportal import models
from skyportal.models import (init_db, join_model, DBSession, ACL,
                              Role, User, Token, Group)

from skyportal.model_util import create_tables, drop_tables
from sqlalchemy.ext.hybrid import hybrid_property

from photometry import phot_sex_auto

from astropy.io import fits
from libztf.yao import yao_photometry_single

from astropy.coordinates import SkyCoord

from astropy.table import Table

import requests

from download import safe_download


from baselayer.app.env import load_env
from datetime import datetime


class IPACProgram(models.Base):
    groups = relationship('Group', secondary='ipacprogram_groups',  back_populates='ipacprograms')#, cascade='all')
    images = relationship('Image', back_populates='ipac_program', cascade='all')


IPACProgramGroup = join_model('ipacprogram_groups', IPACProgram, Group)
Group.ipacprograms = relationship('IPACProgram', secondary='ipacprogram_groups', back_populates='groups', cascade='all')


class Image(models.Base):

    __tablename__ = 'image'

    created_at = sa.Column(sa.DateTime(), nullable=True, default=func.now())
    path = sa.Column(sa.Text, unique=True)
    filtercode = sa.Column(sa.CHAR(2))
    qid = sa.Column(sa.Integer)
    field = sa.Column(sa.Integer)
    ccdid = sa.Column(sa.Integer)
    obsjd = sa.Column(psql.DOUBLE_PRECISION)
    good = sa.Column(sa.Boolean)
    hasvariance = sa.Column(sa.Boolean)
    ra = sa.Column(psql.DOUBLE_PRECISION)
    dec = sa.Column(psql.DOUBLE_PRECISION)
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
    ra1 = sa.Column(psql.DOUBLE_PRECISION)
    dec1 = sa.Column(psql.DOUBLE_PRECISION)
    ra2 = sa.Column(psql.DOUBLE_PRECISION)
    dec2 = sa.Column(psql.DOUBLE_PRECISION)
    ra3 = sa.Column(psql.DOUBLE_PRECISION)
    dec3 = sa.Column(psql.DOUBLE_PRECISION)
    ra4 = sa.Column(psql.DOUBLE_PRECISION)
    dec4 = sa.Column(psql.DOUBLE_PRECISION)
    ipac_pub_date = sa.Column(sa.DateTime)
    ipac_gid = sa.Column(sa.Integer, sa.ForeignKey('ipacprograms.id', ondelete='RESTRICT'))
    imgtypecode = sa.Column(sa.CHAR(1))
    exptime = sa.Column(sa.Float)
    filefracday = sa.Column(psql.BIGINT)

    hpss_sci_path = sa.Column(sa.Text)
    hpss_mask_path = sa.Column(sa.Text)
    hpss_sub_path = sa.Column(sa.Text)
    hpss_psf_path = sa.Column(sa.Text)

    disk_sci_path = sa.Column(sa.Text)
    disk_mask_path = sa.Column(sa.Text)
    disk_sub_path = sa.Column(sa.Text)
    disk_psf_path = sa.Column(sa.Text)

    instrument_id = sa.Column(sa.Integer, sa.ForeignKey('instruments.id', ondelete='RESTRICT'), default=1)
    instrument = relationship('Instrument')

    zp = sa.Column(sa.Float, default=None, nullable=True)
    zpsys = sa.Column(sa.Text, default='ab')

    subtraction_exists = sa.Column(sa.Boolean)

    q3c = Index(f'image_q3c_ang2ipix_idx', func.q3c_ang2ipix(ra, dec))

    def ipac_path(self, suffix):
        sffd = str(self.filefracday)
        return f'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{sffd[:4]}/{sffd[4:8]}/{sffd[8:]}/' \
               f'ztf_{sffd}_{self.field:06d}_{self.filtercode}_c{self.ccdid:02d}_' \
               f'{self.imgtypecode}_q{self.qid}_{suffix}'

    def disk_path(self, suffix):
        sffd = str(self.filefracday)
        base = Path(os.getenv('OUTPUT_DIRECTORY')) / \
               f'{self.field:06d}/c{self.ccdid:02d}/q{self.qid}/{self.filtercode}/' \
               f'ztf_{sffd}_{self.field:06d}_{self.filtercode}_c{self.ccdid:02d}_' \
               f'{self.imgtypecode}_q{self.qid}_{suffix}'

        return f'{base}'

    def hpss_staging_path(self, suffix):
        sffd = str(self.filefracday)
        base = Path(os.getenv('STAGING_DIRECTORY')) / \
               f'{self.field:06d}/c{self.ccdid:02d}/q{self.qid}/{self.filtercode}/' \
               f'ztf_{sffd}_{self.field:06d}_{self.filtercode}_c{self.ccdid:02d}_' \
               f'{self.imgtypecode}_q{self.qid}_{suffix}'

        return f'{base}'


    fcqfo = Index("image_field_ccdid_qid_filtercode_obsjd_idx",  field, ccdid, qid, filtercode, obsjd)
    hmi = Index("image_hpss_mask_path_idx", hpss_mask_path)
    hpi = Index("image_hpss_psf_path_idx", hpss_psf_path)
    hshmi = Index("image_hpss_sci_path_hpss_mask_path_idx" ,hpss_sci_path, hpss_mask_path)
    hsci = Index("image_hpss_sci_path_idx" ,hpss_sci_path)
    hsubi = Index("image_hpss_sub_path_idx", hpss_sub_path)
    obsjdi = Index("image_obsjd_idx", obsjd)
    pathi = Index("image_path_idx", path)

    #groups = relationship('Group', back_populates='images', secondary='join(IPACProgram, ipacprogram_groups).join(groups)')
    ipac_program = relationship('IPACProgram', back_populates='images')
    photometry = relationship('Photometry', cascade='all')

    subtraction = relationship('SingleEpochSubtraction', back_populates='image', cascade='all')
    references = relationship('Reference', back_populates='images', cascade='all', secondary='reference_images')
    stacks = relationship('Stack', back_populates='images', cascade='all', secondary='stack_images')

    @hybrid_property
    def poly(self):
        return array([self.ra1, self.dec1, self.ra2, self.dec2,
                      self.ra3, self.dec3, self.ra4, self.dec4])

    @hybrid_property
    def obsmjd(self):
        return self.obsjd - 2400000.5

    @hybrid_property
    def filter(self):
        return 'ztf' + self.filtercode[-1]

    @property
    def sources(self):
        return DBSession().query(models.Source)\
                          .filter(func.q3c_poly_query(models.Source.ra, models.Source.dec, self.poly))\
                          .all()

    def provided_photometry(self, photometry):
        return photometry in self.photometry

    def force_photometry(self):

        sources_contained = self.sources
        sources_contained_ids = [s.id for s in sources_contained]
        photometered_sources = list(set([phot_point.source for phot_point in self.photometry]))

        # this must be list or setdiff1d will fail
        photometered_source_ids = list(set([s.id for s in photometered_sources]))

        # reject sources where photometry has already been done
        sources_remaining_ids = np.setdiff1d(sources_contained_ids, photometered_source_ids)
        sources_remaining = [s for s in sources_contained if s.id in sources_remaining_ids]

        # get the paths to relevant files on disk
        psf_path = self.disk_psf_path
        sub_path = self.disk_sub_path

        if psf_path.endswith('sciimgdaopsfcent.fits'):
            self.disk_psf_path = sub_path.replace('scimrefdiffimg.fits.fz', 'diffimgpsf.fits')
            psf_path = self.disk_psf_path
            DBSession().add(self)
            DBSession().commit()

        if self.zp is None:
            try:
                with fits.open(sub_path) as hdul:
                    header = hdul[1].header
            except OSError:  # ipac didn't make a subtraction
                self.disk_sub_path = None
                self.disk_psf_path = None
                self.subtraction_exists = False
                DBSession().add(self)
                DBSession().commit()
                raise FileNotFoundError(f'Subtraction for "{self.path}" does not exist or is not on disk.')
            except ValueError:
                raise FileNotFoundError(f'Subtraction for "{self.path}" does not exist or is not on disk.')
            else:
                self.zp = header['MAGZP']
                self.zpsys = 'ab'
                self.subtraction_exists = True
                DBSession().add(self)
                DBSession().commit()

        if self.instrument is None:
            self.instrument_id = 1
            DBSession().add(self)
            DBSession().commit()


        # for all the remaining sources do forced photometry

        new_photometry = []

        for source in sources_remaining:
            try:
                pobj = yao_photometry_single(sub_path, psf_path, source.ra, source.dec)
            except IndexError:
                continue
            phot_point = models.Photometry(image=self, flux=float(pobj.Fpsf), fluxerr=float(pobj.eFpsf),
                                           zp=self.zp, zpsys=self.zpsys, lim_mag=self.maglimit,
                                           filter=self.filter, source=source, instrument=self.instrument,
                                           ra=source.ra, dec=source.dec, mjd=self.obsmjd, provenance='ipac',
                                           method='yao')
            new_photometry.append(phot_point)

        DBSession().add_all(new_photometry)
        DBSession().commit()


class HPSSJob(models.Base):
    user = sa.Column(sa.Text)
    status = sa.Column(sa.Boolean, default=False)
    reason = sa.Column(sa.Text)


class StackDetection(models.Base):

    ra = sa.Column(psql.DOUBLE_PRECISION)
    dec = sa.Column(psql.DOUBLE_PRECISION)

    subtraction_id = sa.Column(sa.Integer, sa.ForeignKey('multiepochsubtractions.id', ondelete='CASCADE'), index=True)
    subtraction = relationship('MultiEpochSubtraction', back_populates='detections', cascade='all')

    flux = sa.Column(sa.Float)
    fluxerr = sa.Column(sa.Float)
    zp = sa.Column(sa.Float)
    zpsys = sa.Column(sa.Text)
    maglimit = sa.Column(sa.Float)
    filter = sa.Column(sa.Text)

    mjd = sa.Column(sa.Float)
    source_id = sa.Column(sa.Text, sa.ForeignKey('sources.id', ondelete='CASCADE'), index=True)
    source = relationship('Source', back_populates='stack_detections', cascade='all')

    thumbnails = relationship('StackThumbnail', cascade='all')

    provenance = sa.Column(sa.Text)
    method = sa.Column(sa.Text)
    a_image = sa.Column(sa.Float)
    b_image = sa.Column(sa.Float)
    theta_image = sa.Column(sa.Float)

    photometry = relationship('Photometry', cascade='all')
    q3c = Index('stackdetections_q3c_ang2ipix_idx', func.q3c_ang2ipix(ra, dec))

    @hybrid_property
    def mag(self):
        return -2.5 * sa.func.log(self.flux) + self.zp

    @hybrid_property
    def magerr(self):
        return 1.08573620476 * self.fluxerr / self.flux


def images(self):
    candidates = DBSession().query(Image).filter(func.q3c_radial_query(Image.ra, Image.dec, self.ra, self.dec, 0.64))\
                                         .filter(func.q3c_poly_query(self.ra, self.dec, Image.poly))
    return candidates.all()


# keep track of the images that the photometry came from
models.Photometry.image_id = sa.Column(sa.Integer, sa.ForeignKey('image.id', ondelete='CASCADE'), index=True)
models.Photometry.image = relationship('Image', back_populates='photometry')
models.Photometry.provenance = sa.Column(sa.Text)
models.Photometry.method = sa.Column(sa.Text)

models.Source.images = property(images)
models.Source.q3c = Index(f'sources_q3c_ang2ipix_idx', func.q3c_ang2ipix(models.Source.ra, models.Source.dec))
models.Source.stack_detections = relationship('StackDetection', cascade='all')

def best_stack_detection(self):
    sds = self.stack_detections
    return max(sds, key=lambda sd: sd.flux / sd.flux_err)


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


class FilterRun(models.Base):
    tstart = sa.Column(sa.DateTime)
    tend = sa.Column(sa.DateTime)
    status = sa.Column(sa.Boolean, default=None)
    reason = sa.Column(sa.Text, nullable=True)


class PittObject(models.Base):

    type = sa.Column(sa.Text)
    ra = sa.Column(psql.DOUBLE_PRECISION)
    dec = sa.Column(psql.DOUBLE_PRECISION)
    gmag = sa.Column(sa.Float)
    rmag = sa.Column(sa.Float)
    zmag = sa.Column(sa.Float)
    w1mag = sa.Column(sa.Float)
    w2mag = sa.Column(sa.Float)
    gmagerr = sa.Column(sa.Float)
    rmagerr = sa.Column(sa.Float)
    zmagerr = sa.Column(sa.Float)
    w1magerr = sa.Column(sa.Float)
    w2magerr = sa.Column(sa.Float)
    z_phot = sa.Column(sa.Float)
    z_phot_err = sa.Column(sa.Float)
    z_spec = sa.Column(sa.Float)

    gaiamatch = sa.Column(sa.Boolean)
    milliquasmatch = sa.Column(sa.Boolean)
    wisematch = sa.Column(sa.Boolean)
    hitsmatch = sa.Column(sa.Boolean)

    @hybrid_property
    def needs_check(self):
        return sa.or_(self.gaiamatch == None,
                      self.milliquasmatch == None,
                      self.wisematch == None,
                      self.hitsmatch == None)

    @hybrid_property
    def lens_cand(self):
        return sa.and_(self.gaiamatch == False,
                       self.milliquasmatch == False,
                       self.wisematch == False,
                       self.hitsmatch == False)

    q3c = Index('dr6object_q3c_ang2ipix_idx', func.q3c_ang2ipix(ra, dec))


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
    errors = sa.Column(models.NumpyArray)
    nfit = sa.Column(sa.Integer)
    data_mask = sa.Column(psql.ARRAY(sa.Boolean))
    source_id = sa.Column(sa.Text, sa.ForeignKey('sources.id', ondelete='SET NULL'))
    source = relationship('Source')


class File(object):

    """Any tracked file (something that resides on disk or on tape) should implement these columns"""

    disk_path = sa.Column(sa.Text)
    hpss_path = sa.Column(sa.Text)


class FITSBase(File):

    """Base class for any FITS image. Any tracked pipeline image product must implement these columns """

    simple = sa.Column(sa.Boolean)
    bitpix = sa.Column(sa.Integer)
    naxis = sa.Column(sa.Integer)
    naxis1 = sa.Column(sa.Integer)
    naxis2 = sa.Column(sa.Integer)
    equinox = sa.Column(sa.Float)
    ctype1 = sa.Column(sa.Text)
    cunit1 = sa.Column(sa.Text)
    crval1 = sa.Column(sa.Float)
    crpix1 = sa.Column(sa.Float)
    cd1_1 = sa.Column(sa.Float)
    cd1_2 = sa.Column(sa.Float)
    ctype2 = sa.Column(sa.Text)
    cunit2 = sa.Column(sa.Text)
    crval2 = sa.Column(sa.Float)
    crpix2 = sa.Column(sa.Float)
    cd2_1 = sa.Column(sa.Float)
    cd2_2 = sa.Column(sa.Float)
    exptime = sa.Column(sa.Float)
    gain = sa.Column(sa.Float)
    saturate = sa.Column(sa.Float)
    filter = sa.Column(sa.Text)
    pixscale = sa.Column(sa.Float)
    magzp = sa.Column(sa.Float)
    seeing = sa.Column(sa.Float)
    medsky = sa.Column(sa.Float)
    lmt_mg = sa.Column(sa.Float)
    lmg_nsigma = sa.Column(sa.Float)

    @declared_attr
    def field(self):
        return sa.Column(sa.Integer)

    @declared_attr
    def ccdid(self):
        return sa.Column(sa.Integer)

    @declared_attr
    def qid(self):
        return sa.Column(sa.Integer)

    @declared_attr
    def filtercode(self):
        return sa.Column(sa.Text)


class SubtractionMixin(FITSBase):

    """Anything Produced by hotpants should implement these columns"""

    bzero = sa.Column(sa.Float)
    bscale = sa.Column(sa.Float)
    region00 = sa.Column(sa.Text)
    convol00 = sa.Column(sa.Text)
    ksum00 = sa.Column(sa.Float)
    sssig00 = sa.Column(sa.Float)
    ssscat00 = sa.Column(sa.Float)
    fsig00 = sa.Column(sa.Float)
    fscat00 = sa.Column(sa.Float)
    x2nrm00 = sa.Column(sa.Float)
    dmean00 = sa.Column(sa.Float)
    dsige00 = sa.Column(sa.Float)
    dsig00 = sa.Column(sa.Float)
    dmeano00 = sa.Column(sa.Float)
    dsigeo00 = sa.Column(sa.Float)
    dsigo00 = sa.Column(sa.Float)
    diffcmd = sa.Column(sa.Text)
    photnorm = sa.Column(sa.Text)
    target = sa.Column(sa.Text)
    template = sa.Column(sa.Text)
    diffim = sa.Column(sa.Text)
    nregion = sa.Column(sa.Integer)
    maskval = sa.Column(sa.Float)
    kerinfo = sa.Column(sa.Boolean)

    @declared_attr
    def reference_id(self):
        return sa.Column('reference_id', sa.Integer, sa.ForeignKey('references.id', ondelete='CASCADE'))

    @declared_attr
    def reference(self):
        return relationship('Reference')


def redundantly_declare_thumbnails(source):
    stack_thumbs = source.stack_thumbnails
    photometry = source.photometry

    if len(source.photometry) == 0:
        return

    highsnr = max(photometry, key=lambda p: p.flux / p.fluxerr)

    seen = []

    for thumb in stack_thumbs:

        if thumb.type not in seen:

            nthumb = models.Thumbnail(type=thumb.type, file_uri=thumb.file_uri,
                                      public_url=thumb.public_url, photometry_id=highsnr.id)
            DBSession().add(nthumb)

            seen.append(thumb.type)

    DBSession().commit()


class SingleEpochSubtraction(SubtractionMixin, models.Base):

    """These correspond to one science image - one reference"""

    image_id = sa.Column(sa.Integer, sa.ForeignKey('image.id', ondelete='CASCADE'))
    image = relationship('Image', back_populates='subtraction')

    photometry = relationship('Photometry', cascade='all')

    def force_photometry(self):

        sources_contained = self.image.sources
        sources_contained_ids = [s.id for s in sources_contained]
        photometered_sources = list(set([phot_point.source for phot_point in self.photometry]))

        # this must be list or setdiff1d will fail
        photometered_source_ids = list(set([s.id for s in photometered_sources]))

        # reject sources where photometry has already been done
        sources_remaining_ids = np.setdiff1d(sources_contained_ids, photometered_source_ids)
        sources_remaining = [s for s in sources_contained if s.id in sources_remaining_ids]

        # we want a model of the PSF on the science image only. we download this here from ipac
        # note the difference image psfs are not correct as they use a convolved science image

        if self.image.instrument is None:
            self.image.instrument_id = 1
            DBSession().add(self)
            DBSession().commit()

        # for all the remaining sources do forced photometry

        new_photometry = []

        for source in sources_remaining:

            # get the best stack detection of the source
            bestpoint = source.best_stack_detection
            flux, fluxerr = phot_sex_auto(self.disk_path, bestpoint)

            phot_point = models.Photometry(subtraction=self, stack_detection=bestpoint,
                                           flux=float(flux), fluxerr=float(fluxerr),
                                           zp=self.magzp, zpsys='ab', lim_mag=self.image.maglimit,
                                           filter=self.filter, source=source, instrument=self.image.instrument,
                                           ra=source.ra, dec=source.dec, mjd=self.image.obsmjd, provenance='gn',
                                           method='sep')
            new_photometry.append(phot_point)

        DBSession().add_all(new_photometry)
        DBSession().commit()


class StackMixin(FITSBase):

    combinet = sa.Column(sa.Text)
    resampt1 = sa.Column(sa.Text)
    centert1 = sa.Column(sa.Text)
    pscalet1 = sa.Column(sa.Text)
    resampt2 = sa.Column(sa.Text)
    centert2 = sa.Column(sa.Text)
    pscalet2 = sa.Column(sa.Text)


class Stack(StackMixin, models.Base):

    subtraction = relationship('MultiEpochSubtraction', back_populates='stack', cascade='all')
    images = relationship('Image', cascade='all', secondary='stack_images')


class Reference(StackMixin, models.Base):
    images = relationship('Image', cascade='all', secondary='reference_images')
    idx = Index('ref_field_idx', 'field', 'ccdid', 'qid', 'filtercode')

ReferenceImage = join_model('reference_images', Reference, Image)
StackImage = join_model('stack_images', Stack, Image)


class MultiEpochSubtraction(StackMixin, SubtractionMixin, models.Base):

    stack_id = sa.Column(sa.Integer, sa.ForeignKey('stacks.id', ondelete='CASCADE'))
    stack = relationship('Stack', back_populates='subtraction', cascade='all')

    detections = relationship('StackDetection', cascade='all')


class StackThumbnail(models.Base):
    type = sa.Column(sa.Enum('new', 'ref', 'sub', 'sdss', 'ps1', "new_gz",
                             'ref_gz', 'sub_gz',
                             name='stackthumbnail_types', validate_strings=True))
    file_uri = sa.Column(sa.String(), nullable=True, index=False, unique=False)
    public_url = sa.Column(sa.String(), nullable=True, index=False, unique=False)
    origin = sa.Column(sa.String, nullable=True)
    stackdetection_id = sa.Column(sa.ForeignKey('stackdetections.id', ondelete='CASCADE'),
                                  nullable=False, index=True)
    stackdetection = relationship('StackDetection', back_populates='thumbnails', cascade='all')
    source = relationship('Source', back_populates='stack_thumbnails', uselist=False,
                          secondary='stackdetections', cascade='all')


models.Source.stack_thumbnails = relationship('StackThumbnail', cascade='all', secondary='stackdetections')
models.Photometry.subtraction_id = sa.Column(sa.Integer, sa.ForeignKey('singleepochsubtractions.id',
                                                                       ondelete='CASCADE'), index=True)
models.Photometry.subtraction = relationship('SingleEpochSubtraction', back_populates='photometry', cascade='all')

models.Photometry.stack_detection_id = sa.Column(sa.Integer,
                                                 sa.ForeignKey('stackdetections.id', ondelete='CASCADE'),
                                                 index=True)
models.Photometry.stack_detection = relationship('StackDetection', back_populates='photometry', cascade='all')


def create_ztf_groups_if_nonexistent():
    groups = [1, 2, 3]
    group_names = ['MSIP/Public', 'Partnership', 'Caltech']
    for g, n in zip(groups, group_names):
        dbe = DBSession().query(Group).get(g)
        if dbe is None:
            dbg = Group(name=f'IPAC GID {g} ({n})')
            dbg.id = g  # match group id to ipac gid
            DBSession().add(dbg)
            iprog = IPACProgram()
            iprog.id = g
            DBSession().add(iprog)
    DBSession().commit()
    for g, n in zip(groups, group_names):
        for i in range(g, 4):
            ipg = DBSession().query(IPACProgramGroup)\
                             .filter(sa.and_(IPACProgramGroup.ipacprogram_id == g, IPACProgramGroup.group_id == i))\
                             .first()
            if ipg is None:
                ipg = IPACProgramGroup(ipacprogram_id=g, group_id=i)
                DBSession.add(ipg)
    DBSession().commit()


    # see if ZTF instrument and telescope exist
    p48 = DBSession().query(models.Telescope).filter(models.Telescope.nickname.like('%p48%')).first()
    if p48 is None:
        p48 = models.Telescope(name='Palomar 48-inch', nickname='p48', lat=33.3581, lon=116.8663,
                               elevation=1870.862, diameter=1.21)
        DBSession().add(p48)
        DBSession().commit()

    ztf = DBSession().query(models.Instrument).get(1)
    if ztf is None:
        ztf = models.Instrument(name='ZTF', type='Camera', band='optical', telescope=p48)
        DBSession().add(ztf)
        DBSession().commit()


def refresh_tables_groups():
    create_tables()
    create_ztf_groups_if_nonexistent()

