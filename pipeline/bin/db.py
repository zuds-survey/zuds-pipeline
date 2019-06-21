import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as psql
import numpy as np

from sqlalchemy.orm import relationship, column_property
from sqlalchemy.dialects.postgresql import array

from sqlalchemy import Index
from sqlalchemy import func

from pathlib import Path
import os
from skyportal import models
from skyportal.models import (init_db, join_model, DBSession, ACL,
                              Role, User, Token, Group)

from skyportal.model_util import create_tables, drop_tables
from sqlalchemy.ext.hybrid import hybrid_property

from astropy.io import fits
from libztf.yao import yao_photometry_single

from baselayer.app.env import load_env
from datetime import datetime


class IPACProgram(models.Base):
    groups = relationship('Group', secondary='ipacprogram_groups',  back_populates='ipacprograms', cascade='all')
    images = relationship('Image', back_populates='ipac_program', cascade='all')

IPACProgramGroup = join_model('ipacprogram_groups', IPACProgram, Group)
Group.ipacprograms = relationship('IPACProgram', secondary='ipacprogram_groups', back_populates='groups', cascade='all')


class Image(models.Base):

    __tablename__ = 'image'

    created_at = sa.Column(sa.DateTime(), nullable=True)
    path = sa.Column(sa.Text)
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


    fcqfo = Index("image_field_ccdid_qid_filtercode_obsjd_idx",  field, ccdid, qid, filtercode, obsjd)
    hmi = Index("image_hpss_mask_path_idx", hpss_mask_path)
    hpi = Index("image_hpss_psf_path_idx", hpss_psf_path)
    hshmi = Index("image_hpss_sci_path_hpss_mask_path_idx" ,hpss_sci_path, hpss_mask_path)
    hsci = Index("image_hpss_sci_path_idx" ,hpss_sci_path)
    hsubi = Index("image_hpss_sub_path_idx", hpss_sub_path)
    obsjdi = Index("image_obsjd_idx", obsjd)
    pathi = Index("image_path_idx", path)

    groups = relationship('Group', back_populates='images', secondary='join(IPACProgram, ipacprogram_groups).join(groups)')
    ipac_program = relationship('IPACProgram', back_populates='images', cascade='all')
    photometry = relationship('Photometry', cascade='all')

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

    def contains_source(self, source):
        return DBSession().execute(sa.select([func.q3c_poly_query(source.ra, source.dec, self.poly)])).first()[0]

    def provided_photometry(self, photometry):
        return photometry in self.photometry

    def force_photometry(self):

        sources_contained = self.sources
        sources_contained_ids = [s.id for s in sources_contained]
        photometered_sources = set([phot_point.source for phot_point in self.photometry])
        photometered_source_ids = set([s.id for s in photometered_sources])

        # reject sources where photometry has already been done
        sources_remaining_ids = np.setdiff1d(sources_contained_ids, photometered_source_ids)
        sources_remaining = [s for s in sources_contained if s.id in sources_remaining_ids]

        # get the paths to relevant files on disk
        psf_path = self.disk_psf_path
        sub_path = self.disk_sub_path

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
                return
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
            pobj = yao_photometry_single(sub_path, psf_path, source.ra, source.dec)
            phot_point = models.Photometry(image=self, flux=float(pobj.Fpsf), fluxerr=float(pobj.eFpsf),
                                           zp=self.zp, zpsys=self.zpsys, lim_mag=self.maglimit,
                                           filter=self.filter, source=source, instrument=self.instrument,
                                           ra=source.ra, dec=source.dec, mjd=self.obsmjd)
            new_photometry.append(phot_point)

        DBSession().add_all(new_photometry)
        DBSession().commit()


# keep track of the images that the photometry came from
models.Photometry.image_id = sa.Column(sa.Integer, sa.ForeignKey('image.id', ondelete='CASCADE'))
models.Photometry.image = relationship('Image', back_populates='photometry')

models.Source.images = relationship('Image', secondary='join(Photometry, Source)')


Group.images = relationship('Image', back_populates='groups',
                            secondary='join(IPACProgram, ipacprogram_groups).join(groups)')


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

