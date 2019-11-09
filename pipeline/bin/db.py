import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as psql
from sqlalchemy.dialects.postgresql import array
from sqlalchemy import Index
from sqlalchemy import func

from pathlib import Path
import os
from skyportal import models
from skyportal.models import (join_model, DBSession, ACL,
                              Role, User, Token, Group, init_db as idb)

from skyportal.model_util import create_tables, drop_tables
from sqlalchemy.ext.hybrid import hybrid_property


from secrets import get_secret



def init_db():
    hpss_dbhost = get_secret('hpss_dbhost')
    hpss_dbport = get_secret('hpss_dbport')
    hpss_dbusername = get_secret('hpss_dbusername')
    hpss_dbname = get_secret('olddb')
    hpss_dbpassword = get_secret('hpss_dbpassword')
    return idb(hpss_dbusername, hpss_dbname, hpss_dbpassword, hpss_dbhost, hpss_dbport)


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
    ipac_gid = sa.Column(sa.Integer)
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

