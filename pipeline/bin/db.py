import sqlalchemy as sa
from sqlalchemy.dialects import psql
from pathlib import Path
import os
from skyportal import models
from skyportal.models import init_db, DBSession

__all__ = ['Image', 'init_db', 'DBSession']


class Image(models.Base):

    __tablename__ = 'image'
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
