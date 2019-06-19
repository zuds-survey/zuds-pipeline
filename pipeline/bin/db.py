import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as psql


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


class IPACProgram(models.Base):
    groups = relationship('Group', secondary='ipacprogram_groups',  back_populates='ipacprograms', cascade='all')
    images = relationship('Image', back_populates='ipac_program', cascade='all')

IPACProgramGroup = join_model('ipacprogram_groups', IPACProgram, Group)
Group.ipacprograms = relationship('IPACProgram', secondary='ipacprogram_groups', back_populates='groups', cascade='all')



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
    photometry = relationship('Photometry', back_populates='image', secondary='image_photometry')

    @hybrid_property
    def poly(self):
        return array([self.ra1, self.dec1, self.ra2, self.dec2,
                      self.ra3, self.dec3, self.ra4, self.dec4])

    @property
    def sources(self):
        return DBSession().query(models.Source)\
                          .filter(func.q3c_poly_query(models.Source.ra, models.Source.dec, self.poly))\
                          .all()

    def contains_source(self, source):
        return DBSession().execute(sa.select([func.q3c_poly_query(source.ra, source.dec, self.poly)])).first()[0]

    def provided_photometry(self, photometry):
        return photometry in self.photometry


ImagePhotometry = join_model('image_photometry', Image, models.Photometry)
models.Source.images = relationship('Image', secondary='join(ImagePhotometry, Photometry).join(sources)')

# keep track of the images that the photometry came from
models.Photometry.image = relationship('Image', back_populates='photometry', secondary='image_photometry')


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


def refresh_tables_groups():
    create_tables()
    create_ztf_groups_if_nonexistent()

