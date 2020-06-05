import os
import sqlalchemy as sa
import numpy as np
from sqlalchemy.exc import UnboundExecutionError
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects import postgresql as psql

from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import sessionmaker, scoped_session, relationship

from .file import File
from .utils import fid_map
from .json_util import to_json


__all__ = ['DBSession', 'Base', 'init_db', 'join_model', 'ZTFFile',
           'without_database']


DBSession = scoped_session(sessionmaker())


# The db has to be initialized later; this is done by the app itself
# See `app_server.py`
def init_db(user, database, password=None, host=None, port=None):
    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(user, password or '', host or '', port or '', database)

    conn = sa.create_engine(url, client_encoding='utf8')

    DBSession.configure(bind=conn)
    Base.metadata.bind = conn

    return conn


class BaseMixin(object):
    query = DBSession.query_property()
    id = sa.Column(sa.Integer, primary_key=True)
    created_at = sa.Column(sa.DateTime, nullable=False, server_default=sa.func.now())
    modified = sa.Column(sa.DateTime, nullable=False, server_default=sa.func.now(),
                         server_onupdate=sa.func.now())

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower() + 's'

    __mapper_args__ = {'confirm_deleted_rows': False}

    def __repr__(self):
        """String representation of sqlalchemy objects."""
        if sa.inspection.inspect(self).expired:
            DBSession().refresh(self)
        inst = sa.inspect(self)
        attr_list = [f"{g.key}={getattr(self, g.key)}"
                     for g in inst.mapper.column_attrs]
        return f"<{type(self).__name__}({', '.join(attr_list)})>"

    def __str__(self):
        if sa.inspection.inspect(self).expired:
            DBSession().refresh(self)
        inst = sa.inspect(self)
        attr_list = {g.key: getattr(self, g.key) for g in inst.mapper.column_attrs}
        return to_json(attr_list)

    def to_dict(self):
        if sa.inspection.inspect(self).expired:
            DBSession().refresh(self)
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def create_or_get(cls, id):
        obj = cls.query.get(id)
        if obj is not None:
            return obj
        else:
            return cls(id=id)


Base = declarative_base(cls=BaseMixin)


def join_model(join_table, model_1, model_2, column_1=None, column_2=None,
               fk_1='id', fk_2='id', base=Base):
    """Helper function to create a join table for a many-to-many relationship.
    Parameters
    ----------
    join_table : str
        Name of the new table to be created.
    model_1 : str
        First model in the relationship.
    model_2 : str
        Second model in the relationship.
    column_1 : str, optional
        Name of the join table column corresponding to `model_1`. If `None`,
        then {`table1`[:-1]_id} will be used (e.g., `user_id` for `users`).
    column_2 : str, optional
        Name of the join table column corresponding to `model_2`. If `None`,
        then {`table2`[:-1]_id} will be used (e.g., `user_id` for `users`).
    fk_1 : str, optional
        Name of the column from `model_1` that the foreign key should refer to.
    fk_2 : str, optional
        Name of the column from `model_2` that the foreign key should refer to.
    base : sqlalchemy.ext.declarative.api.DeclarativeMeta
        SQLAlchemy model base to subclass.
    Returns
    -------
    sqlalchemy.ext.declarative.api.DeclarativeMeta
        SQLAlchemy association model class
    """
    table_1 = model_1.__tablename__
    table_2 = model_2.__tablename__
    if column_1 is None:
        column_1 = f'{table_1[:-1]}_id'
    if column_2 is None:
        column_2 = f'{table_2[:-1]}_id'
    reverse_ind_name = f'{join_table}_reverse_ind'

    model_attrs = {
        '__tablename__': join_table,
        'id': None,
        column_1: sa.Column(column_1, sa.ForeignKey(f'{table_1}.{fk_1}',
                                                    ondelete='CASCADE'),
                            primary_key=True),
        column_2: sa.Column(column_2, sa.ForeignKey(f'{table_2}.{fk_2}',
                                                    ondelete='CASCADE'),
                            primary_key=True)
    }

    model_attrs.update({
        model_1.__name__.lower(): relationship(model_1, cascade='all',
                                               foreign_keys=[
                                                   model_attrs[column_1]
                                               ]),
        model_2.__name__.lower(): relationship(model_2, cascade='all',
                                               foreign_keys=[
                                                   model_attrs[column_2]
                                               ]),
        reverse_ind_name: sa.Index(reverse_ind_name,
                                   model_attrs[column_2],
                                   model_attrs[column_1])

    })
    model = type(model_1.__name__ + model_2.__name__, (base,), model_attrs)

    return model


class NumpyArray(sa.types.TypeDecorator):
    impl = psql.ARRAY(sa.Float)

    def process_result_value(self, value, dialect):
        return np.array(value)


class Source(Base):
    id = sa.Column(sa.String, primary_key=True)
    # TODO should this column type be decimal? fixed-precison numeric
    ra = sa.Column(sa.Float)
    dec = sa.Column(sa.Float)

    ra_dis = sa.Column(sa.Float)
    dec_dis = sa.Column(sa.Float)

    ra_err = sa.Column(sa.Float, nullable=True)
    dec_err = sa.Column(sa.Float, nullable=True)

    offset = sa.Column(sa.Float, default=0.0)
    redshift = sa.Column(sa.Float, nullable=True)

    altdata = sa.Column(psql.JSONB, nullable=True)

    last_detected = sa.Column(sa.DateTime, nullable=True)
    dist_nearest_source = sa.Column(sa.Float, nullable=True)
    mag_nearest_source = sa.Column(sa.Float, nullable=True)
    e_mag_nearest_source = sa.Column(sa.Float, nullable=True)

    transient = sa.Column(sa.Boolean, default=False)
    varstar = sa.Column(sa.Boolean, default=False)
    is_roid = sa.Column(sa.Boolean, default=False)

    score = sa.Column(sa.Float, nullable=True)

    ## pan-starrs
    sgmag1 = sa.Column(sa.Float, nullable=True)
    srmag1 = sa.Column(sa.Float, nullable=True)
    simag1 = sa.Column(sa.Float, nullable=True)
    objectidps1 = sa.Column(sa.BigInteger, nullable=True)
    sgscore1 = sa.Column(sa.Float, nullable=True)
    distpsnr1 = sa.Column(sa.Float, nullable=True)

    origin = sa.Column(sa.String, nullable=True)
    simbad_class = sa.Column(sa.Unicode, nullable=True, )
    simbad_info = sa.Column(psql.JSONB, nullable=True)
    gaia_info = sa.Column(psql.JSONB, nullable=True)
    tns_info = sa.Column(psql.JSONB, nullable=True)
    tns_name = sa.Column(sa.Unicode, nullable=True)

    photometry = relationship('Photometry', back_populates='source',
                              cascade='save-update, merge, refresh-expire, expunge',
                              single_parent=True,
                              passive_deletes=True,
                              order_by="Photometry.observed_at")

    detect_photometry_count = sa.Column(sa.Integer, nullable=True)
    thumbnails = relationship('Thumbnail', back_populates='source',
                              secondary='photometry',
                              cascade='save-update, merge, refresh-expire, expunge')

    def add_linked_thumbnails(self):
        sdss_thumb = Thumbnail(photometry=self.photometry[0],
                               public_url=self.sdss_url,
                               type='sdss')
        dr8_thumb = Thumbnail(photometry=self.photometry[0],
                              public_url=self.desi_dr8_url,
                              type='dr8')
        DBSession().add_all([sdss_thumb, dr8_thumb])
        DBSession().commit()

    @property
    def sdss_url(self):
        """Construct URL for public Sloan Digital Sky Survey (SDSS) cutout."""
        return (f"http://skyservice.pha.jhu.edu/DR9/ImgCutout/getjpeg.aspx"
                f"?ra={self.ra}&dec={self.dec}&scale=0.3&width=200&height=200"
                f"&opt=G&query=&Grid=on")

    @property
    def desi_dr8_url(self):
        """Construct URL for public DESI DR8 cutout."""
        return (f"http://legacysurvey.org/viewer/jpeg-cutout?ra={self.ra}"
                f"&dec={self.dec}&size=200&layer=dr8&pixscale=0.262&bands=grz")


class Telescope(Base):
    name = sa.Column(sa.String, nullable=False)
    nickname = sa.Column(sa.String, nullable=False)
    lat = sa.Column(sa.Float, nullable=False)
    lon = sa.Column(sa.Float, nullable=False)
    elevation = sa.Column(sa.Float, nullable=False)
    diameter = sa.Column(sa.Float, nullable=False)
    instruments = relationship('Instrument', back_populates='telescope',
                               cascade='save-update, merge, refresh-expire, expunge',
                               passive_deletes=True)


class Instrument(Base):
    name = sa.Column(sa.String, nullable=False)
    type = sa.Column(sa.String, nullable=False)
    band = sa.Column(sa.String, nullable=False)

    telescope_id = sa.Column(sa.ForeignKey('telescopes.id',
                                           ondelete='CASCADE'),
                             nullable=False, index=True)
    telescope = relationship('Telescope', back_populates='instruments')
    photometry = relationship('Photometry', back_populates='instrument')


class Thumbnail(Base):
    # TODO delete file after deleting row
    type = sa.Column(sa.Enum('new', 'ref', 'sub', 'sdss', 'dr8', "new_gz",
                             'ref_gz', 'sub_gz',
                             name='thumbnail_types', validate_strings=True))
    file_uri = sa.Column(sa.String(), nullable=True, index=False, unique=False)
    public_url = sa.Column(sa.String(), nullable=True, index=False, unique=False)
    origin = sa.Column(sa.String, nullable=True)
    photometry_id = sa.Column(sa.ForeignKey('photometry.id', ondelete='CASCADE'),
                              nullable=False, index=True)
    photometry = relationship('Photometry', back_populates='thumbnails')
    source = relationship('Source', back_populates='thumbnails', uselist=False,
                          secondary='photometry')


def without_database(retval):
    ## Decorator that tells the wrapped function to return retval if
    ## there is no active database connection
    def wrapped(func):
        def interior(*args, **kwargs):
            try:
                bind = DBSession().get_bind()
            except UnboundExecutionError:
                return retval
            else:
                return func(*args, **kwargs)
        return interior
    return wrapped


class ZTFFile(Base, File):
    """A database-mapped, disk-mappable memory-representation of a file that
    is associated with a ZTF sky partition. This class is abstract and not
    designed to be instantiated, but it is also not a mixin. Think of it as a
    base class for the polymorphic hierarchy of products in SQLalchemy.

    To create an disk-mappable representation of a fits file that stores data in
    memory and is not mapped to rows in the database, instantiate FITSFile
    directly.
    """

    # this is the discriminator that is used to keep track of different types
    #  of fits files produced by the pipeline for the rest of the hierarchy
    type = sa.Column(sa.Text)

    # all pipeline fits products must implement these four key pieces of
    # metadata. These are all assumed to be not None in valid instances of
    # ZTFFile.

    field = sa.Column(sa.Integer)
    qid = sa.Column(sa.Integer)
    fid = sa.Column(sa.Integer)
    ccdid = sa.Column(sa.Integer)

    copies = relationship('ZTFFileCopy', cascade='all')

    # An index on the four indentifying
    idx = sa.Index('fitsproduct_field_ccdid_qid_fid', field, ccdid, qid, fid)

    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'fitsproduct'

    }

    def find_in_dir(self, directory):
        target = os.path.join(directory, self.basename)
        if os.path.exists(target):
            self.map_to_local_file(target)
        else:
            raise FileNotFoundError(
                f'Cannot map "{self.basename}" to "{target}", '
                f'file does not exist.'
            )

    def find_in_dir_of(self, ztffile):
        dirname = os.path.dirname(ztffile.local_path)
        self.find_in_dir(dirname)

    @classmethod
    @without_database(None)
    def get_by_basename(cls, basename):

        obj = DBSession().query(cls).filter(
            cls.basename == basename
        ).first()

        if obj is not None:
            obj.clear()  # get a fresh copy

        if hasattr(obj, 'mask_image'):
            if obj.mask_image is not None:
                obj.mask_image.clear()

        if hasattr(obj, 'catalog'):
            if obj.catalog is not None:
                obj.catalog.clear()

        return obj

    @property
    def relname(self):
        return f'{self.field:06d}/' \
               f'c{self.ccdid:02d}/' \
               f'q{self.qid}/' \
               f'{fid_map[self.fid]}/' \
               f'{self.basename}'

    @hybrid_property
    def relname_hybrid(self):
        return sa.func.format(
            '%s/c%s/q%s/%s/%s',
            sa.func.lpad(sa.func.cast(self.field, sa.Text), 6, '0'),
            sa.func.lpad(sa.func.cast(self.ccdid, sa.Text), 2, '0'),
            self.qid,
            sa.case([
                (self.fid == 1, 'zg'),
                (self.fid == 2, 'zr'),
                (self.fid == 3, 'zi')
            ]),
            self.basename
        )

