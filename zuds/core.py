import os
from skyportal.models import DBSession
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.orm import relationship
from sqlalchemy.exc import UnboundExecutionError
from sqlalchemy.ext.hybrid import hybrid_property

from skyportal import models
from skyportal.model_util import create_tables, drop_tables

from baselayer.app.json_util import to_json

from .file import File
from .secrets import get_secret
from .utils import fid_map

__all__ = ['DBSession', 'create_tables', 'drop_tables',
           'Base', 'init_db', 'join_model', 'ZTFFile']

Base = models.Base


def model_representation(o):
    """String representation of sqlalchemy objects."""
    if sa.inspection.inspect(o).expired:
        DBSession().refresh(o)
    inst = sa.inspect(o)
    attr_list = [f"{g.key}={getattr(o, g.key)}"
                 for g in inst.mapper.column_attrs]
    return f"<{type(o).__name__}({', '.join(attr_list)})>"


def model_str(o):
    if sa.inspection.inspect(o).expired:
        DBSession().refresh(o)
    inst = sa.inspect(o)
    attr_list = {g.key: getattr(o, g.key) for g in inst.mapper.column_attrs}
    return to_json(attr_list)


Base.__repr__ = model_representation
Base.__str__ = model_str
Base.modified = sa.Column(
    sa.DateTime(timezone=False),
    default=sa.func.now(),
    onupdate=sa.func.now()
)


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
    def get_by_basename(cls, basename):

        try:
            obj = DBSession().query(cls).filter(
                cls.basename == basename
            ).first()
        except UnboundExecutionError:
            # Running without a database
            obj = None

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


def init_db(timeout=None):

    username = get_secret('db_username')
    password = get_secret('db_password')
    port = get_secret('db_port')
    host = get_secret('db_host')
    dbname = get_secret('db_name')

    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(username, password or '', host or '', port or '', dbname)

    kwargs = {}
    if timeout is not None:
        kwargs['connect_args'] = {"options": f"-c statement_timeout={timeout}"}

    conn = sa.create_engine(url, client_encoding='utf8', **kwargs)

    DBSession.configure(bind=conn)
    Base.metadata.bind = conn

init_db()

# Automatically update the `modified` attribute of Base
# when objects are updated.
@event.listens_for(DBSession(), 'before_flush')
def bump_modified(session, flush_context, instances):
    for object in session.dirty:
        if isinstance(object, Base) and session.is_modified(object):
            object.modified = sa.func.now()

