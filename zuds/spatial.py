from astropy.coordinates import SkyCoord

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as psql
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.ext.hybrid import hybrid_property

__all__ = ['SpatiallyIndexed', 'HasPoly']


class SpatiallyIndexed(object):
    """A mixin indicating to the database that an object has sky coordinates.
    Classes that mix this class get a q3c spatial index on ra and dec.

    Columns:
        ra: the icrs right ascension of the object in degrees
        dec: the icrs declination of the object in degrees

    Indexes:
        q3c index on ra, dec

    Properties: skycoord: astropy.coordinates.SkyCoord representation of the
    object's coordinate
    """

    # database-mapped
    ra = sa.Column(psql.DOUBLE_PRECISION)
    dec = sa.Column(psql.DOUBLE_PRECISION)

    @property
    def skycoord(self):
        return SkyCoord(self.ra, self.dec, unit='deg')

    @declared_attr
    def __table_args__(cls):
        tn = cls.__tablename__
        return sa.Index(f'{tn}_q3c_ang2ipix_idx', sa.func.q3c_ang2ipix(
            cls.ra, cls.dec)),


class HasPoly(object):
    """Mixin indicating that an object represents an entity with four corners
    on the celestial sphere, connected by great circles.

    The four corners, ra{1..4}, dec{1..4} are database-backed metadata,
    and are thus queryable.

    Provides a hybrid property `poly` (in-memory and in-db), which can be
    used to query against the polygon in the database or to access the
    polygon in memory.
    """

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

