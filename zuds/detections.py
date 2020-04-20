import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

from .spatial import SpatiallyIndexed
from .core import Base, DBSession


from .constants import BRAAI_MODEL

__all__ = ['RealBogus', 'Detection']

class RealBogus(Base):

    __tablename__ = 'realbogus'

    rb_score = sa.Column(sa.Float)
    rb_version = sa.Column(sa.Text)
    detection_id = sa.Column(sa.Integer, sa.ForeignKey('detections.id',
                                                       ondelete='CASCADE'),
                             index=True)
    detection = relationship('Detection', back_populates='rb', cascade='all')


class Detection(Base, SpatiallyIndexed):

    __tablename__ = 'detections'

    x_image = sa.Column(sa.Float)
    y_image = sa.Column(sa.Float)

    elongation = sa.Column(sa.Float)
    a_image = sa.Column(sa.Float)
    b_image = sa.Column(sa.Float)
    fwhm_image = sa.Column(sa.Float)

    flags = sa.Column(sa.Integer)
    imaflags_iso = sa.Column(sa.Integer)
    goodcut = sa.Column(sa.Boolean)
    rb = relationship('RealBogus', cascade='all')

    triggers_alert = sa.Column(sa.Boolean)
    triggers_phot = sa.Column(sa.Boolean)
    alert_ready = sa.Column(sa.Boolean)


    alert = relationship('Alert', cascade='all', uselist=False)

    thumbnails = relationship('Thumbnail', cascade='all')

    image_id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id',
                                                   ondelete='CASCADE'),
                         index=True)
    image = relationship('CalibratableImage', back_populates='detections',
                         cascade='all')

    # thumbnails = relationship('Thumbnail', cascade='all')

    source_id = sa.Column(sa.Text,
                          sa.ForeignKey('sources.id', ondelete='CASCADE'),
                          index=True)
    source = relationship('Source', cascade='all')

    flux = sa.Column(sa.Float)
    fluxerr = sa.Column(sa.Float)

    @hybrid_property
    def snr(self):
        return self.flux / self.fluxerr

    @classmethod
    def from_catalog(cls, cat, filter=True):
        result = []

        from .filterobjects import filter_sexcat

        # TODO: determine if prev dets that are updated should also be returned

        if filter:
            filter_sexcat(cat)

        for row in cat.data:

            if filter and row['GOODCUT'] != 1:
                continue

            # ra and dec are inherited from SpatiallyIndexed
            detection = cls(
                ra=float(row['X_WORLD']), dec=float(row['Y_WORLD']),
                image=cat.image, flux=float(row['FLUX_APER']),
                fluxerr=float(row['FLUXERR_APER']),
                elongation=float(row['ELONGATION']),
                flags=int(row['FLAGS']), imaflags_iso=int(row['IMAFLAGS_ISO']),
                a_image=float(row['A_IMAGE']), b_image=float(row['B_IMAGE']),
                fwhm_image=float(row['FWHM_IMAGE']),
                x_image=float(row['X_IMAGE']), y_image=float(row['Y_IMAGE']),
            )

            rb = RealBogus(rb_score=float(row['rb']), rb_version=BRAAI_MODEL,
                           detection=detection)

            DBSession().add(rb)

            if filter:
                detection.goodcut = True

            result.append(detection)

        return result

