import os
import sqlalchemy as sa
import pandas as pd
from sqlalchemy.orm import relationship

from .core import ZTFFile
from .constants import GROUP_PROPERTIES, BAD_SUM
from .fitsfile import FITSFile

__all__ = ['PipelineFITSCatalog', 'PipelineRegionFile']


class PipelineRegionFile(ZTFFile):

    id = sa.Column(sa.Integer, sa.ForeignKey('ztffiles.id',
                                             ondelete='CASCADE'),
                   primary_key=True)

    __mapper_args__ = {
        'polymorphic_identity': 'regionfile',
        'inherit_condition': id == ZTFFile.id
    }

    catalog_id = sa.Column(sa.Integer, sa.ForeignKey(
        'pipelinefitscatalogs.id', ondelete='CASCADE'
    ), index=True)
    catalog = relationship('PipelineFITSCatalog', cascade='all',
                           foreign_keys=[catalog_id],
                           back_populates='regionfile')

    @classmethod
    def from_catalog(cls, catalog):
        basename = catalog.basename.replace('.cat', '.reg')
        reg = cls.get_by_basename(basename)
        if reg is None:
            reg = cls()
            reg.basename = basename

        catdir = os.path.dirname(catalog.local_path)
        reg.map_to_local_file(os.path.join(catdir, reg.basename))

        reg.field = catalog.field
        reg.ccdid = catalog.ccdid
        reg.qid = catalog.qid
        reg.fid = catalog.fid

        filtered = 'GOODCUT' in catalog.data.dtype.names

        reg.catalog = catalog
        with open(reg.local_path, 'w') as f:
            f.write('global color=green dashlist=8 3 width=1 font="helvetica '
                    '10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 '
                    'delete=1 include=1 source=1\n')
            f.write('icrs\n')
            rad = 13 * catalog.image.pixel_scale.to(
                'arcsec'
            ).value.mean() * 0.00027777
            for row in catalog.data:
                if not filtered:
                    color = 'blue'
                else:
                    color = 'green' if row['GOODCUT'] else 'red'

                f.write(f'circle({row["X_WORLD"]},'
                        f'{row["Y_WORLD"]},{rad}) # width=2 '
                        f'color={color}\n')

        return reg


class PipelineFITSCatalog(ZTFFile, FITSFile):
    """Python object that maps a catalog stored on a fits file on disk."""

    id = sa.Column(sa.Integer, sa.ForeignKey('ztffiles.id',
                                             ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {
        'polymorphic_identity': 'catalog',
        'inherit_condition': id == ZTFFile.id
    }

    image_id = sa.Column(sa.Integer,
                         sa.ForeignKey('calibratableimages.id',
                                       ondelete='CASCADE'),
                         index=True)
    image = relationship('CalibratableImage', cascade='all',
                         foreign_keys=[image_id])

    regionfile = relationship('PipelineRegionFile', cascade='all',
                              uselist=False,
                              primaryjoin=PipelineRegionFile.catalog_id == id)

    # since this object maps a fits binary table, the data lives in the first
    #  extension, not in the primary hdu
    _DATA_HDU = 2
    _HEADER_HDU = 2

    @classmethod
    def from_image(cls, image, tmpdir='/tmp', kill_flagged=True):

        from .image import CalibratableImage

        if not isinstance(image, CalibratableImage):
            raise ValueError('Image is not an instance of '
                             'CalibratableImage.')

        image._call_source_extractor(tmpdir=tmpdir)
        cat = image.catalog

        for prop in GROUP_PROPERTIES:
            setattr(cat, prop, getattr(image, prop))

        df = pd.DataFrame(cat.data)
        rec = df.to_records(index=False)
        cat.data = rec
        cat.basename = image.basename.replace('.fits', '.cat')
        cat.image_id = image.id
        cat.image = image
        image.catalog = cat

        if kill_flagged:
            image.catalog.kill_flagged()

        return cat

    def kill_flagged(self):
        # overwrite the catalog, killing any detections with bad IMAFLAGS_ISO
        self.load()
        oinds = []
        for i, row in enumerate(self.data):
            if row['IMAFLAGS_ISO'] & BAD_SUM == 0 and row['FLAGS_WEIGHT'] == 0:
                oinds.append(i)
        out = self.data[oinds]
        self.data = out
        self.save()
        self.load()

