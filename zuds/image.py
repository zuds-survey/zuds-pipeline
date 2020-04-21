import numpy as np
import os
import subprocess
from pathlib import Path

from astropy.visualization.interval import ZScaleInterval
from matplotlib import colors

import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects import postgresql as psql
from sqlalchemy.ext.declarative import declared_attr

from .core import ZTFFile, DBSession
from . import sextractor
from .fitsfile import HasWCS
from .plotting import discrete_cmap
from .constants import BIG_RMS
from .secrets import get_secret
from .catalog import PipelineFITSCatalog
from .photometry import aperture_photometry, ForcedPhotometry
from .constants import APER_KEY
from .utils import fid_map

__all__ = ['FITSImage', 'CalibratableImageBase', 'CalibratableImage',
           'CalibratedImage', 'ScienceImage']


class FITSImage(HasWCS):
    """A `FITSFile` with a data member representing an image. Same as
    FITSFile, but provides the method show() to render the image in
    matplotlib. Also defines some properties that help to optimally render
    the image (cmap, cmap_limits)"""

    def show(self, axis=None, align_to=None, figsize=(5, 5)):
        if axis is None:
            import matplotlib.pyplot as plt
            fig, axis = plt.subplots(figsize=figsize)

        if align_to is not None:
            image = self.aligned_to(align_to)
        else:
            image = self

        vmin, vmax = image.cmap_limits()

        axis.imshow(image.data,
                    vmin=vmin,
                    vmax=vmax,
                    norm=image.cmap_norm(),
                    cmap=image.cmap(),
                    interpolation='none')

    @property
    def datatype(self):
        dtype = self.data.dtype.name
        if 'float' in dtype:
            return 'float'
        else:
            return 'int'

    def cmap_limits(self):
        if self.datatype == 'float':
            interval = ZScaleInterval()
            return interval.get_limits(self.data)
        else:  # integer
            return (None, None)

    def cmap(self):
        if self.datatype == 'float':
            return 'gray'
        else:
            ncolors = len(np.unique(self.data))
            return discrete_cmap(ncolors)

    def cmap_norm(self):
        if self.datatype == 'float':
            return None
        else:
            boundaries = np.unique(self.data)
            ncolors = len(boundaries)
            return colors.BoundaryNorm(boundaries, ncolors)


class CalibratableImageBase(FITSImage):
    __diskmapped_cached_properties__ = ['_path', '_data', '_weightimg',
                                        '_bkgimg', '_filter_kernel', '_rmsimg',
                                        '_threshimg', '_segmimg',
                                        '_sourcelist', '_bkgsubimg']


    def cmap_limits(self):
        interval = ZScaleInterval()
        return interval.get_limits(self.data)

    def _call_source_extractor(self, checkimage_type=None, tmpdir='/tmp',
                               use_weightmap=True):

        rs = sextractor.run_sextractor
        success = False
        for _ in range(3):
            try:
                results = rs(
                    self, checkimage_type=checkimage_type, tmpdir=tmpdir,
                    use_weightmap=use_weightmap
                )
            except subprocess.CalledProcessError as e:
                print(f'Caught CalledProcessError {e}, retrying... {_+1} / 3')
                continue
            else:
                success = True
                break

        if not success:
            raise ValueError(f'Unable to run SExtractor on {self}...')

        for result in results:
            if result.basename.endswith('.cat'):
                self.catalog = result
            elif result.basename.endswith('.rms.fits'):
                self._rmsimg = result
            elif result.basename.endswith('.bkg.fits'):
                self._bkgimg = result
            elif result.basename.endswith('.bkgsub.fits'):
                self._bkgsubimg = result
            elif result.basename.endswith('.segm.fits'):
                self._segmimg = result

    @property
    def weight_image(self):
        """Image representing the inverse variance map of this calibratable
        image."""

        try:
            return self._weightimg
        except AttributeError:
            # need to calculate the weight map.

            ind = self.mask_image.boolean.data
            wgt = np.empty_like(ind, dtype='<f4')
            wgt[~ind] = 1 / self.rms_image.data[~ind] ** 2
            wgt[ind] = 0.

            try:
                saturval = self.header['SATURATE']
            except KeyError:
                pass
            else:
                saturind = self.data >= 0.9 * saturval
                wgt[saturind] = 0.

            self._weightimg = FITSImage()
            self._weightimg.basename = self.basename.replace('.fits',
                                                             '.weight.fits')
            self._weightimg.data = wgt
            self._weightimg.header = self.header
            self._weightimg.header_comments = self.header_comments
            if self.ismapped:
                dirname = os.path.dirname(self.local_path)
                bn = self._weightimg.basename
                join = os.path.join(dirname, bn)
                self._weightimg.map_to_local_file(join)
                self._weightimg.save()  #  to guarantee mapped file exists
        return self._weightimg

    @property
    def rms_image(self):
        try:
            return self._rmsimg
        except AttributeError:
            if hasattr(self, '_weightimg'):
                ind = self.mask_image.boolean.data
                rms = np.empty_like(ind, dtype='<f4')
                rms[~ind] = 1 / np.sqrt(self.weight_image.data[~ind])
                rms[ind] = BIG_RMS

                try:
                    saturval = self.header['SATURATE']
                except KeyError:
                    pass
                else:
                    saturind = self.data >= 0.9 * saturval
                    rms[saturind] = BIG_RMS

                _rmsimg = FITSImage()
                _rmsimg.basename = self.basename.replace('.fits', '.rms.fits')
                _rmsimg.data = rms
                _rmsimg.header = self.header
                _rmsimg.header_comments = self.header_comments
                if self.ismapped:
                    dirname = os.path.dirname(self.local_path)
                    bn = _rmsimg.basename
                    join = os.path.join(dirname, bn)
                    _rmsimg.map_to_local_file(join)
                    _rmsimg.save()
                self._rmsimg = _rmsimg
                return self._rmsimg
            else:
                self._call_source_extractor(checkimage_type=['rms'],
                                            use_weightmap=False)
        return self._rmsimg

    @property
    def background_image(self):
        try:
            return self._bkgimg
        except AttributeError:
            self._call_source_extractor(checkimage_type=['bkg'])
        return self._bkgimg

    @property
    def background_subtracted_image(self):
        try:
            return self._bkgsubimg
        except AttributeError:
            self._call_source_extractor(checkimage_type=['bkgsub'])
        return self._bkgsubimg

    @property
    def segm_image(self):
        try:
            return self._segmimg
        except AttributeError:
            # segm is set here
            self._call_source_extractor(checkimage_type=['segm'])
        return self._segmimg


    @classmethod
    def from_file(cls, fname, use_existing_record=True, load_others=True):
        obj = super().from_file(
            fname, use_existing_record=use_existing_record,
        )
        dir = Path(fname).parent

        if load_others:
            weightpath = dir / obj.basename.replace('.fits', '.weight.fits')
            rmspath = dir / obj.basename.replace('.fits', '.rms.fits')
            bkgpath = dir / obj.basename.replace('.fits', '.bkg.fits')
            threshpath = dir / obj.basename.replace('.fits', '.thresh.fits')
            bkgsubpath = dir / obj.basename.replace('.fits', '.bkgsub.fits')
            segmpath = dir / obj.basename.replace('.fits', '.segm.fits')

            paths = [weightpath, rmspath, bkgpath, threshpath,
                     bkgsubpath, segmpath]

            types = ['_weightimg', '_rmsimg', '_bkgimg', '_threshimg',
                     '_bkgsubimg', '_segmimg']

            for path, t in zip(paths, types):
                if path.exists():
                    print(f'Reading {obj.basename}.{t} from {path}')
                    setattr(obj, t, FITSImage.from_file(f'{path}'))

        return obj


class CalibratableImage(CalibratableImageBase, ZTFFile):

    id = sa.Column(sa.Integer, sa.ForeignKey('ztffiles.id',
                                             ondelete='CASCADE'),
                   primary_key=True)

    __mapper_args__ = {'polymorphic_identity': 'calibratableimage',
                       'inherit_condition': id == ZTFFile.id}

    detections = relationship('Detection', cascade='all')

    mask_image = relationship('MaskImage',
                              uselist=False,
                              primaryjoin='MaskImage.parent_image_id == CalibratableImage.id')

    catalog = relationship('PipelineFITSCatalog', uselist=False,
                           primaryjoin=PipelineFITSCatalog.image_id == id)

    thumbnails = relationship('Thumbnail',
                              primaryjoin='Thumbnail.image_id == CalibratableImage.id')


    def basic_map(self, quiet=False):
        datadir = Path(get_secret('base_data_directory'))
        self.map_to_local_file(datadir / self.relname, quiet=quiet)
        self.mask_image.map_to_local_file(datadir / self.mask_image.relname,
                                          quiet=quiet)
        weightpath = datadir / self.relname.replace('.fits', '.weight.fits')
        if weightpath.exists():
            self._weightimg = FITSImage()
            self.weight_image.map_to_local_file(weightpath, quiet=quiet)

        else:
            rmspath = datadir / self.relname.replace('.fits', '.rms.fits')
            if rmspath.exists():
                self.rms_image.map_to_local_file(rmspath, quiet=quiet)
            else:
                raise FileNotFoundError(f'Neither "{weightpath}" nor '
                                        f'"{rmspath}" exists for '
                                        f'"{self.basename}".')


    @classmethod
    def from_file(cls, fname, use_existing_record=True, load_others=True):
        from .mask import MaskImage
        obj = super().from_file(
            fname, use_existing_record=use_existing_record,
            load_others=load_others
        )
        dir = Path(fname).parent

        if load_others:

            if obj.mask_image is not None:
                mskpath = dir / obj.mask_image.basename
                if mskpath.exists():
                    print(f'Reading {obj.basename}.mask_image from {mskpath}')
                    obj.mask_image = MaskImage.from_file(mskpath)

            if obj.catalog is not None:
                catpath = dir / obj.catalog.basename
                if catpath.exists():
                    print(f'Reading {obj.basename}.catalog from {catpath}')
                    obj.catalog = PipelineFITSCatalog.from_file(catpath)

        return obj


class CalibratedImage(CalibratableImage):
    """An image on which photometry can be performed."""

    id = sa.Column(sa.Integer, sa.ForeignKey('calibratableimages.id',
                                             ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'calibratedimage',
                       'inherit_condition': id == CalibratableImage.id}

    forced_photometry = relationship('ForcedPhotometry', cascade='all')

    def force_photometry(self, sources, assume_background_subtracted=False,
                         use_cutout=False, direct_load=None):
        """Force aperture photometry at the locations of `sources`.
        Assumes that calibration has already been done.

        """

        # ensure sources is at least 1d
        sources = np.atleast_1d(sources)

        ra = [source.ra for source in sources]
        dec = [source.dec for source in sources]

        result = aperture_photometry(
            self, ra, dec, apply_calibration=True,
            assume_background_subtracted=assume_background_subtracted,
            use_cutout=use_cutout, direct_load=direct_load
        )

        photometry = []
        for row, source, r, d in zip(result, sources, ra, dec):
            phot = ForcedPhotometry(flux=row['flux'],
                                    fluxerr=row['fluxerr'],
                                    flags=int(row['flags']),
                                    image=self,
                                    ra=r,
                                    dec=d,
                                    source=source,
                                    zp=row['zp'],
                                    obsjd=row['obsjd'],
                                    filtercode=row['filtercode'])
            photometry.append(phot)

        return photometry

    @declared_attr
    def __table_args__(cls):
        # override spatial index - only need it on one table (calibratable images)
        return tuple()

    @hybrid_property
    def seeing(self):
        """FWHM of seeing in pixels."""
        return self.header['SEEING']

    @hybrid_property
    def magzp(self):
        return self.header['MAGZP']

    @hybrid_property
    def apcor(self):
        return self.header[APER_KEY]

    @property
    def mjd(self):
        pass

    def find_in_dir(self, directory):
        super().find_in_dir(directory)
        try:
            self.mask_image.find_in_dir(directory)
        except FileNotFoundError:
            pass

    @property
    def unphotometered_sources(self):

        from .source import Source

        jcond2 = sa.and_(
            ForcedPhotometry.image_id == self.id,
            ForcedPhotometry.source_id == Source.id
        )

        query = DBSession().query(
            Source
        ).outerjoin(
            ForcedPhotometry, jcond2
        ).filter(
            ForcedPhotometry.id == None
        ).filter(
            sa.func.q3c_poly_query(
                Source.ra,
                Source.dec,
                self.poly
            )
        )

        return query.all()


class ScienceImage(CalibratedImage):
    """IPAC record of a science image from their pipeline. Contains some
    metadata that IPAC makes available through its irsa metadata query
    service.  This class is primarily intended to enable the reflection of
    IPAC's idea of its science images and which science images exist so that
    IPAC's world can be compared against the results of this pipeline.

    This class does not map to any file on disk, in the sense that it is not
    designed to reflect the data or metadata of any local file to memory,
    but it can be used to download files from the IRSA archive to disk (
    again, it does not make any attempt to represent the contents of these
    files, or synchronize their contents on disk to their representation in
    memory).

    This class represents immutable metadata only.
    """

    # we dont want science image records to be deleted in a cascade.
    id = sa.Column(sa.Integer, sa.ForeignKey('calibratedimages.id',
                                             ondelete='RESTRICT'),
                   primary_key=True)
    __mapper_args__ = {'polymorphic_identity': 'sci',
                       'inherit_condition': id == CalibratedImage.id}

    @classmethod
    def from_file(cls, f, use_existing_record=True, load_others=True):
        obj = super().from_file(
            f, use_existing_record=use_existing_record,
            load_others=load_others
        )
        obj.field = obj.header['FIELDID']
        obj.ccdid = obj.header['CCDID']
        obj.qid = obj.header['QID']
        obj.fid = obj.header['FILTERID']

        if obj.filtercode is None:
            obj.filtercode = fid_map[obj.fid]

        fname = obj.header['FILENAME']

        if obj.imgtypecode is None:
            obj.imgtypecode = fname.split('.')[0][-1]

        if obj.filefracday is None:
            obj.filefracday = int(fname.split('_')[1])

        for attr, hkw in zip(['obsjd', 'infobits',
                              'pid', 'nid', 'expid',
                              'seeing', 'airmass', 'moonillf', 'moonesb',
                              'maglimit', 'crpix1', 'crpix2', 'crval1',
                              'crval2', 'cd11', 'cd12', 'cd21', 'cd22',
                              'ipac_gid', 'exptime'],
                             ['OBSJD', 'INFOBITS', 'DBPID',
                              'DBNID', 'DBEXPID', 'SEEING',
                              'AIRMASS', 'MOONILLF', 'MOONESB',
                              'MAGLIM', 'CRPIX1', 'CRPIX2',
                              'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2',
                              'CD2_1', 'CD2_2', 'PROGRMID', 'EXPTIME']):

            if getattr(obj, attr) is None:
                setattr(obj, attr, obj.header[hkw])

        return obj

    filtercode = sa.Column(sa.CHAR(2))
    obsjd = sa.Column(psql.DOUBLE_PRECISION)
    infobits = sa.Column(sa.Integer)
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
    ipac_gid = sa.Column(sa.Integer)
    imgtypecode = sa.Column(sa.CHAR(1))
    exptime = sa.Column(sa.Float)
    filefracday = sa.Column(psql.BIGINT)

    nidind = sa.Index('sci_nid_ind', nid)
    jdind = sa.Index("sci_obsjd_ind", obsjd)

    @hybrid_property
    def obsmjd(self):
        return self.obsjd - 2400000.5

    @property
    def mjd(self):
        return self.obsmjd

    @hybrid_property
    def filter(self):
        return 'ztf' + self.filtercode[-1]

    def ipac_path(self, suffix):
        """The url of a particular file corresponding to this metadata
        record, with suffix `suffix`, in the IPAC archive. """
        sffd = str(self.filefracday)
        return f'https://irsa.ipac.caltech.edu/ibe/data/ztf/' \
               f'products/sci/{sffd[:4]}/{sffd[4:8]}/{sffd[8:]}/' \
               f'ztf_{sffd}_{self.field:06d}_' \
               f'{self.filtercode}_c{self.ccdid:02d}_' \
               f'{self.imgtypecode}_q{self.qid}_{suffix}'
