import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as psql
from sqlalchemy import func
from pathlib import Path

from astropy.wcs.utils import proj_plane_pixel_scales, WCS
from astropy import units as u

from astropy.io import fits
import fitsio  # for catalogs, faster

import numpy as np

from .file import File, UnmappedFileError
from .core import Base, DBSession
from .spatial import HasPoly, SpatiallyIndexed


__all__ = ['FITSFile', 'HasWCS']


class FITSFile(File):
    """A python object that maps a fits file. Instances of classes mixed with
    FITSFile that implement `Base` map to database rows that store fits file
    metadata.

    Assumes the fits file has only one extension of interest. TODO: add
    support for multi-extension fits files.

    Provides methods for loading fits file data from disk to memory,
    and writing it to disk from memory.

    Methods for loading fits file metadata from the database and back are
    handled by sqlalchemy.
    """

    header = sa.Column(psql.JSONB)
    header_comments = sa.Column(psql.JSONB)
    __diskmapped_cached_properties__ = ['_path', '_data']
    _DATA_HDU = 0
    _HEADER_HDU = 0

    @classmethod
    def from_file(cls, f, use_existing_record=True):
        """Read a file into memory from disk, and set the values of
        database-backed variables that store metadata (e.g., header). These
        can later be flushed to the database using SQLalchemy.

        This is a 'get_or_create' method."""
        f = Path(f)

        load_from_db = issubclass(cls, Base) and \
                       issubclass(cls, FITSFile) and \
                       use_existing_record

        if load_from_db:
            obj = cls.get_by_basename(f.name)
        else:
            obj = None
        if obj is None:
            obj = cls()
            obj.basename = f.name
        else:
            # this should never have to be called, as things are unmapped in
            # get_by_basename
            if obj.ismapped:
                obj.unmap() #  force things to be reloaded from disk
        obj.map_to_local_file(str(f.absolute()))
        obj.load_header()

        return obj

    def load_header(self):
        """Load a header from disk into memory. Sets the values of
        database-backed variables that store metadata. These can later be
        flushed to the database using SQLalchemy."""
        with fits.open(self.local_path) as hdul:
            hd = dict(hdul[self._HEADER_HDU].header)
            hdc = {card.keyword: card.comment
                   for card in hdul[self._HEADER_HDU].header.cards}
        hd2 = hd.copy()
        for k in hd:
            if not isinstance(hd[k], (int, str, bool, float)):
                del hd2[k]
                del hdc[k]
        self.header = hd2
        self.header_comments = hdc

    def load_data(self):
        """Load data from disk into memory"""
        with fits.open(self.local_path, memmap=False) as hdul:  # throws
            # UnmappedFileError
            data = hdul[self._DATA_HDU].data
        if data.dtype.name == 'uint8':
            data = data.astype(bool)
        self._data = data

    def unload_data(self):
        try:
            del self._data
        except AttributeError:
            raise RuntimeError(f'Object "<{self.__class__.__name__} at '
                               f'{hex(id(self))}>" has no data loaded. '
                               f'Load some data with .load_data() and '
                               f'try again.')

    @property
    def data(self):
        """Data are read directly from a mapped file on disk, and cached in
        memory to avoid unnecessary IO.

        This property can be modified, but in order to save the resulting
        changes to disk the save() method must be called.
        """
        try:
            return self._data
        except AttributeError:
            # load the data into memory
            self.load_data()
        return self._data

    @data.setter
    def data(self, d):
        """Update the data member of this object in memory only. To flush
        this to disk you must call `save`."""
        self._data = d

    @property
    def astropy_header(self):
        """astropy.io.fits.Header representation of the database-mapped
        metadata columns header and header_comments. This attribute may not
        reflect the current header on disk"""
        if self.header is None or self.header_comments is None:
            # haven't seen the file on disk yet
            raise AttributeError(f'This image does not have a header '
                                 f'or headercomments record yet. Please map '
                                 f'this object to the corresponding '
                                 f'ScienceImage file on disk, load the header '
                                 f'with .load_header(), and retry.')
        else:
            header = fits.Header()
            header.update(self.header)
            for key in self.header_comments:
                header.comments[key] = self.header_comments[key]
            return header

    def save(self):
        from .image import FITSImage

        try:
            f = self.local_path
        except UnmappedFileError:
            f = self.basename
            self.map_to_local_file(f)
        dname = self.data.dtype.name
        if dname == 'bool':
            data = self.data.astype('uint8')
        else:
            data = self.data

        nhdu = max(self._DATA_HDU, self._HEADER_HDU) + 1


        if isinstance(self, FITSImage):
            if nhdu == 1:
                fits.writeto(f, data, self.astropy_header, overwrite=True)
            else:
                hdul = []
                for i in range(nhdu):
                    if i == 0:
                        hdu = fits.PrimaryHDU()
                    else:
                        hdu = fits.ImageHDU()
                    hdul.append(hdu)
                hdul = fits.HDUList(hdul)
                hdul[self._HEADER_HDU].header = self.astropy_header
                hdul[self._DATA_HDU].data = data

                hdul.writeto(f, overwrite=True)
        else:  # it's a catalog
            with fitsio.FITS(f, 'rw', clobber=True) as out:
                for i in range(nhdu):
                    data = None
                    header = None
                    if i == self._DATA_HDU:
                        data = self.data
                    if i == self._HEADER_HDU:
                        header = []
                        for key in self.header:
                            card = {
                                'name': key,
                                'value': self.header[key],
                                'comment': self.header_comments[key]
                            }
                            header.append(card)
                    out.write(data, header=header)


        self.unload_data()

    def load(self):
        self.load_header()
        self.load_data()


def needs_update(obj, key, value):
    # determine if an angle has changed enough in an object to warrant being
    # updated

    if not hasattr(obj, key):
        return True
    else:
        curval = getattr(obj, key)
        try:
            return not np.isclose(curval, value)
        except TypeError:
            # Duck typing - if np.isclose fails then curval is not a number
            # but instead is probably an sqlalchemy Column, thus needs an update
            return True


class HasWCS(FITSFile, HasPoly, SpatiallyIndexed):
    """Mixin indicating that an object represents a fits file with a WCS
    solution."""

    @property
    def wcs(self):
        """Astropy representation of the fits file's WCS solution.
        Lives in memory only."""
        return WCS(self.astropy_header)

    @classmethod
    def from_file(cls, fname, use_existing_record=True):
        """Read a fits file into memory from disk, and set the values of
        database-backed variables that store metadata (e.g., header). These
        can later be flushed to the database using SQLalchemy. """
        self = super(HasWCS, cls).from_file(
            fname, use_existing_record=use_existing_record,
        )
        corners = self.wcs.calc_footprint()
        for i, values in enumerate(corners):
            keys = [f'ra{i+1}', f'dec{i+1}']
            for key, value in zip(keys, values):
                if needs_update(self, key, value):
                    setattr(self, key, value)

        naxis1 = self.header['NAXIS1']
        naxis2 = self.header['NAXIS2']
        ra, dec = self.wcs.all_pix2world(
            [[naxis1 / 2, naxis2 / 2]], 1
        )[0]

        for key, value in zip(['ra', 'dec'], [ra, dec]):
            if needs_update(self, key, value):
                setattr(self, key, value)

        return self

    @property
    def sources_contained(self):
        """Query the database and return all `Sources` contained by the
        polygon of this object"""
        from .source import Source
        return DBSession().query(Source) \
            .filter(func.q3c_poly_query(Source.ra,
                                        Source.dec,
                                        self.poly))

    @property
    def pixel_scale(self):
        """The pixel scales of the detector in the X and Y image dimensions. """
        units = self.wcs.world_axis_units
        u1 = getattr(u, units[0])
        u2 = getattr(u, units[1])
        scales = proj_plane_pixel_scales(self.wcs)
        ps1 = (scales[0] * u1).to('arcsec').value
        ps2 = (scales[1] * u2).to('arcsec').value
        return np.asarray([ps1, ps2]) * u.arcsec

    def aligned_to(self, other, persist_aligned=False, tmpdir='/tmp',
                   nthreads=1):

        from .swarp import run_align

        """Return a version of this object that is pixel-by-pixel aligned to
        the WCS solution of another image with a WCS solution."""

        if not isinstance(other, HasWCS):
            raise ValueError(f'WCS Alignment target must be an instance of '
                             f'HasWCS (got "{other.__class__}").')

        new = run_align(self, other,
                        tmpdir=tmpdir,
                        nthreads=nthreads,
                        persist_aligned=persist_aligned)

        if hasattr(self, 'mask_image'):
            newmask = run_align(self.mask_image, other,
                                tmpdir=tmpdir,
                                nthreads=nthreads,
                                persist_aligned=persist_aligned)
            new.mask_image = newmask

        return new


