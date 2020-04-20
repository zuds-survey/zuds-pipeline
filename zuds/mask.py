import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy import Index

from .image import FITSImage
from .core import ZTFFile


__all__ = ['MASK_BITS', 'BAD_SUM', 'BAD_BITS', 'MASK_COMMENTS',
           'MaskImageBase', 'MaskImage']


MASK_BITS = {
    'BIT00': 0,
    'BIT01': 1,
    'BIT02': 2,
    'BIT03': 3,
    'BIT04': 4,
    'BIT05': 5,
    'BIT06': 6,
    'BIT07': 7,
    'BIT08': 8,
    'BIT09': 9,
    'BIT10': 10,
    'BIT11': 11,
    'BIT12': 12,
    'BIT13': 13,
    'BIT14': 14,
    'BIT15': 15,
    'BIT16': 16
}

BAD_BITS = np.asarray([0, 2, 3, 4, 5, 7, 8, 9, 10, 16, 17])
BAD_SUM = int(np.sum(2 ** BAD_BITS))

MASK_COMMENTS = {
    'BIT00': 'AIRCRAFT/SATELLITE TRACK',
    'BIT01': 'CONTAINS SEXTRACTOR DETECTION',
    'BIT02': 'LOW RESPONSIVITY',
    'BIT03': 'HIGH RESPONSIVITY',
    'BIT04': 'NOISY',
    'BIT05': 'GHOST FROM BRIGHT SOURCE',
    'BIT06': 'RESERVED FOR FUTURE USE',
    'BIT07': 'PIXEL SPIKE (POSSIBLE RAD HIT)',
    'BIT08': 'SATURATED',
    'BIT09': 'DEAD (UNRESPONSIVE)',
    'BIT10': 'NAN (not a number)',
    'BIT11': 'CONTAINS PSF-EXTRACTED SOURCE POSITION',
    'BIT12': 'HALO FROM BRIGHT SOURCE',
    'BIT13': 'RESERVED FOR FUTURE USE',
    'BIT14': 'RESERVED FOR FUTURE USE',
    'BIT15': 'RESERVED FOR FUTURE USE',
    'BIT16': 'NON-DATA SECTION FROM SWARP ALIGNMENT'
}


class MaskImageBase(FITSImage):

    __diskmapped_cached_properties__ = FITSImage.__diskmapped_cached_properties__ + [
        '_boolean']


    def refresh_bit_mask_entries_in_header(self):
        """Update the database record and the disk record with a constant
        bitmap information stored in in memory. """
        self.header.update(MASK_BITS)
        self.header_comments.update(MASK_COMMENTS)
        self.save()

    def update_from_weight_map(self, weight_image):
        """Update what's in memory based on what's on disk (produced by
        SWarp)."""
        mskarr = self.data
        ftsarr = weight_image.data
        mskarr[ftsarr == 0] += 2 ** 16
        self.data = mskarr
        self.refresh_bit_mask_entries_in_header()

    @classmethod
    def from_file(cls, f, use_existing_record=True):
        return super().from_file(
            f, use_existing_record=use_existing_record,
        )

    @property
    def boolean(self):
        """A boolean array that is True when a masked pixel is 'bad', i.e.,
        not usable for science, and False when a pixel is not masked."""
        try:
            return self._boolean
        except AttributeError:

            # From ZSDS document:

            # For example, to find all science - image pixels which are
            # uncontaminated, but which may contain clean extracted - source
            # signal(bits 1 or 11), one would “logically AND” the
            # corresponding mask - image with the template value 6141 (= 2^0
            # + 2^2 +2^3 + 2^4 + 2^5 + 2^6 + 2^7 + 2^8 + 2^9 + 2^10 + 21^2)
            # #and retain pixels where this operation yields zero. To then
            # find which of these pixels contain source signal, one would
            # “AND” the resulting image with 2050 (= 21 + 211) and retain
            # pixels where this operation is non-zero.

            # DG: also have to add 2^16 (my custom bit) and 2^17 (hotpants
            # masked, another of my custom bits)
            # So 6141 -> 71677 --> 202749

            maskpix = (self.data & BAD_SUM) > 0
            _boolean = FITSImage()
            _boolean.data = maskpix
            _boolean.header = self.header
            _boolean.header_comments = self.header_comments
            _boolean.basename = self.basename.replace('.fits', '.bpm.fits')
            self._boolean = _boolean
        return self._boolean


class MaskImage(ZTFFile, MaskImageBase):

    id = sa.Column(sa.Integer, sa.ForeignKey('ztffiles.id',
                                             ondelete='CASCADE'),
                   primary_key=True)
    __mapper_args__ = {
        'polymorphic_identity': 'mask',
        'inherit_condition': id == ZTFFile.id
    }

    parent_image_id = sa.Column(sa.Integer,
                                sa.ForeignKey('calibratableimages.id',
                                              ondelete='CASCADE'))
    parent_image = relationship('CalibratableImage', cascade='all',
                                back_populates='mask_image',
                                foreign_keys=[parent_image_id])

    idx = Index('maskimages_parent_image_id_idx', parent_image_id)



