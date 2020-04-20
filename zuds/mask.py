import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy import Index

from .image import FITSImage
from .core import ZTFFile
from .constants import MASK_BITS, BAD_SUM, MASK_COMMENTS


__all__ = ['MaskImageBase', 'MaskImage']


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



