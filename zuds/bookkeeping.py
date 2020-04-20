import sqlalchemy as sa
from sqlalchemy.orm import relationship

from .core import Base

__all__ = ['ForcePhotJob', 'AlertJob', 'FailedSubtraction']


class Job(Base):
    status = sa.Column(sa.Text, index=True, default='unsubmitted',
                       nullable=False)
    images = relationship('CalibratableImage', secondary='job_images')
    slurm_id = sa.Column(sa.Text, index=True, nullable=False)




class ForcePhotJob(Base):
    status = sa.Column(sa.Text, index=True, default='unsubmitted',
                       nullable=False)
    slurm_id = sa.Column(sa.Text, index=True, nullable=False)
    detection_file = sa.Column(sa.Text)
    output_file = sa.Column(sa.Text)


class AlertJob(Base):
    status = sa.Column(sa.Text, index=True, default='unsubmitted',
                       nullable=False)
    slurm_id = sa.Column(sa.Text, index=True, nullable=False)


class FailedSubtraction(Base):

    target_image_id = sa.Column(sa.Integer, sa.ForeignKey(
        'calibratableimages.id'
    ), index=True)
    reference_image_id = sa.Column(sa.Integer, sa.ForeignKey(
        'referenceimages.id'
    ), index=True)

    target_image = relationship('CalibratableImage', cascade='all',
                                foreign_keys=[target_image_id])
    reference_image = relationship('ReferenceImage', cascade='all',
                                   foreign_keys=[reference_image_id])

    reason = sa.Column(sa.Text)


