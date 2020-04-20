import time
import sqlalchemy as sa
import numpy as np

from sqlalchemy.orm import relationship
from sqlalchemy.dialects import postgresql as psql

from copy import deepcopy

from .crossmatch import xmatch
from .core import Base, DBSession
from .utils import print_time
from .constants import MJD_TO_JD

__all__ = ['Alert']

class Alert(Base):



    alert = sa.Column(psql.JSONB)
    creation_index = sa.Index('created_at_index', 'created_at')

    cutoutscience_id = sa.Column(sa.Integer, sa.ForeignKey(
        'thumbnails.id', ondelete='SET NULL'
    ))

    cutouttemplate_id = sa.Column(sa.Integer, sa.ForeignKey(
        'thumbnails.id', ondelete='SET NULL'
    ))

    cutoutdifference_id = sa.Column(sa.Integer, sa.ForeignKey(
        'thumbnails.id', ondelete='SET NULL'
    ))

    detection_id = sa.Column(sa.Integer, sa.ForeignKey(
        'detections.id', ondelete='SET NULL'
    ), index=True)

    cutoutscience = relationship('Thumbnail', cascade='all',
                                 foreign_keys=[cutoutscience_id])
    cutouttemplate = relationship('Thumbnail', cascade='all',
                                  foreign_keys=[cutouttemplate_id])
    cutoutdifference = relationship('Thumbnail', cascade='all',
                                    foreign_keys=[cutoutdifference_id])

    detection = relationship('Detection', cascade='all',
                             foreign_keys=[detection_id])

    sent = sa.Column(sa.Boolean, default=False)

    def to_dict(self):
        base = deepcopy(self.alert)
        base['cutoutScience'] = self.cutoutscience.bytes
        base['cutoutTemplate'] = self.cutouttemplate.bytes
        base['cutoutDifference'] = self.cutoutdifference.bytes
        return base

    @classmethod
    def from_detection(cls, detection):

        from .subtraction import SingleEpochSubtraction, MultiEpochSubtraction
        from .image import ScienceImage
        from .coadd import CoaddImage, ScienceCoadd
        from .detections import Detection

        obj = cls()


        if detection.source is None:
            raise ValueError('Cannot issue an alert for a detection that is '
                             f'not associated with a source ({detection.id})')

        # prepare the JSONB
        alert = dict()
        alert['objectId'] = detection.source.id
        alert['candid'] = detection.id
        alert['schemavsn'] = '0.4'
        alert['publisher'] = 'ZUDS/NERSC'

        # do a bunch of cross matches to initially populate the candidate
        # subschema
        start = time.time()
        candidate = xmatch(detection.ra, detection.dec, detection.source.id)
        alert['candidate'] = candidate
        stop = time.time()
        print_time(start, stop, detection, 'xmatch')


        # indicate whether this alert is generated based on a stack detection
        #  or a single epoch detection
        start = time.time()
        alert_type = 'single' if isinstance(
            detection.image, SingleEpochSubtraction
        ) else 'stack'
        candidate['alert_type'] = alert_type
        stop = time.time()
        print_time(start, stop, detection , 'alert_type')


        # add some basic info about the image this was found on and metadata
        start = time.time()
        candidate['fid'] = detection.image.fid
        candidate['pid'] = detection.image.id

        if isinstance(detection.image, SingleEpochSubtraction):
            candidate['programpi'] = detection.image.target_image.header['PROGRMPI']
            candidate['programid'] = detection.image.target_image.header['PROGRMID']
        else:
            candidate['programpi'] = \
                detection.image.target_image.input_images[0].header['PROGRMPI']
            candidate['programid'] = \
                detection.image.target_image.input_images[0].header['PROGRMID']
        candidate['pdiffimfilename'] = detection.image.basename
        candidate['candid'] = detection.id
        candidate['isdiffpos'] = 't'
        candidate['field'] = detection.image.field
        candidate['ra'] = detection.ra
        candidate['dec'] = detection.dec
        candidate['rcid'] = (detection.image.ccdid - 1) * 4 + (
            detection.image.qid - 1)

        # shape and position information
        candidate['aimage'] = detection.a_image
        candidate['bimage'] = detection.b_image
        candidate['elong'] = detection.elongation
        candidate['fwhm'] = detection.fwhm_image
        candidate['aimagerat'] = detection.a_image / detection.fwhm_image
        candidate['bimagerat'] = detection.b_image / detection.fwhm_image
        candidate['xpos'] = detection.x_image
        candidate['ypos'] = detection.y_image

        # flux information
        candidate['snr'] = detection.snr

        # machine learning
        candidate['drb'] = detection.rb[0].rb_score
        candidate['drbversion'] = detection.rb[0].rb_version
        stop = time.time()
        print_time(start, stop, detection, 'basic')

        # information about the reference images

        start = time.time()

        refmin, refmax, refcount = DBSession().query(
            sa.func.min(ScienceImage.obsjd) - MJD_TO_JD,
            sa.func.max(ScienceImage.obsjd) - MJD_TO_JD,
            sa.func.count(ScienceImage.id)
        ).select_from(ScienceImage.__table__).join(
            CoaddImage, ScienceImage.id == CoaddImage.calibratableimage_id
        ).filter(
            CoaddImage.coadd_id == detection.image.reference_image_id
        ).first()
        stop = time.time()
        print_time(start, stop, detection, 'get refimgs')

        start = time.time()

        if alert_type == 'single':
            candidate['jd'] = detection.image.mjd + MJD_TO_JD
            candidate['nid'] = detection.image.target_image.nid
            candidate['diffmaglim'] = detection.image.target_image.maglimit
            candidate['exptime'] = detection.image.target_image.exptime
            mjdcut = detection.image.mjd
        else:
            stackimgs = sorted(
                detection.image.target_image.input_images,
                key=lambda i: i.mjd
            )
            candidate['jdstartstack'] = stackimgs[0].mjd + MJD_TO_JD
            candidate['jdendstack'] = stackimgs[-1].mjd + MJD_TO_JD
            candidate['jdmed'] = np.median([i.mjd + MJD_TO_JD for i in stackimgs])
            candidate['nframesstack'] = len(stackimgs)
            candidate['exptime'] = np.sum([i.exptime for i in stackimgs])
            mjdcut = stackimgs[-1].mjd

        # add half a second to mjdcut to ensure expected behavior in
        # floating point comparisons
        mjdcut += 0.5 / 3600. / 24.

        stop = time.time()
        print_time(start, stop, detection, 'histstats')

        start = time.time()

        # calculate the detection history

        if alert_type == 'single':
            mymjd = detection.image.mjd
        else:
            mymjd = detection.image.target_image.max_mjd

        # get dates of single epoch detections
        singledates = DBSession().query(ScienceImage.obsjd - MJD_TO_JD).select_from(
            ScienceImage.__table__
        ).join(
            SingleEpochSubtraction.__table__, SingleEpochSubtraction.target_image_id == ScienceImage.id
        ).join(
            Detection.__table__, Detection.image_id == SingleEpochSubtraction.id
        ).filter(
            Detection.source_id == detection.source.id
        ).all()
        singledates = list(sorted([date[0] for date in singledates if date[0] < mymjd]))

        multidates = DBSession().query(sa.func.min(ScienceImage.obsjd) - MJD_TO_JD,
                                       sa.func.max(ScienceImage.obsjd) - MJD_TO_JD
        ).select_from(
            ScienceImage.__table__
        ).join(
            CoaddImage, CoaddImage.calibratableimage_id == ScienceImage.id
        ).join(
            ScienceCoadd.__table__, ScienceCoadd.id == CoaddImage.coadd_id
        ).join(
            MultiEpochSubtraction.__table__, MultiEpochSubtraction.target_image_id == ScienceCoadd.id
        ).join(
            Detection.__table__, Detection.image_id == MultiEpochSubtraction.id
        ).filter(
            Detection.source_id == detection.source.id
        ).group_by(ScienceCoadd.id)

        multidates = list(sorted([date for date in multidates if date[0] < mymjd],
                                 key=lambda d: d[1]))

        candidate['ndethist_single'] = len(singledates)
        candidate['ndethist_stack'] = len(multidates)

        stop = time.time()
        print_time(start, stop, detection, 'prevdets')

        start = time.time()

        if len(singledates) > 0:
            starthist = singledates[0] + MJD_TO_JD
            endhist = singledates[-1] + MJD_TO_JD
            candidate['jdstarthist_single'] = starthist
            candidate['jdendhist_single'] = endhist
        else:
            candidate['jdstarthist_single'] = None
            candidate['jdendhist_single'] = None

        if len(multidates) > 0:
            starthist = multidates[0][0] + MJD_TO_JD
            endhist = multidates[-1][1] + MJD_TO_JD
            candidate['jdstarthist_stack'] = starthist
            candidate['jdendhist_stack'] = endhist
        else:
            candidate['jdstarthist_stack'] = None
            candidate['jdendhist_stack'] = None

        stop = time.time()
        print_time(start, stop, detection, 'jdstarthist')

        start = time.time()
        # make the light curve
        lc = detection.source.light_curve()
        stop = time.time()
        print_time(start, stop, detection, 'light_curve')

        if len(lc) == 0:
            raise RuntimeError('Cannot issue an alert for an object with no '
                               'light curve, please rerun forced photometry '
                               'to update the light curve of this object:' 
                               f'"{detection.source.id}".')

        lc = lc[lc['mjd'] <= mjdcut]
        alert['light_curve'] = lc.to_pandas().to_dict(orient='records')

        candidate['jdstartref'] = refmin
        candidate['jdendref'] = refmax
        candidate['nframesref'] = refcount

        start = time.time()
        # now do the cutouts
        for stamp in detection.thumbnails:
            if stamp.type == 'ref':
                obj.cutouttemplate = stamp
            elif stamp.type == 'new':
                obj.cutoutscience = stamp
            elif stamp.type == 'sub':
                obj.cutoutdifference = stamp

        stop = time.time()
        print_time(start, stop, detection, 'cutouts')

        obj.alert = alert

        # this is to prevent detections from being re-inserted, triggering
        # a unique key violation

        obj.detection_id = detection.id
        return obj
