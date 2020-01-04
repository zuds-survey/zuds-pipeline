import db
import sys
import mpi
import os
import time
import archive
import numpy as np
from sqlalchemy.orm import aliased
import publish
from tqdm import tqdm

fmap = {1: 'zg',
        2: 'zr',
        3: 'zi'}

db.init_db()
#db.DBSession().autoflush = False
#db.DBSession().get_bind().echo = True

# association radius 2 arcsec
ASSOC_RADIUS = 2 * 0.0002777
N_PREV_SINGLE = 1
N_PREV_MULTI = 1
DEFAULT_GROUP = 1
DEFAULT_INSTRUMENT = 1


__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Associate detections into sources for ZUDS.'


# get distinct fields
fields = db.DBSession().query(
    db.CalibratableImage.field.distinct()
).select_from(db.Detection).join(
    db.CalibratableImage
).all()

# split the list up
my_fields = mpi.get_my_share_of_work(fields, reader=lambda x: x)

# get unassigned detections

for field in my_fields:

    unassigned = db.DBSession().query(
        db.Detection
    ).filter(
        db.Detection.source_id == None,
        db.CalibratableImage.field == field
    ).join(
        db.CalibratableImage
    ).order_by(
        db.CalibratableImage.basename.asc()
    )

    for detection in tqdm(unassigned.all()):

        with db.DBSession().no_autoflush:
            source = db.DBSession().query(db.models.Source).filter(
                db.sa.func.q3c_radial_query(
                    db.models.Source.ra,
                    db.models.Source.dec,
                    detection.ra,
                    detection.dec,
                    ASSOC_RADIUS
                )
            ).first()

        if source is None:

            # source creation logic. at least 2 single epoch detections,
            # or at least 1 stack detection

            # get other detections nearby

            with db.DBSession().no_autoflush:
                prev_dets = db.DBSession().query(
                    db.Detection,
                    db.CalibratableImage.type
                ).join(db.CalibratableImage).filter(
                    db.sa.func.q3c_radial_query(
                        db.Detection.ra,
                        db.Detection.dec,
                        detection.ra,
                        detection.dec,
                        ASSOC_RADIUS
                    ),
                    db.Detection.id != detection.id,
                    db.CalibratableImage.field == field
                ).all()


            n_prev_single = sum([1 for _ in prev_dets if _[1] == 'sesub'])
            n_prev_multi = sum([1 for _ in prev_dets if _[1] == 'mesub'])

            incr_single = 1 if isinstance(detection.image,
                                          db.SingleEpochSubtraction) else 0
            incr_multi = 1 if isinstance(detection.image,
                                         db.MultiEpochSubtraction) else 0

            single_criteria = n_prev_single + incr_single > N_PREV_SINGLE
            multi_criteria = n_prev_multi + incr_multi > N_PREV_MULTI
            create_new_source = single_criteria or multi_criteria

            if create_new_source:

                with db.DBSession().no_autoflush:
                    default_group = db.DBSession().query(
                        db.models.Group
                    ).get(DEFAULT_GROUP)

                    default_instrument = db.DBSession().query(
                        db.models.Instrument
                    ).get(DEFAULT_INSTRUMENT)

                # need to create a new source
                name = publish.get_next_name()
                source = db.models.Source(
                    id=name,
                    ra=detection.ra,
                    dec=detection.dec,
                    groups=[default_group]
                )

                # need this to make stamps.
                dummy_phot = db.models.Photometry(
                    source=source,
                    instrument=default_instrument
                )

                for det, _ in prev_dets:
                    det.source = source
                    db.DBSession().add(det)

                detection.source = source

                db.DBSession().add(dummy_phot)
                db.DBSession().add(source)
                db.DBSession().flush()

                for t in detection.thumbnails:
                    t.photometry = dummy_phot
                    t.source = source
                    t.persist()
                    db.DBSession().add(t)

            else:
                continue
        else:
            detection.source = source

        # update the source ra and dec
        # best = source.best_detection

        # just doing this in case the new LC point
        # isn't yet flushed to the DB

        # if detection.snr > best.snr:
        #    best = detection

        # source.ra = best.ra
        # source.dec = best.dec

        db.DBSession().flush()

        if len(source.thumbnails) == len(source.photometry[0].thumbnails):
            lthumbs = source.return_linked_thumbnails()
            db.DBSession().add_all(lthumbs)
        db.DBSession().add(detection)

db.DBSession().commit()

