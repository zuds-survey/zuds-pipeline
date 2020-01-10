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
from astropy.coordinates import SkyCoord

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


# associate detection_id with a source or create one and associate
# the other detections with it

def _update_source_coordinate(source_object, detections):
    # assumes source_object and detections are locked

    snrs = [d.flux / d.fluxerr for d in detections]
    top = np.argmax(snrs)
    det = detections[top]
    source_object.ra = det.ra
    source_object.dec = det.dec


def associate(detection, do_historical_phot=False):

    # assume `detection` is only visible to this transactionnn

    if detection.source is not None:
        # it was assigned in a previous iteration of this loop
        print(f'detection {detection.id} is alread associated with '
              f'{detection.source_id}, skipping...')
        return
        #db.DBSession().rollback()

    else:

        source = db.DBSession().query(
            db.models.Source
        ).join(
            db.Detection
        ).filter(
            db.sa.func.q3c_radial_query(
                db.models.Source.ra,
                db.models.Source.dec,
                detection.ra,
                detection.dec,
                ASSOC_RADIUS
            )
        ).order_by(
            db.sa.func.q3c_dist(
                db.models.Source.ra,
                db.models.Source.dec,
                detection.ra,
                detection.dec
            ).asc()
        ).with_for_update(of=db.models.Source).first()

        if source is None:

            # source creation logic. at least 2 single epoch detections,
            # or at least 1 stack detection

            # get other detections nearby

            # if source is none then the associatable detections are already in
            # the `unassigned` list, so cross match against that

            match_dets = db.DBSession().query(
                db.Detection,
                db.CalibratableImage.type
            ).join(
                db.CalibratableImage
            ).filter(
                db.sa.func.q3c_radial_query(
                    db.Detection.ra,
                    db.Detection.dec,
                    detection.ra,
                    detection.dec,
                    ASSOC_RADIUS
                ),
                db.Detection.id != detection.id
            ).with_for_update(of=db.Detection).all()

            n_prev_single = sum([1 for _ in match_dets if _[1] == 'sesub'])
            n_prev_multi = sum([1 for _ in match_dets if _[1] == 'mesub'])

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

                udets = [m[0] for m in match_dets] + [detection]
                _update_source_coordinate(source, udets)

                # need this to make stamps.
                dummy_phot = db.models.Photometry(
                    source=source,
                    instrument=default_instrument
                )

                for det, _ in match_dets:
                    det.source = source
                    db.DBSession().add(det)

                detection.source = source

                db.DBSession().add(dummy_phot)
                db.DBSession().add(source)
                db.DBSession().flush()

                # run forced photometry on the new source
                if do_historical_phot:
                    fp = source.forced_photometry()
                    db.DBSession().add_all(fp)

                for t in detection.thumbnails:
                    t.photometry = dummy_phot
                    t.source = source
                    t.persist()
                    db.DBSession().add(t)

            else:
                # release the locks and move on
                #db.DBSession().rollback()
                return
        else:
            _update_source_coordinate(source, source.detections + [detection])
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




def associate_field_chip_quad(field, chip, quad):

    # to avoid race conditions, get an exclusive lock to  all the unassigned
    # detections from this field/chip

    unassigned = db.DBSession().query(
        db.Detection,
    ).filter(
        db.Detection.source_id == None,
        db.CalibratableImage.field == int(field),
        db.CalibratableImage.ccdid == int(chip),
        db.CalibratableImage.qid == int(quad)
    ).join(
        db.CalibratableImage
    ).order_by(
        db.CalibratableImage.basename.asc()
    )

    for detection in tqdm(unassigned):
        associate(detection.id)


if __name__ == '__main__':
    field_file = sys.argv[1]

    # split the list up
    def reader(fname):
        fields = np.genfromtxt(fname, dtype=None, encoding='ascii')
        result = []
        for f in fields:
            for i in range(1, 17):
                for j in range(1, 5):
                    result.append((f, i, j))

        return result


    work = mpi.get_my_share_of_work(field_file, reader=reader)
    for field, chip, quad in work:
        associate_field_chip_quad(field, chip, quad)
