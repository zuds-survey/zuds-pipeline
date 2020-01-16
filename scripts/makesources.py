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
import logging
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

if mpi.has_mpi():
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    FORMAT = f'%(asctime)-15s {rank} %(message)s'
else:
    FORMAT = f'%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


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


def associate(detection):


    # assume `detection` is only visible to this transactionnn

    if detection.source is not None:
        # it was assigned in a previous iteration of this loop
        logging.debug(f'detection {detection.id} is alread associated with '
                      f'{detection.source_id}, skipping...')
        # do nothing
        return

    if all([r.rb_score < db.RB_ASSOC_MIN for r in detection.rb]):
        logging.debug(f'detection {detection.id} does not have a high enough '
                      f'rb score to be associated')
        return

    else:

        # source creation logic. at least 2 single epoch detections,
        # or at least 1 stack detection

        # get other detections nearby

        # if source is none then the associatable detections are already in
        # the `unassigned` list, so cross match against that

        logging.debug(f'querying for detections within 2 arcsec of {detection.id}')

        start = time.time()

        match_dets = db.DBSession().query(
            db.Detection,
            db.models.Source,
            db.CalibratableImage.type
        ).join(
            db.CalibratableImage,
        ).join(
            db.RealBogus
        ).outerjoin(
            db.models.Source, db.Detection.source_id == db.Source.id
        ).filter(
            db.sa.func.q3c_radial_query(
                db.Detection.ra,
                db.Detection.dec,
                detection.ra,
                detection.dec,
                ASSOC_RADIUS
            ),
            db.RealBogus.rb_score > db.RB_ASSOC_MIN
        ).with_for_update(of=[db.Detection.__table__,
                              db.ObjectWithFlux.__table__,
                              db.models.Source.__table__]).all()
        stop = time.time()

        # remove myself from the match detections
        for i, dd in enumerate(match_dets):
            if dd[0] == detection:
                match_dets.remove(dd)

        logging.debug(f'found {len(match_dets)} with rb > 0.2 in 2arcsec of {detection.id}, '
                      f'locking detections and objectswithflux rows: '
                      f'{[d[0].id for d in match_dets]}, took {stop-start:.2f} sec '
                      f'to execute the query')

        for _, s, __ in match_dets:
            if s is not None:
                logging.debug(f'one of the locked detections within 2arcsec of {detection.id} '
                              f'is associated with a source: {m.source.id}. now locking'
                              f'that source...')
                start = time.time()
                stop = time.time()
                logging.debug(f'took {stop-start:.2f} sec to lock source {m.source.id}'
                              f' for {detection.id}. now associating detection with '
                              f'source and updating ra/dec...')
                prev_dets = [m[0] for m in match_dets]
                _update_source_coordinate(s, prev_dets + [detection])
                detection.source = s
                detection.triggers_phot = False
                detection.triggers_alert = True
                logging.debug(f'done associating detection {detection.id} with'
                              f'source {m.source.id} and done updating source location')
                return

        logging.debug('did not find any sources associated with detections within'
                      f' 2 arcsec of {detection.id}. '
                      'determining if a new source should be created...')


        n_prev_single = sum([1 for _ in match_dets if _[2] == 'sesub'])
        n_prev_multi = sum([1 for _ in match_dets if _[2] == 'mesub'])

        incr_single = 1 if isinstance(detection.image,
                                      db.SingleEpochSubtraction) else 0
        incr_multi = 1 if isinstance(detection.image,
                                     db.MultiEpochSubtraction) else 0

        single_criteria = n_prev_single + incr_single > N_PREV_SINGLE
        multi_criteria = n_prev_multi + incr_multi > N_PREV_MULTI

        create_new_source = single_criteria or multi_criteria

        if create_new_source:

            logging.debug(f'Source creation criteria met for detection {detection.id}. '
                          f'Detection {detection.id} has {n_prev_single} previous'
                          f'single detections and {n_prev_multi} previous multi'
                          f'epoch detections. '
                          f'Making a new source now...')

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

            logging.debug(f'Locally created an unpersisted source called {name}. '
                          f'for detection {detection.id}')

            udets = [m[0] for m in match_dets] + [detection]
            _update_source_coordinate(source, udets)

            # need this to make stamps.
            dummy_phot = db.models.Photometry(
                source=source,
                instrument=default_instrument
            )

            for det, _ in match_dets:
                logging.debug(f'As part of association of detection {detection.id},'
                              f'associating detection {det.id}.source = {source.id} ')
                det.source = source
                db.DBSession().add(det)

            logging.debug(f'As part of association of detection {detection.id},'
                          f'associating detection {detection.id}.source = {source.id} ')
            detection.source = source

            db.DBSession().add(dummy_phot)
            db.DBSession().add(source)

            logging.debug(f'Finished associating detection {detection.id}')

            detection.triggers_alert = True
            detection.triggers_phot = True

        else:
            logging.debug(f'No new source to be created for deteection {detection.id}')
            detection.triggers_phot = False
            detection.triggers_alert = False


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

    for d in tqdm(unassigned):
        associate(d)
        db.DBSession().add(d)
        db.DBSession().commit()
        if d.triggers_phot:
            # it's a new source -- update thumbnails post commit.
            # doing this post commit (with the triggers_phot flag)
            # avoids database deadlocks

            for t in d.thumbnails:
                t.photometry = d.source.photometry[0]
                t.source = d.source
                t.persist()
                db.DBSession().add(t)

            # update the source ra and dec
            # best = source.best_detection

            # just doing this in case the new LC point
            # isn't yet flushed to the DB

            # if detection.snr > best.snr:
            #    best = detection

            # source.ra = best.ra
            # source.dec = best.dec

            db.DBSession().flush()

            if len(d.source.thumbnails) == len(d.source.photometry[0].thumbnails):
                lthumbs = d.source.return_linked_thumbnails()
                db.DBSession().add_all(lthumbs)
            db.DBSession().add(d)
            db.DBSession().commit()

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
