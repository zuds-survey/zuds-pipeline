import db
import sys
import mpi
import dosub
import send
import makesources
import traceback
from argparse import ArgumentParser

if __name__ == '__main__':

    infile = sys.argv[1]
    refvers = sys.argv[2]

    # subclass = db.MultiEpochSubtraction
    # sciclass = db.ScienceCoadd

    subclass = db.SingleEpochSubtraction
    sciclass = db.ScienceImage

    # get the work
    imgs = mpi.get_my_share_of_work(infile)
    for fn in imgs:

        # commits
        try:
            detections, sub = dosub.do_one(fn, sciclass, subclass, refvers)
        except Exception as e:
            db.DBSession().rollback()
            traceback.print_exception(*sys.exc_info())
            continue

        db.DBSession().flush()
        for d in detections:
            # each call commits
            makesources.associate(d, do_historical_phot=True)

        # requires manual commit
        fp = sub.force_photometry(sub.unphotometered_sources,
                                  assume_background_subtracted=True)
        db.DBSession().add_all(fp)
        db.DBSession().flush()

        # try to conserve memory?
        sub.target_image.unmap()
        sub.reference_image.unmap()
        sub.unmap()

        # issue an alert for each detection
        alerts = []
        for d in detections:
            if d.source is not None:
                alert = db.Alert.from_detection(d)
                db.DBSession().add(alert)
                alerts.append(alert)

        db.DBSession().commit()
        for alert in alerts:
            send.send_alert(alert)
            alert.sent = True
            db.DBSession().add(alert)
            db.DBSession().commit()


