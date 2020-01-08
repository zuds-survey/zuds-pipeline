import db
import sys
import mpi
import dosub
import send
import makesources
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

        detections, sub = dosub.do_one(fn, sciclass, subclass, refvers)
        for d in detections:
            # each call commits
            makesources.associate(d.id, do_historical_phot=True)

        # requires manual commit
        sub.force_photometry(sub.unphotometered_sources,
                             assume_background_subtracted=True)
        db.DBSession().commit()

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

