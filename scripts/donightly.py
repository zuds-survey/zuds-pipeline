import db
import os
import sys
import time
import mpi
import dosub
import send
import shutil
import makesources
import traceback
from argparse import ArgumentParser

fid_map = {1: 'zg', 2: 'zr', 3: 'zi'}

if __name__ == '__main__':

    infile = sys.argv[1]
    refvers = sys.argv[2]

    # subclass = db.MultiEpochSubtraction
    # sciclass = db.ScienceCoadd

    subclass = db.SingleEpochSubtraction
    sciclass = db.ScienceImage

    # get the work
    imgs = mpi.get_my_share_of_work(infile)

    subs = []
    all_detections = []
    for inpt in imgs:

        s = db.ScienceImage.get_by_basename(os.path.basename(inpt))
        fn = f'/global/cscratch1/sd/dgold/zuds/{s.field:06d}/' \
             f'c{s.ccdid:02d}/q{s.qid}/{fid_map[s.fid]}/{s.basename}'

        shutil.copy(inpt, fn)
        shutil.copy(
            inpt.replace('sciimg', 'mskimg'),
            fn.replace('sciimg', 'mskimg')
        )

        # commits
        try:
            detections, sub = dosub.do_one(fn, sciclass, subclass, refvers)
        except dosub.TooManyDetectionsError as e:
            db.DBSession().rollback()
            print(f'Error: too many detections on {fn} sub')
            sci = db.ScienceImage.get_by_basename(os.path.basename(fn))
            ref = db.DBSession().query(
                db.ReferenceImage
            ).filter(
                db.ReferenceImage.field == sci.field,
                db.ReferenceImage.ccdid == sci.ccdid,
                db.ReferenceImage.qid == sci.qid,
                db.ReferenceImage.fid == sci.fid,
                db.ReferenceImage.version == refvers
            ).first()
            blocker = db.FailedSubtraction(
                target_image=sci,
                reference_image=ref,
                reason=str(e)
            )
            db.DBSession().add(blocker)
            continue

        except Exception as e:
            db.DBSession().rollback()
            traceback.print_exception(*sys.exc_info())
            continue

        all_detections.extend(detections)
        subs.append(sub)

        fp = sub.force_photometry(sub.unphotometered_sources,
                                  assume_background_subtracted=True)
        db.DBSession().add_all(fp)
        db.DBSession().add(sub)
        db.DBSession().add_all(detections)

    db.DBSession().commit()

    issue_alert = {}
    for sub in subs:
        for d in sub.detections:
            tstart = time.time()
            needs_alert = makesources.associate(d, do_historical_phot=True)
            tstop = time.time()

            issue_alert[d] = needs_alert
    db.DBSession().commit()

    # issue an alert for each detection
    alerts = []
    for sub in subs:
        for d in sub.detections:
            if issue_alert[d]:
                alert = db.Alert.from_detection(d)
                db.DBSession().add(alert)
                alerts.append(alert)

    db.DBSession().commit()
    for alert in alerts:
        send.send_alert(alert)
        alert.sent = True
        db.DBSession().add(alert)
        db.DBSession().commit()
