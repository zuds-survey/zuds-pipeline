import db
import os
import sys
import time
import mpi
import numpy as np
import dosub
import send
import shutil
import makesources
import traceback
from argparse import ArgumentParser

fid_map = {1: 'zg', 2: 'zr', 3: 'zi'}

if __name__ == '__main__':

    send_alerts = False

    infile = sys.argv[1]
    refvers = sys.argv[2]

    # subclass = db.MultiEpochSubtraction
    # sciclass = db.ScienceCoadd

    subclass = db.SingleEpochSubtraction
    sciclass = db.ScienceImage

    # get the work
    imgs = mpi.get_my_share_of_work(infile)

    subs = []
    dirs = []
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
        except (dosub.TooManyDetectionsError, OSError) as e:
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

        except dosub.PredecessorError as e:
            db.DBSession().rollback()
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
            basename = db.sub_name(sci.basename, ref.basename)
            subname = os.path.join(os.path.dirname(fn), basename)
            prev = subclass.from_file(subname)
            subs.append(prev)
            dirs.append(os.path.dirname(fn))
            continue

        except Exception as e:
            db.DBSession().rollback()
            traceback.print_exception(*sys.exc_info())
            continue



        else:
            subs.append(sub)
            dirs.append(os.path.dirname(fn))
            db.DBSession().add(sub)
            db.DBSession().add_all(detections)
            db.DBSession().commit()

    for sub in subs:
        for d in sub.detections:
            tstart = time.time()
            makesources.associate(d)
            db.DBSession().commit()
            tstop = time.time()
            print(f'took {tstop-tstart:.2f} to associate {d.id}', flush=True)

    for sub, d in zip(subs, dirs):
        start = time.time()
        # have to remap the sub
        sub.find_in_dir(d)
        sub.mask_image.find_in_dir(d)


        sub._rmsimg = db.FITSImage()
        sub.rms_image.map_to_local_file(sub.local_path.replace(
            '.fits', '.rms.fits'
        ))

        sources = sub.unphotometered_sources,
        if len(sources) == 0:
            continue

        fp = sub.force_photometry(sources,
                                  assume_background_subtracted=True,
                                  use_cutout=True)
        stop = time.time()

        print(f'took {stop-start:.2f} sec for single-epoch photometry on '
              f'{sub.basename}', flush=True)

        db.DBSession().add_all(fp)
        db.DBSession().commit()

    # now do the forced photometry
    detections = []
    for sub in subs:
        for d in sub.detections:
            if d.triggers_phot:
                detections.append(d)

    tasks = {}
    for detection in detections:
        hits = detection.source.unphotometered_images
        for hit in hits:
            if hit not in tasks:
                tasks[hit] = [detection.source]
                hitname = '/global/cscratch1/sd/dgold/zuds/' \
                          f'{hit.field:06d}/c{hit.ccdid:02d}/' \
                          f'q{hit.qid}/{fid_map[hit.fid]}/{hit.basename}'
                hit.map_to_local_file(hitname)
                hit.mask_image.map_to_local_file(hitname.replace(
                    '.fits', '.mask.fits'
                ))
                rmsname = hitname.replace('.fits', '.rms.fits')
                hit._rmsimg = db.FITSImage()
                hit.rms_image.map_to_local_file(rmsname)


            else:
                tasks[hit].append(detection.source)

    for hit in tasks:
        sources = tasks[hit]
        start = time.time()

        try:
            fp = hit.force_photometry(
                sources, assume_background_subtracted=True
            )
        except np.AxisError as e:
            # this image doesn't contain the coordinate
            continue
        #h.mask_image.clear()
        #h.rms_image.clear()
        #h.clear()
        db.DBSession().add_all(fp)
        stop = time.time()
        print(f'took {stop-start:.2f} sec to do forcephot on {hit.basename}')
    db.DBSession().commit()

    # issue an alert for each detection

    alerts = []
    for sub in subs:
        for d in sub.detections:
            if d.triggers_alert and d.alert is None:
                alert = db.Alert.from_detection(d)
                db.DBSession().add(alert)
                alerts.append(alert)
                print(f'made alert for {d.id} (source {d.source.id})', flush=True)
                assert alert.alert['candidate']['jd'] == max(a['mjd'] for a in alert.alert['light_curve'])

    if send_alerts:
        db.DBSession().commit()

        for alert in alerts:
            send.send_alert(alert)
            print(f'sent alert for {alert.detection_id} '
                  f'(source {alert.detection.source.id})')
            alert.sent = True
            db.DBSession().add(alert)
            db.DBSession().commit()
            print(f'alert id {alert.id}, alert detection id {alert.detection_id}, '
                  f'detection id {alert.detection.id}, '
                  f'source id {alert.detection.source_id}', flush=True)
