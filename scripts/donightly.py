import os
import sys
import zuds
import dosub
import shutil
import traceback


if __name__ == '__main__':

    send_alerts = True
    zuds.init_db()

    infile = sys.argv[1]
    refvers = sys.argv[2]

    subclass = zuds.SingleEpochSubtraction
    sciclass = zuds.ScienceImage

    # get the work
    imgs = zuds.get_my_share_of_work(infile)

    subs = []
    dirs = []
    all_detections = []
    for inpt in imgs:

        s = zuds.ScienceImage.get_by_basename(os.path.basename(inpt))
        fn = f'/global/cfs/cdirs/m937/www/data/scratch/{s.field:06d}/' \
             f'c{s.ccdid:02d}/q{s.qid}/{zuds.fid_map[s.fid]}/{s.basename}'

        shutil.copy(inpt, fn)
        shutil.copy(
            inpt.replace('sciimg', 'mskimg'),
            fn.replace('sciimg', 'mskimg')
        )

        # commits
        try:
            detections, sub = dosub.do_one(fn, sciclass, subclass, refvers, tmpdir='tmp')
        except (dosub.TooManyDetectionsError, OSError, ValueError) as e:
            zuds.DBSession().rollback()
            print(f'Error: too many detections on {fn} sub')
            sci = zuds.ScienceImage.get_by_basename(os.path.basename(fn))
            ref = zuds.DBSession().query(
                zuds.ReferenceImage
            ).filter(
                zuds.ReferenceImage.field == sci.field,
                zuds.ReferenceImage.ccdid == sci.ccdid,
                zuds.ReferenceImage.qid == sci.qid,
                zuds.ReferenceImage.fid == sci.fid,
                zuds.ReferenceImage.version == refvers
            ).first()
            blocker = zuds.FailedSubtraction(
                target_image=sci,
                reference_image=ref,
                reason=str(e)
            )
            zuds.DBSession().add(blocker)
            zuds.DBSession().commit()
            continue
        except dosub.PredecessorError as e:
            zuds.DBSession().rollback()
            sci = zuds.ScienceImage.get_by_basename(os.path.basename(fn))
            ref = zuds.DBSession().query(
                zuds.ReferenceImage
            ).filter(
                zuds.ReferenceImage.field == sci.field,
                zuds.ReferenceImage.ccdid == sci.ccdid,
                zuds.ReferenceImage.qid == sci.qid,
                zuds.ReferenceImage.fid == sci.fid,
                zuds.ReferenceImage.version == refvers
            ).first()
            basename = zuds.sub_name(sci.basename, ref.basename)
            subname = os.path.join(os.path.dirname(fn), basename)
            prev = subclass.from_file(subname)
            subs.append(prev)
            dirs.append(os.path.dirname(fn))
            continue

        except Exception as e:
            zuds.DBSession().rollback()
            traceback.print_exception(*sys.exc_info())
            continue

        else:
            subs.append(sub)
            dirs.append(os.path.dirname(fn))
            zuds.DBSession().add(sub)
            zuds.DBSession().add_all(detections)
            zuds.DBSession().commit()
