import db
import sys
import mpi
import os
import traceback
import time
import archive
import numpy as np

fmap = {1: 'zg',
        2: 'zr',
        3: 'zi'}

db.init_db()
#db.DBSession().autoflush = False
#db.DBSession().get_bind().echo = True

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make the subtractions for ZUDS.'

MAX_DETS = 50

class TooManyDetectionsError(Exception):
    pass

class PredecessorError(Exception):
    pass



# make a reference for each directory
def do_one(fn, sciclass, subclass, refvers, tmpdir='/tmp'):
    tstart = time.time()

    sstart = time.time()
    sci = sciclass.get_by_basename(os.path.basename(fn))
    sci.map_to_local_file(fn)
    maskname = os.path.join(os.path.dirname(fn), sci.mask_image.basename)
    sci.mask_image.map_to_local_file(maskname)

    weightname = fn.replace('.fits', '.weight.fits')
    rmsname = fn.replace('.fits', '.rms.fits')
    if os.path.exists(weightname):
        sci._weightimg = db.FITSImage.from_file(weightname)
    elif os.path.exists(rmsname):
        sci._rmsimg = db.FITSImage.from_file(rmsname)
    else:
        if sciclass == db.ScienceImage:
            # use sextractor to make the science image
            _ = sci.rms_image
        else:
            raise RuntimeError(f'Cannot produce a subtraction for {fn},'
                               f' the image has no weightmap or rms map.')


    sstop = time.time()
    print(
        f'sci: {sstop-sstart:.2f} sec to load  {sci.basename}',
        flush=True
    )

    field = f'{sci.field:06d}'
    ccdid = f'c{sci.ccdid:02d}'
    qid = f'q{sci.qid}'
    fid = f'{fmap[sci.fid]}'
    refname = f'/global/cfs/cdirs/m937/www/data/scratch/{field}/{ccdid}/{qid}/' \
              f'{fid}/ref.{field}_{ccdid}_{qid}_{fid}.{refvers}.fits'

    if not (db.ReferenceImage.get_by_basename(os.path.basename(refname))
            and os.path.exists(refname)):
        db.DBSession().rollback()
        raise RuntimeError(f'Ref {refname} does not exist. Skipping...')


    rstart = time.time()

    ref = db.ReferenceImage.get_by_basename(os.path.basename(refname))
    ref.map_to_local_file(refname)
    ref.mask_image.map_to_local_file(refname.replace('.fits', '.mask.fits'))
    ref._weightimg = db.FITSImage.from_file(refname.replace('.fits',
                                                            '.weight.fits'))
    rstop = time.time()

    print(
        f'ref: {rstop-rstart:.2f} sec to load ref for {sci.basename}',
        flush=True
    )


    basename = db.sub_name(sci.basename, ref.basename)

    prev = subclass.get_by_basename(basename)
    #prev=None

    #if (prev is not None) and (prev.modified is not None) and \
    #   (prev.modified > datetime.now() - timedelta(hours=24)):
    #    db.DBSession().rollback()
    #    continue

    if prev is not None:
        raise PredecessorError(f'{basename} already has a predecessor')

    substart = time.time()
    sub = subclass.from_images(sci, ref,
                               data_product=False,
                               tmpdir=tmpdir)

    substop = time.time()
    print(
        f'sub: {substop-substart:.2f} sec to make {sub.basename}',
        flush=True
    )


    catstart = time.time()
    cat = db.PipelineFITSCatalog.from_image(sub)

    catstop = time.time()
    print(
        f'cat: {catstop-catstart:.2f} sec to make catalog for {sub.basename}',
        flush=True
    )

    dstart = time.time()
    detections = db.Detection.from_catalog(cat, filter=True)

    if len(detections) > MAX_DETS:
        raise TooManyDetectionsError(
            f'Error: {len(detections)} detections (>{MAX_DETS}) '
            f'on "{sub.basename}", something wrong with the image probably'
        )

    dstop = time.time()
    print(
        f'det: {dstop-dstart:.2f} sec to make detections for {sub.basename}',
        flush=True
    )

    stampstart = time.time()

    if isinstance(sub, db.SingleEpochSubtraction):
        sub_target = sub.aligned_to(sub.reference_image)
    else:
        sub_target = sub
    if isinstance(sub.target_image, db.ScienceImage):
        new_target = sub.target_image.aligned_to(sub.reference_image)
    else:
        new_target = sub.target_image
    stamps = []
    for detection in detections:
        for i in [sub_target, new_target, sub.reference_image]:
            # make a stamp for the first detection
            stamp = db.models.Thumbnail.from_detection(
                detection, i
            )
            stamps.append(stamp)

    archstart = time.time()
    #subcopy = db.HTTPArchiveCopy.from_product(sub)
    #catcopy = db.HTTPArchiveCopy.from_product(cat)
    #mskcopy = db.HTTPArchiveCopy.from_product(sub.mask_image)
    db.DBSession().add(sub)
    #db.DBSession().add(cat)
    db.DBSession().add_all(detections)
    db.DBSession().add_all(stamps)

    #db.DBSession().add(mskcopy)
    #db.DBSession().add(catcopy)
    #db.DBSession().add(subcopy)
    #archive.archive(subcopy)
    #archive.archive(catcopy)
    #archive.archive(mskcopy)
    #db.DBSession().commit()
    archstop = time.time()
    print(
        f'archive: {archstop-archstart:.2f} sec to archive stuff for '
        f'{sub.basename}',
        flush=True
    )



    cleanstart = time.time()
    sci.unmap()
    cleanstop = time.time()

    tstop = time.time()
    print(f'clean: took {cleanstop - cleanstart} sec to clean '
          f'up after {sub.basename}"',
          flush=True)
    print(f'took {tstop - tstart} sec to make "{sub.basename}"', flush=True)

    return detections, sub


if __name__ == '__main__':

    infile = sys.argv[1]  # file listing all the images to make subtractions of
    refvers = sys.argv[2]

    subclass = db.MultiEpochSubtraction
    sciclass = db.ScienceCoadd

    #subclass = db.SingleEpochSubtraction
    #sciclass = db.ScienceImage

    # get the work
    imgs = mpi.get_my_share_of_work(infile)
    for fn in imgs:
        try:
            detections, sub = do_one(fn, sciclass, subclass, refvers)
        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            db.DBSession().rollback()
            continue
        else:
            db.DBSession().commit()

