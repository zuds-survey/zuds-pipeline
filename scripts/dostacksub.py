import db
import sys
import mpi
import os
import time
import archive
import pandas as pd

fmap = {1: 'zg',
        2: 'zr',
        3: 'zi'}

db.init_db()
db.DBSession().get_bind().echo = True

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make the references for ZUDS.'

infile = sys.argv[1]  # file listing all the images to make subtractions of
# get the work
jobs = mpi.get_my_share_of_work(infile, reader=pd.read_csv)


for _, job in jobs.iterrows():

    tstart = time.time()
    sstart = time.time()
    images = db.DBSession().query(db.SingleEpochSubtraction).filter(
        db.SingleEpochSubtraction.target_image_id.in_(eval(job['target']))
    ).all()

    sameprops = db.GROUP_PROPERTIES + ['reference_image_id']
    db.ensure_images_have_the_same_properties(images, sameprops)

    field = f'{images[0].field:06d}'
    ccdid = f'c{images[0].ccdid:02d}'
    qid = f'q{images[0].qid}'
    fid = f'{fmap[images[0].fid]}'

    for image in images:
        path = f'/global/cscratch1/sd/dgold/zuds/{field}/{ccdid}/{qid}/' \
               f'{fid}/{image.basename}'
        image.map_to_local_file(path)
        image.mask_image.map_to_local_file(path.replace('sciimg', 'mskimg'))

    basename = f'sub.{field}_{ccdid}_{qid}_{fid}_{job["left"]}_' \
               f'{job["right"]}.coadd.fits'

    prev = db.StackedSubtraction.get_by_basename(basename)
    outname = os.path.join(os.path.dirname(images[0].local_path), basename)
    sstop = time.time()

    print(
        f'load: {sstop-sstart:.2f} sec to load input images for {outname}',
        flush=True
    )

    stackstart = time.time()
    try:
        sub = db.StackedSubtraction.from_images(
            images, outfile_name=outname,
            data_product=False,
            tmpdir='tmp'
        )
    except Exception as e:
        print(e, [i.basename for i in images], flush=True)
        db.DBSession().rollback()
        continue

    sub.binleft = job['left']
    sub.binright = job['right']

    stackstop = time.time()
    print(
        f'stack: {stackstop-stackstart:.2f} sec to make {sub.basename}',
        flush=True
    )


    catstart = time.time()
    try:
        cat = db.PipelineFITSCatalog.from_image(sub)
    except Exception as e:
        print(e, [sub.basename], flush=True)
        db.DBSession.rollback()
        continue
    catstop = time.time()
    print(
        f'cat: {catstop-catstart:.2f} sec to make catalog for {sub.basename}',
        flush=True
    )

    dstart = time.time()
    try:
        detections = db.Detection.from_catalog(cat, filter=True)
    except Exception as e:
        print(e, [cat.basename], flush=True)
        db.DBSession.rollback()
        continue
    dstop = time.time()
    print(
        f'det: {dstop-dstart:.2f} sec to make detections for {sub.basename}',
        flush=True
    )

    stampstart = time.time()
    try:
        remapped = sub.reference_image.aligned_to(sub)
        stamps = []
        for i in [sub, sub.target_image, remapped]:
            for detection in detections:
                if len(detection.source.detections) == 0:
                    # make a stamp for the first detection
                    stamp = db.Stamp.from_detection(detection, i)
                    stamps.append(stamp)
    except Exception as e:
        print(e, [cat.basename], flush=True)
        db.DBSession.rollback()
        continue

    stampstop = time.time()
    print(
        f'stamp: {stampstop-stampstart:.2f} sec to make stamps for {sub.basename}',
        flush=True
    )

    archstart = time.time()
    scopy = db.HTTPArchiveCopy.from_product(sub)
    archive.archive(scopy)
    db.DBSession().add(scopy)
    db.DBSession().add(sub)
    db.DBSession().commit()
    archstop = time.time()

    print(
        f'archive: {archstop-archstart:.2f} sec to archive {sub.basename}',
        flush=True
    )

    cleanstart = time.time()
    targets = []
    for sci in images + [sub]:
        if hasattr(sci, '_rmsimg'):
            targets.append(sci.rms_image.local_path)
        if hasattr(sci, '_weightimg'):
            targets.append(sci.weight_image.local_path)

        sci.unmap()

    for target in targets:
        os.remove(target)

    cleanstop = time.time()

    tstop = time.time()
    print(f'clean: took {cleanstop - cleanstart} sec to clean '
          f'up after {sub.basename}"',
          flush=True)
    print(f'took {tstop - tstart} sec to make "{sub.basename}"', flush=True)
