import db
import sys
import mpi
import os
import time
import archive
from datetime import datetime, timedelta

fmap = {1: 'zg',
        2: 'zr',
        3: 'zi'}

db.init_db()
#db.DBSession().autoflush = False
#db.DBSession().get_bind().echo = True

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make the subtractions for ZUDS.'

infile = sys.argv[1]  # file listing all the subs to do photometry on


# get the work
imgs = mpi.get_my_share_of_work(infile)

for fn in imgs:
    sub = db.SingleEpochSubtraction.from_file(fn)

    start = time.time()
    sources = sub.unphotometered_sources
    if len(sources) == 0:
        continue

    try:
        phot = sub.force_photometry(sources,
                                    assume_background_subtracted=True)
    except Exception as e:
        print(e)
        continue

    stop = time.time()

    db.DBSession().add_all(phot)
    db.DBSession().commit()
    print(f'phot: took {stop-start:.2f} sec to do phot on {sub.basename}')


