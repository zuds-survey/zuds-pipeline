import db
import sys
import mpi
import os
import send
import time
import archive
from datetime import datetime, timedelta

fmap = {1: 'zg',
        2: 'zr',
        3: 'zi'}

db.init_db()
# db.DBSession().autoflush = False
# db.DBSession().get_bind().echo = True

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make the subtractions for ZUDS.'

infile = sys.argv[1]  # file listing all the subs to do photometry on

my_work = mpi.get_my_share_of_work(infile)

for thumbid in my_work:
    start = time.time()
    t = db.DBSession().query(db.models.Thumbnail).get(int(thumbid))
    t.persist()
    stop = time.time()
    db.print_time(start, stop, t, 'get and persist')

    start = time.time()
    db.DBSession().commit()
    stop = time.time()
    db.print_time(start, stop, t, 'commit')


