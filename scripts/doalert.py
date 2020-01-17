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
#db.DBSession().autoflush = False
#db.DBSession().get_bind().echo = True

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make the subtractions for ZUDS.'

infile = sys.argv[1]  # file listing all the subs to do photometry on

my_work = mpi.get_my_share_of_work(infile)

for detid in my_work:
    d = db.DBSession().query(db.Detection).get(detid)
    alert = db.Alert.from_detection(d)
    db.DBSession().add(alert)
    db.DBSession().commit()
    send.send_alert(alert)
    alert.sent = True
    db.DBSession().add(alert)
    db.DBSession().commit()
