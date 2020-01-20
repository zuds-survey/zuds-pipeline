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

alerts = []
for detid in my_work:
    start = time.time()
    d = db.DBSession().query(db.Detection).get(int(detid))
    if d.alert is not None:
        alert = d.alert
    else:
        alert = db.Alert.from_detection(d)
        db.DBSession().add(alert)
        db.DBSession().commit()
    stop = time.time()
    print(f'made alert for {detid} ({d.source.id}) in {stop-start:.2f} sec',
          flush=True)
    if not alert.sent:
        send.send_alert(alert)
        alert.sent = True
        print(f'sent alert for {alert.detection.id} ({alert.detection.source.id})',
              flush=True)
        db.DBSession().add(alert)
        db.DBSession().commit()
