import db
import sys
import mpi
import os
import time
import archive
import numpy as np
from sqlalchemy.orm import aliased
import publish

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make alerts for ZUDS.'

# get unalerted detections
unalerted = db.DBSession().query(
    db.Detection
).filter(
    db.Detection.source_id != None
).outerjoin(
    db.Alert
).filter(
    db.Alert.id == None
).all()

print(f'Need to make alerts for {len(unalerted)} detections')

alerts = []
for detection in unalerted:
    tstart = time.time()
    alert = db.Alert.from_detection(detection)
    alerts.append(alert)
    tstop = time.time()
    print(f'took {tstop - tstart:.2f} sec to make alert '
          f'for {detection.source_id}')
