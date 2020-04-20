import time
import zuds


__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make alerts for ZUDS.'

# get unalerted detections
unalerted = zuds.DBSession().query(
    zuds.Detection
).filter(
    zuds.Detection.source_id != None
).outerjoin(
    zuds.Alert
).filter(
    zuds.Alert.id == None
).all()

print(f'Need to make alerts for {len(unalerted)} detections')

alerts = []
for detection in unalerted:
    tstart = time.time()
    alert = zuds.Alert.from_detection(detection)
    alerts.append(alert)
    tstop = time.time()
    print(f'took {tstop - tstart:.2f} sec to make alert '
          f'for {detection.source_id}')
