import sys
import time
import zuds

zuds.init_db()
# db.DBSession().autoflush = False
# db.DBSession().get_bind().echo = True

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make the subtractions for ZUDS.'

infile = sys.argv[1]  # file listing all the subs to do photometry on

my_work = zuds.get_my_share_of_work(infile)

alerts = []
for detid in my_work:
    start = time.time()
    d = zuds.DBSession().query(zuds.Detection).get(int(detid))
    if d.alert is not None:
        alert = d.alert
    else:
        alert = zuds.Alert.from_detection(d)
        zuds.DBSession().add(alert)
        zuds.DBSession().commit()
    stop = time.time()
    print(f'made alert for {detid} ({d.source.id}) in {stop-start:.2f} sec',
          flush=True)
    if not alert.sent:
        zuds.send_alert(alert)
        alert.sent = True
        print(f'sent alert for {alert.detection.id} ({alert.detection.source.id})',
              flush=True)
        zuds.DBSession().add(alert)
        zuds.DBSession().commit()
