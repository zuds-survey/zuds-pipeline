
import db
import sys

jobid = sys.argv[1]

j = db.DBSession().query(db.Job).filter(db.Job.id == jobid).first()
j.status = 'complete'
db.DBSession().add(j)
db.DBSession().commit()

