import db
import sys

if __name__ == '__main__':

    env, cfg = db.load_env()
    db.init_db(**cfg['database'])

    jobid = int(sys.argv[1])

    hpssjob = None

    while hpssjob is None:
        hpssjob = db.DBSession().query(db.HPSSJob).get(jobid)

    hpssjob.status = True
    db.DBSession().add(hpssjob)
    db.DBSession().commit()
