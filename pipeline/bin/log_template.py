import db
import sys
from astropy.io import fits

fname = sys.argv[1]
refid = int(sys.argv[2])

env, cfg = db.load_env()
db.init_db(**cfg['database'])

with fits.open(fname) as hdul:
    header = hdul[0].header

q = db.DBSession().query(db.Reference)
ref = q.get(refid)
d = {}


#for col in ref.__table__.columns.keys():
for key in header:
    dbkey = key.lower()
    if dbkey in ref.__table__.columns.keys():
        setattr(ref, dbkey, header[key])

db.DBSession().add(ref)
db.DBSession().commit()

