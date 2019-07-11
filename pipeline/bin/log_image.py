import db
import sys
from astropy.io import fits

fname = sys.argv[1]
id = int(sys.argv[2])
type = sys.argv[3]
dclass = getattr(db, type)

env, cfg = db.load_env()
db.init_db(**cfg['database'])

with fits.open(fname) as hdul:
    header = hdul[0].header

q = db.DBSession().query(dclass)
img = q.get(id)
d = {}


#for col in ref.__table__.columns.keys():
for key in header:
    dbkey = key.lower()
    if dbkey in img.__table__.columns.keys():
        setattr(img, dbkey, header[key])

db.DBSession().add(img)
db.DBSession().commit()

