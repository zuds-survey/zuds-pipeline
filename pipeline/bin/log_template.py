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

for col in q.column_descriptions:
    if col['name'] not in ['id', 'created_at', 'modified']:
        setattr(ref, col['name'], header[col['name'].upper()])

db.DBSession().add(ref)
db.DBSession().commit()

