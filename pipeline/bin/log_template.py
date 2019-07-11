import db
import sys
from astropy.io import fits

fname = sys.argv[1]
inimids = sys.argv[2]

env, cfg = db.load_env()
db.init_db(**cfg['database'])

with fits.open(fname) as hdul:
    header = hdul[0].header

q = db.DBSession().query(db.Reference)
d = {}

for col in q.column_descriptions:
    if col['name'] not in ['id', 'created_at', 'modified']:
        d[col['name']] = header[col['name'].upper()]

ref = db.Reference(**d)
db.DBSession().add(ref)
db.DBSession().commit()

for imid in map(int, inimids.split()):
    assoc = db.ReferenceImage(imag_id=imid, reference_id=ref.id)
    db.DBSession().add(assoc)
db.DBSession().commit()
