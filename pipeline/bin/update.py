import db
import numpy as np
import pandas as pd
from datetime import datetime
from ztfquery import query

QUERY_WINDOWSIZE = 30 # days
WHERECLAUSE = ''

# this script should be run as a cron job to update metadata table of ZTF images

if __name__ == '__main__':

    env, cfg = db.load_env()
    db.init_db(**cfg['database'])

    zquery = query.ZTFQuery()

    # get the maximum nid
    max_nid = db.DBSession().query(db.sa.func.max(db.Image.nid)).first()[0]
    max_date = db.DBSession().query(db.sa.func.max(db.Image.obsdate)).first()[0]
    today = datetime.utcnow()
    todays_nid = max_nid + (today - max_date).days

    metatables = []
    nid_diff = todays_nid - max_nid
    quotient = nid_diff // QUERY_WINDOWSIZE
    mod = nid_diff % QUERY_WINDOWSIZE

    n_chunks = quotient if mod == 0 else quotient + 1

    for i in range(n_chunks):
        zquery.load_metadata(kind='sci', sql_query=WHERECLAUSE + (' AND ' if WHERECLAUSE != '' else '') +
                             f'NID BETWEEN {max_nid + QUERY_WINDOWSIZE * i + 1} AND '
                             f'{max_nid + QUERY_WINDOWSIZE * (i + 1)}')
        metatable = zquery.metatable
        metatables.append(metatable)

    metatable = pd.concat(metatables)
    current_paths = db.DBSession().query(db.Image.path).all()

    # dont need this
    del metatable['imgtype']
    del metatable['ipac_pub_date']

    meta_images = [db.Image(**row.to_dict()) for _, row in metatable.iterrows() if _ < 10]
    basenames = [i.ipac_path('sciimg.fits').split('/')[-1] for i in meta_images]

    indices = np.nonzero(~np.in1d(basenames, current_paths, assume_unique=True))

    for index in indices:
        meta_images[index].path = basenames[index]
        db.DBSession().add(meta_images[index])
    db.DBSession().commit()
