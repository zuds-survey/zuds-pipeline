import db
import numpy as np
from ztfquery import query

QUERY_WINDOWSIZE = 30 # days
WHERECLAUSE = ''
FIRST_NID = 411

# this script should be run as a cron job to update metadata table of ZTF images

if __name__ == '__main__':

    zquery = query.ZTFQuery()

    # get the maximum nid
    max_nid = db.DBSession().query(db.Image.nid.max()).first()
    max_date = db.DBSession().query(db.Image.obsdate.max()).first()
    today = db.DBSession().query(db.func.now()).first()
    todays_nid = int(max_nid + (today - max_date))

    zquery.load_metadata(WHERECLAUSE + (' AND' if WHERECLAUSE == '' else '') +
                         f'NID BETWEEN {max_nid + 1} AND {todays_nid}')
    metatable = zquery.metatable

    current_paths = db.DBSession().query(db.Image.path).all()
    meta_images = [db.Image(row.to_dict()) for _, row in metatable.iterrows()]
    basenames = [i.ipac_path('sciimg.fits').split('/')[-1] for i in meta_images]

    indices = np.nonzero(np.in1d(basenames, current_paths))

    for index in indices:
        meta_images[index].path = basenames[index]
        db.DBSession().add(meta_images[index])
    db.DBSession().commit()
