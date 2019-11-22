import os
from secrets import get_secret
import db
import numpy as np
import pandas as pd
from datetime import datetime
from ztfquery import query


QUERY_WINDOWSIZE = 30 # days
WHERECLAUSE = ''

# this script should be run as a cron job to update metadata table of ZTF images

if __name__ == '__main__':

    db.init_db()

    zquery = query.ZTFQuery()
    start = datetime.now()

    # get the maximum nid
    max_nid = db.DBSession().query(db.sa.func.max(db.ScienceImage.nid)).first()[0]
    max_date = db.DBSession().query(db.sa.func.max(db.ScienceImage.obsdate)).first()[0]
    today = datetime.utcnow()
    todays_nid = max_nid + (today - max_date).days

    metatables = []
    nid_diff = todays_nid - max_nid
    quotient = nid_diff // QUERY_WINDOWSIZE
    mod = nid_diff % QUERY_WINDOWSIZE

    n_chunks = quotient if mod == 0 else quotient + 1

    print('querying')

    for i in range(n_chunks):
        zquery.load_metadata(kind='sci', sql_query=WHERECLAUSE + (' AND ' if WHERECLAUSE != '' else '') +
                             f'NID BETWEEN {max_nid + QUERY_WINDOWSIZE * i} AND '
                             f'{max_nid + QUERY_WINDOWSIZE * (i + 1)} AND IPAC_GID > 0',
                             auth=[get_secret('ipac_username'),
                                   get_secret('ipac_password')])
        metatable = zquery.metatable
        metatables.append(metatable)

    metatable = pd.concat(metatables)
    current_paths = db.DBSession().query(db.ScienceImage.basename).all()
    print(f'pulled {len(metatable)} images')

    # dont need this
    del metatable['imgtype']
    del metatable['ipac_pub_date']
    del metatable['rcid']


    meta_images = [db.ScienceImage(**row.to_dict()) for _, row
                   in metatable.iterrows()]
    meta_masks = [db.MaskImage(parent_image=s, field=s.field, qid=s.qid,
                               ccdid=s.ccdid, fid=s.fid, ra1=s.ra1,
                               ra2=s.ra2, ra3=s.ra3, ra4=s.ra4,
                               dec1=s.dec1, dec2=s.dec2, dec3=s.dec3,
                               dec4=s.dec4, ra=s.ra, dec=s.dec,
                               basename=s.basename.replace('sciimg', 'mskimg'))
                  for s in meta_images]

    basenames = [i.ipac_path('sciimg.fits').split('/')[-1] for i in meta_images]

    indices = np.nonzero(~np.in1d(basenames, current_paths, assume_unique=True))[0]

    print(f'uploading images')

    for index in indices:
        meta_images[index].path = basenames[index]
        db.DBSession().add(meta_images[index])
    db.DBSession().commit()

    end = datetime.now()

    print(f'done in {end - start}')
