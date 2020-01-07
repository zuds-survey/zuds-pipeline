import os
from secrets import get_secret
import db
import numpy as np
import pandas as pd
from datetime import datetime
from ztfquery import query
from astropy.time import Time

QUERY_WINDOWSIZE = 30 # days
WHERECLAUSE = ''

# this script should be run as a cron job to update metadata table of ZTF images

if __name__ == '__main__':

    db.init_db()

    zquery = query.ZTFQuery()
    start = datetime.now()

    # get the maximum nid
    max_jd, max_ffd = db.DBSession().query(db.sa.func.max(
        db.ScienceImage.obsjd
    ), db.sa.func.max(
        db.ScienceImage.filefracday
    )).select_from(db.ScienceImage.__table__).first()
    current_jd = Time.now().jd

    metatables = []
    jd_diff = current_jd - max_jd
    n_chunks = int(jd_diff // QUERY_WINDOWSIZE) + 1
    print('querying')

    for i in range(n_chunks):

        jd_lo = max_jd + i * QUERY_WINDOWSIZE
        jd_hi = max_jd + (i + 1) * QUERY_WINDOWSIZE

        zquery.load_metadata(kind='sci', sql_query=WHERECLAUSE + (' AND ' if WHERECLAUSE != '' else '') +
                             f'OBSJD >= {jd_lo} AND OBSJD <= {jd_hi} AND IPAC_GID > 0',
                             auth=[get_secret('ipac_username'),
                                   get_secret('ipac_password')])
        metatable = zquery.metatable
        metatables.append(metatable)

    metatable = pd.concat(metatables)
    current_paths = db.DBSession().query(db.ScienceImage.basename).filter(
        db.ScienceImage.filefracday == max_ffd
    ).all()
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
                               dec4=s.dec4, ra=s.ra, dec=s.dec)
                  for s in meta_images]

    basenames = [i.ipac_path('sciimg.fits').split('/')[-1] for i in meta_images]

    indices = np.nonzero(~np.in1d(basenames, current_paths, assume_unique=True))[0]

    print(f'uploading images')

    for index in indices:
        meta_images[index].basename = basenames[index]
        meta_masks[index].basename = basenames[index].replace('sciimg',
                                                              'mskimg')

        db.DBSession().add(meta_images[index])
        db.DBSession().add(meta_masks[index])
        
    #for basename, img, mask in zip(basenames, meta_images, meta_masks):
    #    img.basename = basename
    #    mask.basename = basename.replace('sciimg', 'mskimg')

    

    #db.DBSession().add_all(meta_images)
    #db.DBSession().add_all(meta_masks)
    db.DBSession().commit()

    end = datetime.now()

    print(f'done in {end - start}')
