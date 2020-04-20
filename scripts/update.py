import zuds
import numpy as np
import pandas as pd
from datetime import datetime
from ztfquery import query
from astropy.time import Time

QUERY_WINDOWSIZE = 30 # days
WHERECLAUSE = ''


# this script should be run as a cron job to update metadata table of ZTF images

if __name__ == '__main__':

    zuds.init_db()

    zquery = query.ZTFQuery()
    start = datetime.now()

    # get the maximum nid
    sm = zuds.DBSession().query(zuds.ScienceImage).order_by(
        zuds.ScienceImage.obsjd.desc()
    ).first()

    if sm is None:
        max_jd = 2458165.6030208
        max_nid = 411
        max_ffd = 20180216102002

    else:
        max_jd = sm.obsjd
        max_ffd = sm.filefracday
        max_nid = sm.nid

    current_jd = Time.now().jd

    metatables = []
    jd_diff = current_jd - max_jd
    n_chunks = int(jd_diff // QUERY_WINDOWSIZE) + 1
    print('querying')

    for i in range(n_chunks):

        nid_lo = max_nid + i * QUERY_WINDOWSIZE
        nid_hi = max_nid + (i + 1) * QUERY_WINDOWSIZE

        zquery.load_metadata(kind='sci', sql_query=WHERECLAUSE + (' AND ' if WHERECLAUSE != '' else '') +
                             f'NID >= {nid_lo} AND NID <= {nid_hi} AND '
                             f'IPAC_GID > 0',
                             auth=[zuds.get_secret('ipac_username'),
                                   zuds.get_secret('ipac_password')])
        metatable = zquery.metatable
        metatables.append(metatable)

    metatable = pd.concat(metatables)
    current_paths = zuds.DBSession().query(zuds.ScienceImage.basename).filter(
        zuds.ScienceImage.nid == max_nid
    ).all()
    print(f'pulled {len(metatable)} images')

    # dont need this
    del metatable['imgtype']
    del metatable['ipac_pub_date']
    del metatable['rcid']

    meta_images = [zuds.ScienceImage(**row.to_dict()) for _, row
                   in metatable.iterrows()]
    meta_masks = [zuds.MaskImage(parent_image=s, field=s.field, qid=s.qid,
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

        zuds.DBSession().add(meta_images[index])
        zuds.DBSession().add(meta_masks[index])

    #for basename, img, mask in zip(basenames, meta_images, meta_masks):
    #    img.basename = basename
    #    mask.basename = basename.replace('sciimg', 'mskimg')



    #db.DBSession().add_all(meta_images)
    #db.DBSession().add_all(meta_masks)
    zuds.DBSession().commit()

    end = datetime.now()

    print(f'done in {end - start}')
