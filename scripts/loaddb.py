import db
db.init_db()  # zuds2

import pandas as pd

# this may need to be parallelized
df = pd.read_csv('../../image.csv')

archives = {}
# first group by "hpss_sci_path"

fid_map = {
    'zg': 1,
    'zr': 2,
    'zi': 3
}

archives = {}

# load in chunks of 50k
for i, row in df.iterrows():
    sci = db.ScienceImage(basename=row['path'], filtercode=row['filtercode'],
                          qid=row['qid'], field=row['field'], ccdid=row['ccdid'],
                          obsjd=row['obsjd'], infobits=row['infobits'],
                          rcid=row['rcid'], pid=row['pid'], nid=row['nid'],
                          expid=row['expid'], itid=row['itid'], obsdate=row['obsdate'],
                          seeing=row['seeing'], airmass=row['airmass'],
                          moonillf=row['moonillf'], moonesb=row['moonesb'], maglimit=row['maglimit'],
                          crpix1=row['crpix1'], crpix2=row['crpix2'], crval1=row['crval1'],
                          crval2=row['crval2'], cd11=row['cd11'], cd12=row['cd12'], cd21=row['cd21'],
                          cd22=row['cd22'], ipac_gid=row['ipac_gid'], imgtypecode=row['imgtypecode'],
                          exptime=row['exptime'], filefracday=row['filefracday'], fid=fid_map[row['filtercode']])

    msk = db.MaskImage(parent_image=sci, basename=row['path'].replace('sciimg', 'mskimg'),
                       field=row['field'], ccdid=row['ccdid'], fid=fid_map[row['filtercode']],
                       qid=row['qid'])


    hpss_sci_path = row['hpss_sci_path']
    if hpss_sci_path is not None:
        try:
            scitar = archives[hpss_sci_path]
        except KeyError:
            scitar = db.TapeArchive(id=hpss_sci_path)
            archives[hpss_sci_path] = scitar
            db.DBSession().add(scitar)

        scicopy = db.TapeCopy(archive=scitar, product=sci)
        db.DBSession().add(scicopy)

    hpss_mask_path = row['hpss_mask_path']
    if hpss_mask_path is not None:
        try:
            masktar = archives[hpss_mask_path]
        except KeyError:
            masktar = db.TapeArchive(id=hpss_mask_path)
            archives[hpss_mask_path] = masktar
            db.DBSession().add(masktar)
        maskcopy = db.TapeCopy(archive=masktar, product=mask)
        db.DBSession().add(masktar)

    db.DBSession().add(msk)
    db.DBSession().add(sci)

    if (i != 0 and i % 50000 == 0) or (i == len(df) - 1):
        db.DBSession().commit()
        

