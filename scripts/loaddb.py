import db
db.init_db()  # zuds2
import numpy as np
import pandas as pd

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mpi = True
except ImportError:
    mpi = False
    rank = 0
    size = 1

fid_map = {
    'zg': 1,
    'zr': 2,
    'zi': 3
}

nrows = 18892958

import os
jobarray_idx = int(os.getenv('SLURM_ARRAY_TASK_ID'))
jobarray_count = int(os.getenv('SLURM_ARRAY_TASK_MAX')) + 1

# data row indices
inds = np.arange(1, nrows)
inds = np.array_split(inds, jobarray_count)[jobarray_idx]

splinds = np.array_split(inds, size)
my_inds = splinds[rank]

lb = min(my_inds)
skiprows = lambda r: 0 < r < lb

nrows = len(my_inds)  # number of DATA ROWS to read

myframes = pd.read_csv('/global/cscratch1/sd/dgold/image.csv',
                       skiprows=skiprows, nrows=nrows)

for i, row in myframes.iterrows():
    poly_dict = {}
    for key in ['ra', 'dec', 'ra1', 'dec1', 'ra2', 'dec2', 'ra3',
                'dec3', 'ra4', 'dec4']:
        poly_dict[key] = row[key]

    sci = db.ScienceImage(basename=row['path'], filtercode=row['filtercode'],
                          qid=row['qid'], field=row['field'],
                          ccdid=row['ccdid'],
                          obsjd=row['obsjd'], infobits=row['infobits'],
                          pid=row['pid'], nid=row['nid'],
                          expid=row['expid'], itid=row['itid'],
                          obsdate=row['obsdate'],
                          seeing=row['seeing'], airmass=row['airmass'],
                          moonillf=row['moonillf'], moonesb=row['moonesb'],
                          maglimit=row['maglimit'],
                          crpix1=row['crpix1'], crpix2=row['crpix2'],
                          crval1=row['crval1'],
                          crval2=row['crval2'], cd11=row['cd11'],
                          cd12=row['cd12'],
                          cd21=row['cd21'],
                          cd22=row['cd22'], ipac_gid=row['ipac_gid'],
                          imgtypecode=row['imgtypecode'],
                          exptime=row['exptime'],
                          filefracday=row['filefracday'],
                          fid=fid_map[row['filtercode']], **poly_dict)

    msk = db.MaskImage(parent_image=sci,
                       basename=row['path'].replace('sciimg', 'mskimg'),
                       field=row['field'],
                       ccdid=row['ccdid'], fid=fid_map[row['filtercode']],
                       qid=row['qid'], **poly_dict)

    hpss_sci_path = row['hpss_sci_path']
    if not hpss_sci_path.isna():
        scicopy = db.TapeCopy(archive_id=hpss_sci_path, product=sci)
        db.DBSession().add(scicopy)

    hpss_mask_path = row['hpss_mask_path']
    if not hpss_mask_path.isna():
        maskcopy = db.TapeCopy(archive_id=hpss_mask_path, product=msk)
        db.DBSession().add(maskcopy)

    db.DBSession().add(msk)
    db.DBSession().add(sci)
db.DBSession().commit()


