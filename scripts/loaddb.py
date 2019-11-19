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

# this may need to be parallelized
df = pd.read_csv('/global/cscratch1/sd/dgold/image.csv')

fid_map = {
    'zg': 1,
    'zr': 2,
    'zi': 3
}


if rank == 0:
    archives = pd.concat([df['hpss_sci_path'], df['hpss_mask_path']]).unique()
    tarchs = []
    arch_map = {}
    for archive in archives:
        if archive is None:
            continue
        dbarch = db.TapeArchive(id=archive)
        db.DBSession().add(dbarch)
        tarchs.append(dbarch)
    db.DBSession().commit()
    for dbarch in tarchs:
        arch_map[dbarch.id] = dbarch

    if mpi:
        subframes = np.array_split(df, size)

else:
    subframes = None
    arch_map = None

if mpi: 
    arch_map = comm.bcast(arch_map, root=0)
    myframes = comm.scatter(subframes, root=0)
else:
    myframes = df

# load in chunks of 50k
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
    if hpss_sci_path is not None:
        scitar = arch_map[hpss_sci_path]
        scicopy = db.TapeCopy(archive=scitar, product=sci)
        db.DBSession().add(scicopy)

    hpss_mask_path = row['hpss_mask_path']
    if hpss_mask_path is not None:
        masktar = arch_map[hpss_mask_path]
        maskcopy = db.TapeCopy(archive=masktar, product=msk)
        db.DBSession().add(maskcopy)

    db.DBSession().add(msk)
    db.DBSession().add(sci)

    if (i != 0 and i % 50000 == 0) or (i == len(myframes) - 1):
        db.DBSession().commit()


