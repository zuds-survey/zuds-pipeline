import os
import sys
import shutil

news = sys.argv[1:-1]
coadd = sys.argv[-1]

coadd_dir = os.path.dirname(coadd)

for new in news:
    for suffix in ['.weight.fits', '.bpm.fits', '.rms.fits', '.cat']:
        inf = new.replace('.fits', suffix)
        shutil.copy(inf, coadd_dir)
