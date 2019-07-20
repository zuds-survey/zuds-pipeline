import os
import sys
import shutil

news = sys.argv[1:-1]
coadd = sys.argv[-1]

coadd_dir = os.path.dirname(coadd)

for new in news:
    for suffix in ['.weight.fits', '.bpm.fits', '.rms.fits', '.cat']:
        basename = os.path.basename(new).replace('.fits', suffix)
        dirname = os.path.dirname(new)
        inf = os.path.join(dirname, '..', basename)
        shutil.copy(inf, coadd_dir)
