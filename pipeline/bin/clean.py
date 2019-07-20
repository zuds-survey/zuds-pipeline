import os

def clean(dir):
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            fname = os.path.join(root, name)
            c1 = fname.endswith('sciimg.fits')
            c2 = fname.endswith('mskimg.fits')
            c3 = fname.endswith('deepref.fits')
            c4 = fname.endswith('.cat')
            c5 = fname.endswith('coadd.fits')
            c6 = fname.endswith('.weight.fits')
            c7 = fname.endswith('.mask.fits')
            if not any([c1, c2, c3, c4, c5, c6, c7]):
                os.remove(fname)
