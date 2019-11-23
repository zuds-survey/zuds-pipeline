import db
import os
import sys
import mpi

db.init_db()

__author__ = 'Danny Goldstein <danny@caltech.edu>'
__whatami__ = 'Make the weight maps for ZUDS.'

infile = sys.argv[1]  # file listing all the images to build weight maps for

# get the work
sci_fns = mpi.get_my_share_of_work(infile)

cwd = os.getcwd()

# load the objects into memory
for fn in sci_fns:

    # this line assumes that .mskimg.fits is in the same directory
    sci = db.ScienceImage.from_file(fn)
    weightname = os.path.abspath(fn.replace('.fits', '.weight.fits'))
    rmsname = os.path.abspath(fn.replace('.fits', '.rms.fits'))

    # save the results
    os.chdir(os.path.dirname(fn))
    if not os.path.exists(weightname):

        sci.weight_image.save()


    if not os.path.exists(rmsname):
        sci.rms_image.save()

    os.chdir(cwd)

    # save new header 
    db.DBSession().commit()

