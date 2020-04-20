import numpy as np
from astropy import units as u
BIG_RMS = np.sqrt(50000.)
BKG_BOX_SIZE = 128
DETECT_NSIGMA = 1.5
DETECT_NPIX = 5
MJD_TO_JD = 2400000.5
MATCH_RADIUS_DEG = 0.0002777 * 2.0
N_PREV_SINGLE = 1
N_PREV_MULTI = 1
RB_ASSOC_MIN = 0.2
CUTOUT_SIZE = 63  # pix
APER_KEY = 'APCOR4'
APERTURE_RADIUS = 3 * u.pixel
GROUP_PROPERTIES = ['field', 'ccdid', 'qid', 'fid']
NTHREADS_PER_NODE = 64
CMAP_RANDOM_SEED = 8675309
RB_CUT = {1: 0.3,
          2: 0.3,
          3: 0.6}
BRAAI_MODEL = 'braai_d6_m9'
MASK_BORDER = 10  # pix
BKG_VAL = 150.  # counts
