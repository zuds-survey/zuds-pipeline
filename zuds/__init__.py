import warnings
from sqlalchemy.exc import SAWarning
warnings.filterwarnings(action='ignore', category=SAWarning,
                        message='.*Thumbnail.source')
warnings.filterwarnings(action='ignore', category=SAWarning,
                        message='.*Source.thumbnails')

# disable annoying tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from .alert import *
from .archive import *
from .bookkeeping import *
from .catalog import *
from .coadd import *
from .constants import *
from .core import *
from .crossmatch import *
from .detections import *
from .external import *
from .file import *
from .filterobjects import *
from .fitsfile import *
from .hotpants import *
from .image import *
from .mask import *
from .mpi import *
from .photometry import *
from .plotting import *
from .secrets import *
from .seeing import *
from .send import *
from .sextractor import *

from .thumbnails import *
from .source import *
from .spatial import *
from .subtraction import *
from .swarp import *
from .utils import *



# this one needs to be last
from .joins import *
