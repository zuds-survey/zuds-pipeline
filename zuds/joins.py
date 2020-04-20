from .core import join_model, Base
from .subtraction import SingleEpochSubtraction, MultiEpochSubtraction
from .bookkeeping import Job
from .image import CalibratableImage
from .coadd import Coadd


JobImage = join_model('job_images', Job, CalibratableImage, base=Base)
StackedSubtractionFrame = join_model('stackedsubtraction_frames',
                                     MultiEpochSubtraction,
                                     SingleEpochSubtraction,
                                     base=Base)

CoaddImage = join_model('coadd_images', Coadd, CalibratableImage, base=Base)
