# ZUDS Image Processing Pipeline

An object-oriented pipeline for ZTF image processing. Current capabilities include:

  * Image coaddition
  * Image subbtraction
  * Multi-epoch subtractions
  * Image alignment and registration
  * Forced photometry 
  * Source detection and machine learning
  * Record keeping (via postgres)
  * Alert generation
  * Image and catalog display
  
## Installation

### Prerequisites

The pipeline requires that you install the following on your own:
 
  * [SExtractor >= 2.19.5](https://github.com/astromatic/sextractor)
  * [SWarp >= 2.38.0](https://github.com/astromatic/swarp)
  * [hotpants >= 5.1.11](https://github.com/zuds-survey/hotpants)
  * [postgresql >= 10](https://www.postgresql.org/)
  * [cfitsio](https://heasarc.gsfc.nasa.gov/fitsio/)
  
If you have `conda` installed on your machine, everything above but hotpants can be installed by executing

     bash build.conda.sh
     
from within this repository. This should work on OSX, Linux, and Windows. 

To build hotpants, first install the other dependencies above. Clone the zuds-survey hotpants repository linked above, cd into the directory, and type `make`. It should build an executable called `hotpants`. Copy that executable into your `PATH`, and you should be good to go.

### Installing the python package

Once you have the prerequisites, do 

    pip install zuds
    
to install the python package. `build.conda.sh` will do this for you, if you use it.



