# ZUDS Image Processing Pipeline [WIP]

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

To install the package you can simply do 

    pip install zuds
    
However, there are some external libraries you need to build first. The sections below present some methods for how you can install everything needed to run the pipeline. 

Prerequisites:

  * [SExtractor >= 2.25.0](https://github.com/astromatic/sextractor)
  * [SWarp >= 2.38.0](https://github.com/astromatic/swarp)
  * [postgresql >= 10](https://www.postgresql.org/)
  * [cfitsio](https://heasarc.gsfc.nasa.gov/fitsio/)
  * [hotpants >= 5.1.11](https://github.com/zuds-survey/hotpants)


### Recommended method (`conda`)

This approach requires the `conda` executable to be installed on your system and used to manage your python and python packages. 

Clone this repository, then run 

    bash build.conda.sh
    
to install `zuds`, as well as:

  * [SExtractor >= 2.25.0](https://github.com/astromatic/sextractor)
  * [SWarp >= 2.38.0](https://github.com/astromatic/swarp)
  * [postgresql >= 10](https://www.postgresql.org/)
  * [cfitsio](https://heasarc.gsfc.nasa.gov/fitsio/)
  
After you have completed this step, to install hotpants, cd into the `hotpants` directory and type `make`. Then copy the `hotpants` executable to your `PATH`. It should then be available to the `zuds` library. 


### Via docker-compose

A complete setup of the pipeline+database is available via `docker-compose`. Clone this repository, then run `docker-compose up`. A container running a jupyter notebook with the `zuds` pipeline and all dependencies installed should spin up, as well as a separate container for the database. Navigate to `localhost:8174` and open up `demo/demo.ipynb` to start running a demo of the pipeline in a jupyter notebook. 

Note: This approach is not recommended for Mac OSX users due to [resource issues with docker on Mac](https://github.com/docker/for-mac/issues/178). 

### Roll your own

You can download and build all of the prerequisite packages manually, then do `pip install zuds` to install the pipeline.
