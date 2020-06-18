<h1 align="center">
  <br>
  <img
    src="https://avatars2.githubusercontent.com/u/63957543?s=400&u=ebbfb09abc72ec77cf865a17d13918231985c236&v=4"
    alt="ZUDS Logo"
    width="100px"
  />
  <br>
  The ZUDS Pipeline
  <br>
</h1>

<h2 align="center">
A High-Performance Image Processing Pipeline for ZTF
</h2>

![Build](https://github.com/zuds-survey/zuds-pipeline/workflows/build-and-test/badge.svg)

Current capabilities include:

  * Image coaddition
  * Image subbtraction
  * Multi-epoch subtractions
  * Image alignment and registration
  * Forced photometry 
  * Source detection and machine learning
  * Record keeping (via postgres)
  * Alert generation
  * Image and catalog display
  
## Quick start & tutorial
```bash
$ docker-compose up
```

In the browser, navigate to `localhost:8174`. Open 
`/zuds/demo.ipynb`. Execute the cells to run the pipeline tutorial.
  
## Installation

To install the zuds Python library, do

    pip install zuds
    

`zuds` requires that the following executables be on your path:

  * [SExtractor >= 2.25.0](https://github.com/astromatic/sextractor)
  * [SWarp >= 2.38.0](https://github.com/astromatic/swarp)
  * [psql >= 9.6](https://www.postgresql.org/)
  * [hotpants >= 5.1.11](https://github.com/zuds-survey/hotpants)

