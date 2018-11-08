#!/bin/bash
#SBATCH -N 1
#SBATCH -J diffem
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -A ***REMOVED***
#SBATCH --mail-type=ALL
#SBATCH --partition=realtime
#SBATCH --mail-user=dgold@berkeley.edu
#SBATCH --image=registry.services.nersc.gov/dgold/improc:latest
#SBATCH --dependency=afterok:{dlist:s}

frames="$1"
cats="$2"
obase="$3"

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

shifter python /lensgrinder/pipeline/bin/makecoadd.py --input-frames=${frames} --input-catalogs=${cats} \
               --output-basename=${obase}
