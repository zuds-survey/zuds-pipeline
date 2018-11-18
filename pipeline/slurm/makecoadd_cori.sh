#!/bin/bash
#SBATCH -N 1
#SBATCH -J $5
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -A ***REMOVED***
#SBATCH --mail-type=ALL
#SBATCH --partition=realtime
#SBATCH --mail-user=dgold@berkeley.edu
#SBATCH --image=registry.services.nersc.gov/dgold/improc:latest
#SBATCH --dependency=afterok:DLIST
#SBATCH -C haswell
#SBATCH --exclusive
#SBATCH --volume=/global/homes/d/dgold:/home/desi
#SBATCH -o $4/slurm-%A.out

frames="$1"
cats="$2"
obase="$3"

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

cd /global/cscratch1/sd/dgold/ztfcoadd/job_scripts

shifter python /pipeline/bin/makecoadd.py --input-frames ${frames} --input-catalogs ${cats} \
               --output-basename ${obase}
