#!/bin/bash
#SBATCH -N 1
#SBATCH -J $5
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -A ***REMOVED***
#SBATCH --mail-type=ALL
#SBATCH --partition=realtime
#SBATCH --mail-user=ztfcoadd@gmail.com
#SBATCH --image=registry.services.nersc.gov/dgold/improc:latest
#SBATCH --dependency=afterok:DLIST
#SBATCH -C haswell
#SBATCH --exclusive
#SBATCH --volume=/global/homes/d/dgold:/home/desi;/global/cscratch1/sd/dgold/lensgrinder/pipeline/astromatic:/config
#SBATCH -o $4/slurm-%A.out

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

cd /global/cscratch1/sd/dgold/ztfcoadd/job_scripts

# job script lines go here
