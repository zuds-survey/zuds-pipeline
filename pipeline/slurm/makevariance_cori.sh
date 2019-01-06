#!/bin/bash
#SBATCH -N 1
#SBATCH -J $4
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -A ***REMOVED***
#SBATCH --mail-type=ALL
#SBATCH --partition=realtime
#SBATCH --mail-user=ztfcoadd@gmail.com
#SBATCH --image=registry.services.nersc.gov/dgold/improc:latest
#SBATCH --exclusive
#SBATCH -C haswell
#SBATCH --volume=/global/homes/d/dgold:/home/desi;/global/cscratch1/sd/dgold/lensgrinder/pipeline:/pipeline;/global/cscratch1/sd/dgold/lensgrinder/pipeline/astromatic:/pipeline/config
#SBATCH -o $3/slurm-%A.out

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

news="$1"
masks="$2"
srun -n 64 shifter python /pipeline/bin/makevariance.py --input-frames ${news} --input-masks ${masks}

