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
#SBATCH --exclusive
#SBATCH -C haswell
#SBATCH --volume/global/homes/d/dgold:/home/desi

news="$1"
masks="$2"
srun -n 64 shifter python /lensgrinder/pipeline/bin/makevariance.py --input-frames=${news} --mask-frames=${masks}

