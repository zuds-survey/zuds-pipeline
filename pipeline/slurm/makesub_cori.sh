#!/bin/bash
#SBATCH -N 1
#SBATCH -J stckdiff
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -A ***REMOVED***
#SBATCH --mail-type=ALL
#SBATCH --partition=realtime
#SBATCH --mail-user=dgold@berkeley.edu
#SBATCH --image=registry.services.nersc.gov/dgold/improc:latest
#SBATCH --dependency=afterok:{dlist:s}
#SBATCH --exclusive
#SBATCH -C haswell
#SBATCH --volume=/global/homes/d/dgold:/home/desi

news="$1"
template="$2"


srun -n 64 shifter python /pipeline/bin/makesub.py --science-images=${news} --templates=${refs}
