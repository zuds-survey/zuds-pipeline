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
#SBATCH --volume=/global/homes/d/dgold:/home/desi
#SBATCH -o $4/slurm-%A.out

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

news="$1"
cats="$2"
obase="$3"
template="$6"

cd /global/cscratch1/sd/dgold/ztfcoadd/job_scripts

shifter python /pipeline/bin/makecoadd.py --input-frames ${news} --input-catalogs ${cats} \
               --output-basename=${obase}

shifter python /pipeline/bin/makesub.py --science-frames ${obase}.fits \
               --templates ${template}
