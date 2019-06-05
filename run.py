import os
import yaml
from argparse import ArgumentParser

#########################################################################
# Modify these variables to point to the right values for you.
# Then execute this file to propagate these values forward to the pipeline
#########################################################################

nersc_account = '***REMOVED***'
nersc_username = 'dgold'
nersc_host = 'cori.nersc.gov'
nersc_password = '***REMOVED***'

lensgrinder_home = '/global/cscratch1/sd/dgold/lensgrinder'
run_topdirectory = '/global/cscratch1/sd/dgold/coadd'

hpss_dbhost = '***REMOVED***'
hpss_dbport = 6666
hpss_dbusername = '***REMOVED***'
hpss_dbname = 'ztfimages'
hpss_dbpassword = '***REMOVED***'

skyportal_dbhost = hpss_dbhost
skyportal_dbport = 7777
skyportal_dbusername = 'skyportal'
skyportal_dbpassword = '***REMOVED***'
skyportal_dbname = 'skyportal'

shifter_image = 'registry.services.nersc.gov/dgold/improc:latest'
slurm_email = 'ztfcoadd@gmail.com'


#########################################################################
# Don't change the values of anything after this line
#########################################################################

volume_mounts = {
    os.path.join(lensgrinder_home, 'pipeline'): '/pipeline',
    os.path.join(run_topdirectory, 'output'): '/output',
    f'/global/homes/{nersc_username[0].lower()}/{nersc_username}': '/home/desi',
    os.path.join(run_topdirectory, 'job_scripts'): '/job_scripts',
    os.path.join(lensgrinder_home, 'pipeline', 'astromatic'): '/astromatic',
    lensgrinder_home: '/lg'
}

logdir = os.path.join(run_topdirectory, 'logs')

environment_variables = {
    'HPSS_DBHOST': hpss_dbhost,
    'HPSS_DBPORT': hpss_dbport,
    'HPSS_DBUSERNAME': hpss_dbusername,
    'HPSS_DBPASSWORD': hpss_dbpassword,
    'HPSS_DBNAME': hpss_dbname,
    'SKYPORTAL_DBHOST': skyportal_dbhost,
    'SKYPORTAL_DBPORT': skyportal_dbport,
    'SKYPORTAL_DBUSERNAME': skyportal_dbusername,
    'SKYPORTAL_DBPASSWORD': skyportal_dbpassword,
    'SKYPORTAL_DBNAME': skyportal_dbname,
    'NERSC_USERNAME': nersc_username,
    'NERSC_PASSWORD': nersc_password,
    'NERSC_HOST': nersc_host
}
estring = ' '.join([f" -e {k}='{environment_variables[k]}'" for k in environment_variables])
vstring = ';'.join([f'{k}:{volume_mounts[k]}' for k in volume_mounts])

with open('shifter.sh', 'w') as f:
    f.write(f'''#!/bin/bash
shifter --volume="{vstring}" \
        --image={shifter_image} \
        {estring} /bin/bash
''')


with open('retrieve_hpss.sh', 'w') as f:
    f.write(f'''#!/usr/bin/env bash
#SBATCH -J $1
#SBATCH -L SCRATCH
#SBATCH -q xfer
#SBATCH -N 1
#SBATCH -A {nersc_account}
#SBATCH -t 48:00:00
#SBATCH -C haswell

start=`date +%s`
/usr/common/mss/bin/htar xvf  -L /global/cscratch1/sd/dgold/ztf_xfer_jobscripts/rank02_chunk1624_hpjob0.cmd
for f in ``; do
    rm $f
done

end=`date +%s`

runtime=$((end-start))

echo runtime was $runtime


''')

with open('process_focalplane.sh', 'w') as f:
    f.write(f'''#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -J $1
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -A {nersc_account}
#SBATCH --mail-type=ALL
#SBATCH --partition=realtime
#SBATCH --mail-user={slurm_email}
#SBATCH --image={shifter_image}
#SBATCH -C haswell
#SBATCH --exclusive
#SBATCH --volume="{vstring}"
#SBATCH -o {os.path.join(logdir, '$1.out')}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

start=`date +%s`

# split the exposure into individual fitsfiles
shifter python /pipeline/bin/split_exposure.py /output/$1

# run preprocess and get the good wcs calibrations
srun -n 64 shifter {estring} python /pipeline/bin/preprocess.py /output/$2/$1.list

# determine all of the images that overlap with all of the chip images
if ! srun -n 64 shifter {estring} python /pipeline/bin/make_base_template.py /output/$2/$1.list; then
    echo "image $1 has no template coverage, so cannot be processed. exiting..."
    exit 1 
fi

# now actually run the pipeline
srun -n 64 shifter {estring} python /process/process.py /output/$2/$1.list 

end=`date +%s`

runtime=$((end-start))

echo runtime was $runtime

''')


with open('interactive.sh', 'w') as f:
    f.write(f'''salloc -N 1 -t 00:30:00 -L SCRATCH -A {nersc_account} \
--partition=realtime --image={shifter_image} -C haswell --exclusive \
--volume="{vstring}"''')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('task', help='Path to yaml file describing the workflow.')
    args = parser.parse_args()

    task_file = args.task
    task_spec = yaml.load(open(task_file, 'r'))

    if 'hpss' in task_spec:
        from retrieve import retrieve_images
        whereclause = task_spec['hpss']['whereclause']
        exclude_masks = task_spec['hpss']['exclude_masks']
        hpss_jobids = retrieve_images(whereclause, exclude_masks=exclude_masks)

