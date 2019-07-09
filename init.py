import os
import yaml
from argparse import ArgumentParser
from pathlib import Path
import shutil

#########################################################################
# Modify these variables to point to the right values for you.
# Then execute this file to propagate these values forward to the pipeline
#########################################################################

nersc_account = '***REMOVED***'
nersc_username = 'ztfproc'
nersc_host = 'cori.nersc.gov'
nersc_password = '***REMOVED***'
nersc_xfer_username = 'dgold'

lensgrinder_home = '/global/cscratch1/sd/ztfproc/lensgrinder'
run_topdirectory = '/global/cscratch1/sd/ztfproc/coadd'
staging_topdir = '/global/cscratch1/sd/ztfproc/staging'

hpss_dbhost = '***REMOVED***'
hpss_dbport = 6666
hpss_dbusername = '***REMOVED***'
hpss_dbname = 'ztfimages'
hpss_dbpassword = '***REMOVED***'

ipac_username = 'dgold@berkeley.edu'
ipac_password = '***REMOVED***'

skyportal_dbhost = hpss_dbhost
skyportal_dbport = 7777
skyportal_dbusername = 'skyportal'
skyportal_dbpassword = '***REMOVED***'
skyportal_dbname = 'skyportal'
skyportal_home = '/global/cscratch1/sd/dgold/skyportal_private'

hits_dbhost = '***REMOVED***'
hits_dbport = 5432
hits_dbusername = '***REMOVED***'
hits_dbpassword = '***REMOVED***'
hits_dbname = 'iptf'

shifter_image = 'registry.services.nersc.gov/dgold/improc:latest'
slurm_email = 'ztfcoadd@gmail.com'

staging_directory = f'{Path(staging_topdir) / "xfer/data" / os.getenv("HOSTNAME")}'
staging_cmddir = f'{Path(staging_topdir) / "cmdlist"}'

ssh_keyfile = f'{Path(os.getenv("HOME")) / ".ssh/id_rsa"}'


#########################################################################
# Don't change the values of anything after this line
#########################################################################

volume_mounts = {
    os.path.join(lensgrinder_home, 'pipeline'): '/pipeline',
    f'/global/homes/{nersc_username[0].lower()}/{nersc_username}': '/home/desi',
    os.path.join(lensgrinder_home, 'pipeline', 'astromatic'): '/astromatic',
    lensgrinder_home: '/lg',
    skyportal_home: '/skyportal'
}

vstring = ';'.join([f'{k}:{volume_mounts[k]}' for k in volume_mounts])

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
    'SKYPORTAL_HOME': skyportal_home,
    'NERSC_USERNAME': nersc_username,
    'NERSC_PASSWORD': nersc_password,
    'NERSC_HOST': nersc_host,
    'NERSC_ACCOUNT': nersc_account,
    'NERSC_XFER_USERNAME': nersc_xfer_username,
    'LENSGRINDER_HOME': lensgrinder_home,
    'VOLUMES': vstring,
    'SHIFTER_IMAGE': shifter_image,
    'OUTPUT_DIRECTORY': run_topdirectory,
    'COADDSUB_EXEC': os.path.join(lensgrinder_home, 'slurm', 'single_coaddsub.sh'),
    'IPAC_USERNAME': ipac_username,
    'IPAC_PASSWORD': ipac_password,
    'STAGING_DIRECTORY': staging_directory,
    'STAGING_CMDDIR': staging_cmddir,
    'SSH_KEYFILE': ssh_keyfile,
    'HITS_DBHOST': hits_dbhost,
    'HITS_DBPORT':hits_dbport,
    'HITS_DBPASSWORD': hits_dbpassword,
    'HITS_DBUSERNAME': hits_dbusername,
    'HITS_DBNAME': hits_dbname
}

estring = ' '.join([f" -e {k}='{environment_variables[k]}'" for k in environment_variables])



with open('shifter.sh', 'w') as f:
    f.write(f'''#!/bin/bash
shifter --volume="{vstring}" \
        --image={shifter_image} \
        {estring} /bin/bash
''')


with open('interactive.sh', 'w') as f:
    f.write(f'''salloc -N 1 -t 00:30:00 -L SCRATCH -A {nersc_account} \
--partition=realtime --image={shifter_image} -C haswell --exclusive \
--volume="{vstring}"''')

with open('env.sh', 'w') as f:
    f.write(f'''#!/bin/bash
        {estring.replace('-e', '')} /bin/bash''')

