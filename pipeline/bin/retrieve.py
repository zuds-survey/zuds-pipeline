import os
import psycopg2
import pandas as pd
from argparse import ArgumentParser
from subprocess import check_call
import tempfile, io
import paramiko
from pathlib import Path


class HPSSDB(object):

    def __init__(self):

        dbname = os.getenv('HPSS_DBNAME')
        password = os.getenv('HPSS_DBPASSWORD')
        username = os.getenv('HPSS_DBUSERNAME')
        port = os.getenv('HPSS_DBPORT')
        host = os.getenv('HPSS_DBHOST')
        dsn = f'host={host} user={username} password={password} dbname={dbname} port={port}'

        self.connection = psycopg2.connect(dsn)
        self.cursor = self.connection.cursor()

    def __del__(self):

        del self.cursor
        del self.connection


def submit_hpss_job(tarfiles, rmimages, job_script_destination, frame_destination, tape_number):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    nersc_username = os.getenv('NERSC_USERNAME')
    nersc_password = os.getenv('NERSC_PASSWORD')
    nersc_host = os.getenv('NERSC_HOST')
    nersc_account = os.getenv('NERSC_ACCOUNT')

    ssh_client.connect(hostname=nersc_host, username=nersc_username, password=nersc_password)

    if job_script_destination is None:
        # then just use temporary files

        jobscript = tempfile.NamedTemporaryFile()
        subscript = tempfile.NamedTemporaryFile()

    else:

        jobscript = open(Path(job_script_destination) / f'hpss.{tape_number}.sh', 'w')
        subscript = open(Path(job_script_destination) / f'hpss.{tape_number}.sub.sh', 'w')

    subscript.write(f'''#!/usr/bin/env bash
module load esslurm
sbatch {jobscript.name}
''')

    jobstr = f'''#!/usr/bin/env bash
#SBATCH -J {tape_number}
#SBATCH -L SCRATCH,project
#SBATCH -q xfer
#SBATCH -N 1
#SBATCH -A {nersc_account}
#SBATCH -t 48:00:00
#SBATCH -C haswell

cd {frame_destination}

'''

    for tarfile in tarfiles:
        directive = f'''
hsi get {tarfile}
tar xvf --strip-components=12 {os.path.basename(tarfile)}
rm {os.path.basename(tarfile)}

'''
        jobstr += directive

    for image in rmimages:
        jobstr += f'rm {image}\n'

    jobscript.write(jobstr)

    stdin, stdout, stderr = ssh_client.exec_command(f'/bin/bash {subscript.name}')
    out = stdout.readlines()
    err = stderr.readlines()

    print(out, flush=True)
    print(err, flush=True)

    jobid = err.split()[-1]

    ssh_client.close()
    jobscript.close()
    subscript.close()

    return jobid


def retrieve_images(whereclause, exclude_masks=False, job_script_destination=None, frame_destination='.'):

    # interface to HPSS and database
    hpssdb = HPSSDB()

    # this is the query to get the image paths
    query = f'SELECT PATH, HPSS_SCI_PATH, HPSS_MASK_PATH FROM IMAGE WHERE HPSS_SCI_PATH IS NOT NULL ' \
            f'AND {whereclause}'
    hpssdb.cursor.execute(query)
    results = hpssdb.cursor.fetchall()

    df = pd.DataFrame(results, columns=['path', 'hpss_sci_path', 'hpss_mask_path'])

    dfsci = df[['path', 'hpss_sci_path']]
    dfsci = dfsci.rename({'hpss_sci_path': 'tarpath'}, axis='columns')

    if not exclude_masks:
        dfmask = df[['path', 'hpss_mask_path']]
        dfmask = dfmask.rename({'hpss_mask_path': 'tarpath'}, axis='columns')
        dfmask.dropna(inplace=True)
        df = pd.concat((dfsci, dfmask))
    else:
        df = dfsci

    tars = df['tarpath'].unique()
    instr = '\n'.join(tars.tolist())

    with tempfile.NamedTemporaryFile() as f:
        f.write(f"{instr}\n".encode('ASCII'))

        # rewind the file
        f.seek(0)

        # sort tarball retrieval by location on tape
        sortexec = Path(os.getenv('LENSGRINDER_HOME')) / 'pipeline/bin/hpsssort.sh'

        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        nersc_username = os.getenv('NERSC_USERNAME')
        nersc_password = os.getenv('NERSC_PASSWORD')
        nersc_host = os.getenv('NERSC_HOST')

        ssh_client.connect(hostname=nersc_host, username=nersc_username, password=nersc_password)

        syscall = f'bash {sortexec} {f.name}'
        _, stdout, _ = ssh_client.connect(syscall)

        # read it into pandas
        ordered = pd.read_csv(stdout, delim_whitespace=True, names=['tape', 'position', '_', 'hpsspath'])

    # submit the jobs based on which tape the tar files reside on
    # and in what order they are on the tape

    dependency_dict = {}
    for tape, group in ordered.groupby('tape'):

        # get the tarfiles
        tarnames = group['hpsspath'].tolist()

        for tarname in tarnames:
            query = f'SELECT PATH FROM IMAGE WHERE HPSS_SCI_PATH={tarname} OR HPSS_MASK_PATH={tarname}'
            hpssdb.cursor.execute(query)
            allims = [t[0] for t in hpssdb.cursor.fetchall()]

        rmimages = [im for im in allims if im not in df['path']]

        jobid = submit_hpss_job(group, rmimages, job_script_destination, frame_destination, tape)
        for image in df['path']:
            dependency_dict[image] = jobid

    del hpssdb
    return dependency_dict


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("whereclause", required=True, default=None, type=str,
                        help='SQL where clause that tells the program which images to retrieve.')
    parser.add_argument('--exclude-masks', default=False, action='store_true',
                        help='Only retrieve the science images.')
    args = parser.parse_args()
    retrieve_images(args.whereclause, exclude_masks=args.exclude_masks)
