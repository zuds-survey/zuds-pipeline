import time
import datetime
import os
import pandas as pd
from pathlib import Path
import subprocess
import zuds
from zuds import get_secret
import requests
import io
import shutil
import sqlalchemy as sa
import numpy as np



def submit_hpss_job(tarfiles, images, job_script_destination,
                    frame_destination, log_destination,
                    tape_number, preserve_dirs):

    nersc_account = get_secret('nersc_account')

    cmdlist = open(Path(job_script_destination) /
                   f'hpss.{tape_number}.cmd.sh', 'w')
    jobscript = open(Path(job_script_destination) /
                     f'hpss.{tape_number}.sh', 'w')
    subscript = open(Path(job_script_destination) /
                     f'hpss.{tape_number}.sub.sh', 'w')


    substr =  f'''#!/usr/bin/env bash
module load esslurm
sbatch {Path(jobscript.name).resolve()}
'''

    subscript.write(substr)

    hpt = f'hpss.{tape_number}'

    jobstr = f'''#!/usr/bin/env bash
#SBATCH -J {tape_number}
#SBATCH -L SCRATCH,project
#SBATCH -q xfer
#SBATCH -N 1
#SBATCH -A {nersc_account}
#SBATCH -t 48:00:00
#SBATCH -C haswell
#SBATCH -o {(Path(log_destination) / hpt).resolve()}.out

bash {os.path.abspath(cmdlist.name)}

'''

    cmdstr = f'cd {frame_destination}\n'

    sc = 12 if not preserve_dirs else 8

    for tarfile, imlist in zip(tarfiles, images):
        wildimages = '\n'.join([f'*{p}' for p in imlist])

        directive = f'''
/usr/common/mss/bin/hsi get {tarfile}
echo "{wildimages}" | tar --strip-components={sc} -i --wildcards --wildcards-match-slash --files-from=- -xvf {os.path.basename(tarfile)}
rm {os.path.basename(tarfile)}

'''
        cmdstr += directive

    jobscript.write(jobstr)
    cmdlist.write(cmdstr)

    jobscript.close()
    subscript.close()
    cmdlist.close()

    command = f'/bin/bash {Path(subscript.name).resolve()}'
    p = subprocess.Popen(command.split(), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    #stdout, stderr = p.communicate()

    while True:
        if p.poll() is not None:
            break
        else:
            time.sleep(0.01)

    stdout, stderr = p.stdout, p.stderr
    retcode = p.returncode

    out = stdout.readlines()
    err = stderr.readlines()

    print(out, flush=True)
    print(err, flush=True)

    if retcode != 0:
        raise ValueError(f'Nonzero return code ({retcode}) on command, "{command}"')

    jobscript.close()
    subscript.close()

    jobid = int(out[0].strip().split()[-1])

    return jobid


def retrieve_images(images_or_ids,
                    job_script_destination='.',
                    frame_destination='.', log_destination='.',
                    preserve_dirs=False, n_jobs=14, tape=True,
                    http=True, ipac=True, archive_new=True):

    """Image whereclause should be a clause element on ZTFFile."""

    images_or_ids = np.atleast_1d(images_or_ids)
    ids = [int(i) if not hasattr(i, 'id') else i.id for i in images_or_ids]

    got = []

    if tape:
        jt = sa.join(zuds.ZTFFile, zuds.TapeCopy,
                     zuds.ZTFFile.id == zuds.TapeCopy.product_id)
        full_query = zuds.DBSession().query(
            zuds.ZTFFile, zuds.TapeCopy
        ).select_from(jt).outerjoin(
            zuds.HTTPArchiveCopy, zuds.ZTFFile.id == zuds.HTTPArchiveCopy.product_id
        ).filter(
            zuds.HTTPArchiveCopy.product_id == None
        )
        full_query = full_query.filter(zuds.ZTFFile.id.in_(ids),
                                       zuds.ZTFFile.fid != 3)  # dont use i-band from tape

        # this is the query to get the image paths
        metatable = pd.read_sql(full_query.statement, zuds.DBSession().get_bind())

        df = metatable[['basename', 'archive_id']]
        df = df.rename({'archive_id': 'tarpath'}, axis='columns')
        tars = df['tarpath'].unique()
        got.extend(metatable['product_id'].tolist())

        # sort tarball retrieval by location on tape
        t = datetime.datetime.utcnow().isoformat().replace(' ', '_')
        hpss_in = Path(job_script_destination) / f'hpss_{t}.in'
        hpss_out = Path(job_script_destination) / f'hpss_{t}.out'

        with open(hpss_in, 'w') as f:
            f.write("\n".join([f'ls -P {tar}' for tar in tars]))
            f.write("\n")  # always end with a \n

        syscall = f'/usr/common/mss/bin/hsi -O {hpss_out} in {hpss_in}'

        # for some reason hsi writes in >> mode, so need to delete the output
        # file if it exists to prevent it from mixing with results of a previous
        # run

        if hpss_out.exists():
            os.remove(hpss_out)

        p = subprocess.Popen(syscall.split(),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

        while True:
            if p.poll() is not None:
                break
            else:
                time.sleep(0.01)

        retcode = p.returncode
        stderr, stdout = p.stderr, p.stdout

        # 64 means some of the files didnt exist, that's ok
        if retcode not in [0, 64]:
            raise subprocess.CalledProcessError(stderr.read())

        # filter out the lines that dont start with FILE
        with open(hpss_out, 'r') as f:
            lines = [line for line in f.readlines() if line.startswith('FILE')]
        stream = io.StringIO(''.join(lines))

        # read it into pandas
        ordered = pd.read_csv(stream, delim_whitespace=True, names=['ignore-2',
                                                                    'hpsspath',
                                                                    'ignore-1',
                                                                    'ignore0',
                                                                    'position',
                                                                    'tape',
                                                                    'ignore1',
                                                                    'ignore2',
                                                                    'ignore3',
                                                                    'ignore4',
                                                                    'ignore5',
                                                                    'ignore6',
                                                                    'ignore7'])
        ordered['tape'] = [t[:-2] for t in ordered['tape']]
        ordered['position'] = [t.split('+')[0] for t in ordered['position']]
        ordered = ordered.sort_values(['tape', 'position'])

        for column in ordered.copy().columns:
            if column.startswith('ignore'):
                del ordered[column]

        # submit the jobs based on which tape the tar files reside on
        # and in what order they are on the tape

        dependency_dict = {}

        for tape, group in ordered.groupby(np.arange(len(ordered)) // (len(
                ordered) // n_jobs)):

            # get the tarfiles
            tarnames = group['hpsspath'].tolist()
            images = [df[df['tarpath'] == tarname]['basename'].tolist() for tarname
                      in tarnames]

            jobid = submit_hpss_job(tarnames, images, job_script_destination,
                                    frame_destination, log_destination, tape,
                                    preserve_dirs)

            for image in df[[name in tarnames for name in df['tarpath']]][
                'basename']:
                dependency_dict[image] = jobid

    if http:

        # now do the ones that are on disk
        jt = sa.join(zuds.ZTFFile, zuds.HTTPArchiveCopy,
                     zuds.ZTFFile.id == zuds.HTTPArchiveCopy.product_id)

        full_query = zuds.DBSession().query(
            zuds.ZTFFile, zuds.HTTPArchiveCopy
        ).select_from(jt)

        full_query = full_query.filter(zuds.ZTFFile.id.in_(ids))

        # this is the query to get the image paths
        metatable2 = pd.read_sql(full_query.statement, zuds.DBSession().get_bind())
        got.extend(metatable2['product_id'].tolist())

        # copy each image over
        for _, row in metatable2.iterrows():
            path = row['archive_path']
            if preserve_dirs:
                target = Path(frame_destination) / os.path.join(*path.split('/')[-5:])
            else:
                target = Path(frame_destination) / os.path.basename(path)

            if Path(target).absolute() == Path(path).absolute():
                # don't overwrite an already existing file
                continue

            target.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(path, target)

    # download the remaining images individually from IPAC

    if ipac:
        remaining = [int(i) for i in np.setdiff1d(ids, got)]
        remaining = zuds.DBSession().query(zuds.ZTFFile).filter(
            zuds.ZTFFile.id.in_(remaining)
        ).all()

        cookie = zuds.ipac_authenticate()

        for i in remaining:
            if preserve_dirs:
                destination = Path(frame_destination) / i.relname
            else:
                destination = Path(frame_destination) / i.basename

            suffix = 'mskimg.fits' if isinstance(i, zuds.MaskImage) else 'sciimg.fits'

            try:
                if isinstance(i, zuds.MaskImage):
                    i.parent_image.download(suffix=suffix, destination=destination, cookie=cookie)
                else:
                    i.download(suffix=suffix, destination=destination, cookie=cookie)
            except requests.RequestException:
                continue

            if archive_new:

                i.map_to_local_file(destination)

                # ensure the image header is written to the DB
                i.load_header()

                acopy = zuds.HTTPArchiveCopy.from_product(i, check=False)
                acopy.put()

                # and archive the file to disk
                zuds.DBSession().add(acopy)
                zuds.DBSession().commit()


