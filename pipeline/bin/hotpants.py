import os
import db
import numpy as np
import shutil
import pandas as pd

from utils import initialize_directory
from seeing import estimate_seeing


import tempfile
from pathlib import Path
import paramiko

# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


def chunk(iterable, chunksize):
    isize = len(iterable)
    nchunks = isize // chunksize if isize % chunksize == 0 else isize // chunksize + 1
    for i in range(nchunks):
        yield i, iterable[i * chunksize : (i + 1) * chunksize]


def submit_coaddsub(template_dependencies, variance_dependencies, science_metatable, ref,
                    rolling=False, coadd_windowsize=3, batch_size=32, job_script_destination='.',
                    log_destination='.', frame_destination='.', task_name=None):


    log_destination = Path(log_destination)
    frame_destination = Path(frame_destination)
    job_script_destination = Path(job_script_destination)
    job_list = []

    nersc_username = os.getenv('NERSC_USERNAME')
    nersc_password = os.getenv('NERSC_PASSWORD')
    nersc_host = os.getenv('NERSC_HOST')
    nersc_account = os.getenv('NERSC_ACCOUNT')
    shifter_image = os.getenv('SHIFTER_IMAGE')
    volumes = os.getenv('VOLUMES')
    coaddsub_exec = os.getenv('COADDSUB_EXEC')

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=nersc_host, username=nersc_username, password=nersc_password)

    dependency_dict = {}

    for (field, quadrant, band, ccdnum), group in science_metatable.groupby(['field',
                                                                             'qid',
                                                                             'filtercode',
                                                                             'ccdid']):
        pass

    science_rows = group.copy()
    science_rows = science_rows.sort_values('obsjd')

    if len(template_dependencies) > 0:
        template_dependency_list = [template_dependencies[ref.disk_path]]
    else:
        template_dependency_list = []

    if coadd_windowsize > 0:
        bin_edges = make_coadd_bins(science_rows, window_size=coadd_windowsize, rolling=rolling)
        for l, r in bin_edges:
            frames_c1 = science_rows['obsdate'] >= l
            frames_c2 = science_rows['obsdate'] < r
            frames = science_rows[frames_c1 & frames_c2]

            if len(frames) > 3:
                seeing = frames['seeing']
                med = np.median(seeing)
                std = 1.4826 * np.median(np.abs(seeing - med))

                frames = frames[seeing < med + 2 * std]

            if len(frames) == 0:
                continue

            if len(variance_dependencies) > 0:
                variance_dependency_list = list(set([variance_dependencies[frame] for frame in frames['path']]))
            else:
                variance_dependency_list = []


            cdep_list = variance_dependency_list + template_dependency_list

            lstr = f'{l}'.split()[0].replace('-', '')
            rstr = f'{r}'.split()[0].replace('-', '')

            coadd_base = f'{field:06d}_c{ccdnum:02d}_{quadrant:d}_{band:s}_{lstr}_{rstr}_coadd'
            coadd_dir = frame_destination / Path(frames['path'].iloc[0]).parent / coadd_base
            coadd_dir.mkdir(parents=True, exist_ok=True)

            coadd_name = coadd_dir / f'{coadd_base}.fits'

            # log the stack to the database
            stack = db.Stack(disk_path=f'{coadd_name}', field=int(field), ccdid=int(ccdnum), qid=int(quadrant),
                             filtercode=band)

            db.DBSession().add(stack)
            db.DBSession().commit()

            for imid in frames['id']:
                stackimage = db.StackImage(imag_id=int(imid), stack_id=stack.id)
                db.DBSession().add(stackimage)
            db.DBSession().commit()

            framepaths_in = [(frame_destination / frame).resolve() for frame in frames['path']]
            framepaths_out = [(coadd_dir / os.path.basename(frame)) for frame in frames['path']]

            for i, o in zip(framepaths_in, framepaths_out):
                shutil.copy(i, o)

            mesub = db.MultiEpochSubtraction(stack=stack, reference=ref,
                                             disk_path=f'{coadd_dir / sub_name(stack.disk_path, ref.disk_path)}',
                                             qid=int(quadrant), ccdid=int(ccdnum), field=int(field),
                                             filtercode=band)
            db.DBSession().add(mesub)
            db.DBSession().commit()

            job = {'type':'coaddsub',
                   'frames': [f'{framepath}' for framepath in framepaths_out],
                   'template': ref.disk_path, 'coadd_name': coadd_name,
                   'sub': mesub,
                   'dependencies': cdep_list}

            job_list.append(job)

    else:
        # just run straight up subtractions
        for i, row in science_rows.iterrows():
            if len(variance_dependencies) > 0:
                variance_dependency_list = [variance_dependencies[row['path']]]
            else:
                variance_dependency_list = []
            cdep_list = variance_dependency_list + template_dependency_list

            image = db.DBSession().query(db.Image).get(int(row['id']))
            sesub = db.SingleEpochSubtraction(image=image, reference=ref,
                                              disk_path=sub_name(row['full_path'], ref.disk_path),
                                              qid=int(quadrant), ccdid=int(ccdnum), field=int(field),
                                              filtercode=band)

            db.DBSession().add(sesub)
            db.DBSession().commit()

            job = {'type': 'sub', 'frame': f"{(frame_destination / row['path']).resolve()}",
                   'template': ref.disk_path, 'dependencies': cdep_list, 'sub': sesub}

            job_list.append(job)

    for i, ch in chunk(job_list, batch_size):

        my_deps = []
        for j in ch:
            my_deps += j['dependencies']
        my_deps = ':'.join(list(map(str, set(my_deps))))

        job_name = f'sub.{task_name}.{field}.{ccdnum}.{quadrant}.{band}.bin{coadd_windowsize}.{i}'

        jobstr = f'''#!/bin/bash
#SBATCH -N 1
#SBATCH -J {job_name}
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -A {nersc_account}
#SBATCH --partition=realtime
#SBATCH --image={shifter_image}
#SBATCH --exclusive
#SBATCH -C haswell
#SBATCH --volume="{volumes}"
#SBATCH -o {log_destination.resolve()}/{job_name}.out
#SBATCH --dependency=afterok:{my_deps}

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

'''

        estring = os.getenv("ESTRING").replace(r"\x27", "'")

        if len(my_deps) == 0:
            jobstr = jobstr.replace('#SBATCH --dependency=afterok:\n', '')

        for j in ch:

            template = j['template']

            if j['type'] == 'coaddsub':

                frames = j['frames']
                cats = [frame.replace('.fits', '.cat') for frame in frames]
                coadd = j['coadd_name']
                execstr = f'shifter {estring} bash {coaddsub_exec} \"{" ".join(frames)}\" \"{" ".join(cats)}\"' \
                          f' \"{coadd}\" \"{template}\" \"{j["sub"].id}\" \"{j["sub"].stack.id}\"' \
                          f' \"{j["sub"].disk_path}\"&\n'
            else:

                frame = j['frame']
                execstr = f'shifter {estring} python {os.getenv("LENSGRINDER_HOME")}/pipeline/bin/makesub.py ' \
                          f'--science-frames {frame} --templates {template}  --no-publish &&' \
                          f'shifter {estring} python  {os.getenv("LENSGRINDER_HOME")}/pipeline/bin/log_image.py ' \
                          f'{j["sub"].disk_path} {j["sub"].id} SingleEpochSubtraction &\n'


            jobstr += execstr
        jobstr += 'wait\n'

        if job_script_destination is None:
            job_script = tempfile.NamedTemporaryFile()
            jobstr = jobstr.encode('ASCII')
        else:
            job_script = open(job_script_destination.resolve() / f'sub.{task_name}.bin{coadd_windowsize}.{i}.sh', 'w')

        job_script.write(jobstr)
        job_script.seek(0)

        syscall = f'sbatch {job_script.name}'
        stdin, stdout, stderr = ssh_client.exec_command(syscall)

        out = stdout.read()
        err = stderr.read()

        jobid = int(out.strip().split()[-1])

        for j in ch:
            dependency_dict[j['sub'].id] = jobid

        print(out, flush=True)
        print(err, flush=True)

        retcode = stdout.channel.recv_exit_status()
        if retcode != 0:
            raise RuntimeError(f'Unable to submit job with script: "{jobstr}", nonzero retcode')

        job_script.close()

    return dependency_dict


def make_coadd_bins(science_rows, window_size=3, rolling=False):

    mindate = pd.to_datetime(science_rows['obsdate'].min())
    maxdate = pd.to_datetime(science_rows['obsdate'].max())


    if rolling:
        dates = pd.date_range(mindate, maxdate, freq='1D')
        bins = []
        for i, date in enumerate(dates):
            if i + window_size >= len(dates):
                break
            bins.append((date, dates[i + window_size]))

    else:
        binedges = pd.date_range(mindate, maxdate, freq=f'{window_size}D')

        bins = []
        for i, lbin in enumerate(binedges[:-1]):
            bins.append((lbin, binedges[i + 1]))

    return bins


def prepare_hotpants(sci, ref, outname, submask, directory,
                     copy_inputs=False):

    initialize_directory(directory)

    # if requested, copy the input images to a temporary working directory
    if copy_inputs:
        impaths = []
        for image in [sci, ref]:
            shutil.copy(image.local_path, directory)
            impaths.append(str(directory / image.basename))
    else:
        impaths = [im.local_path for im in [sci, ref]]
    scipath, refpath = impaths

    if 'SEEING' not in sci.header:
        estimate_seeing(sci)
        sci.save()

    seepix = sci.header['SEEING']  # header seeing is FWHM in pixels
    r = 2.5 * seepix
    rss = 6. * seepix

    nsx = sci.header['NAXIS1'] / 100.
    nsy = sci.header['NAXIS2'] / 100.

    # get the background for the input images
    scirms = sci.rms_image
    refrms = ref.rms_image

    # save temporary copies of rms images if necessary
    if not scirms.ismapped or copy_inputs:
        scirms_tmpnam = str((directory / scirms.basename).absolute())
        scirms.map_to_local_file(scirms_tmpnam)
        scirms.save()

    if not refrms.ismapped or copy_inputs:
        refrms_tmpnam = str((directory / refrms.basename).absolute())
        refrms.map_to_local_file(refrms_tmpnam)
        refrms.save()

    scibkg = np.median(sci.data)
    refbkg = np.median(ref.data)

    scibkgstd = np.median(sci.rms_image.data)
    refbkgstd = 1.4826 * np.median(np.abs(ref.data - np.median(ref.data)))

    il = scibkg - 10 * scibkgstd
    tl = refbkg - 10 * refbkgstd

    satlev = 5e4  # not perfect, but close enough.

    syscall = f'hotpants -inim {scipath} -hki -n i -c t ' \
              f'-tmplim {ref.local_path} -outim {outname} ' \
              f'-tu {satlev} -iu {satlev}  -tl {tl} -il {il} -r {r} ' \
              f'-rss {rss} -tni {refrms.local_path} ' \
              f'-ini {scirms.local_path} ' \
              f'-imi {submask.local_path} ' \
              f'-nsx {nsx} -nsy {nsy}'

    return syscall



