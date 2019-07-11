import os
import db
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from libztf import make_rms, cmbmask, execute, cmbrms
import uuid
import logging
import shutil
import pandas as pd

from galsim import des
import galsim

import tempfile
from pathlib import Path
import paramiko


def sub_name(frame, template):

    frame = f'{frame}'
    template = f'{template}'

    refp = os.path.basename(template)[:-5]
    newp = os.path.basename(frame)[:-5]

    outdir = os.path.dirname(frame)

    subp = '_'.join([newp, refp])

    sub = os.path.join(outdir, 'sub.%s.fits' % subp)
    return sub



from libztf.yao import yao_photometry_single

from filterobjects import filter_sexcat
from publish import load_catalog

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

            if len(frames) == 0:
                continue

            variance_dependency_list = list(set([variance_dependencies[frame] for frame in frames['path']]))
            cdep_list = variance_dependency_list + template_dependency_list

            lstr = f'{l}'.split()[0].replace('-', '')
            rstr = f'{r}'.split()[0].replace('-', '')

            coadd_name = frame_destination / f'{field:06d}_c{ccdnum:02d}_{quadrant:d}_' \
                                             f'{band:s}_{lstr}_{rstr}_coadd.fits'

            # log the stack to the database
            stack = db.Stack(disk_path=f'{coadd_name}', field=int(field), ccdid=int(ccdnum), qid=int(quadrant),
                             filtercode=band)
            db.DBSession().add(stack)
            db.DBSession().commit()

            for imid in frames['id']:
                stackimage = db.StackImage(imag_id=int(imid), stack_id=stack.id)
                db.DBSession().add(stackimage)
            db.DBSession().commit()

            framepaths = [(frame_destination / frame).resolve() for frame in frames['path']]

            mesub = db.MultiEpochSubtraction(stack=stack, reference=ref,
                                             disk_path=sub_name(stack.disk_path, ref.disk_path),
                                             qid=int(quadrant), ccdid=int(ccdnum), field=int(field),
                                             filtercode=band)
            db.DBSession().add(mesub)
            db.DBSession().commit()

            job = {'type':'coaddsub',
                   'frames': [f'{framepath}' for framepath in framepaths],
                   'template': ref.disk_path, 'coadd_name': coadd_name,
                   'sub': mesub,
                   'dependencies': cdep_list}

            job_list.append(job)

    else:
        # just run straight up subtractions
        for i, row in science_rows.iterrows():
            variance_dependency_list = [variance_dependencies[row['path']]]
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

        jobstr = f'''#!/bin/bash
#SBATCH -N 1
#SBATCH -J sub.{task_name}.{i}
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -A {nersc_account}
#SBATCH --partition=realtime
#SBATCH --image={shifter_image}
#SBATCH --exclusive
#SBATCH -C haswell
#SBATCH --volume="{volumes}"
#SBATCH -o {log_destination.resolve()}/sub.{task_name}.bin{coadd_windowsize}.{i}.out
#SBATCH --dependency=afterok:{my_deps}

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

'''

        for j in ch:

            template = j['template']

            if j['type'] == 'coaddsub':

                frames = j['frames']
                cats = [frame.replace('.fits', '.cat') for frame in frames]
                coadd = j['coadd_name']
                execstr = f'shifter bash {coaddsub_exec} \"{" ".join(frames)}\" \"{" ".join(cats)}\"' \
                          f'\"{coadd}\" \"{template}\" \"{j["sub"].id}\" \"{j["sub"].stack.id}\"' \
                          f' \"{j["sub"].disk_path}\"&\n'
            else:

                frame = j['frame']
                execstr = f'shifter python {os.getenv("LENSGRINDER_HOME")}/pipeline/bin/makesub.py ' \
                          f'--science-frames {frame} --templates {template} &&' \
                          f'shifter python  {os.getenv("LENSGRINDER_HOME")}/pipeline/bin/log_image.py ' \
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

        print(out, flush=True)
        print(err, flush=True)

        retcode = stdout.channel.recv_exit_status()
        if retcode != 0:
            raise RuntimeError(f'Unable to submit job with script: "{jobstr}", nonzero retcode')

        job_script.close()


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


def make_sub(myframes, mytemplates, publish=True):

    myframes = np.atleast_1d(myframes).tolist()
    mytemplates = np.atleast_1d(mytemplates).tolist()

    # now set up a few pointers to auxiliary files read by sextractor
    wd = os.path.dirname(__file__)
    confdir = os.path.join(wd, '..', 'astromatic', 'makesub')
    scampconfcat = os.path.join(confdir, 'scamp.conf.cat')
    defswarp = os.path.join(confdir, 'default.swarp')
    defsexref = os.path.join(confdir, 'default.sex.ref')
    defsexaper = os.path.join(confdir, 'default.sex.aper')
    defsexsub = os.path.join(confdir, 'default.sex.sub')
    defparref = os.path.join(confdir, 'default.param.ref')
    defparaper = os.path.join(confdir, 'default.param.aper')
    defparsub = os.path.join(confdir, 'default.param.sub')
    defconv = os.path.join(confdir, 'default.conv')
    defnnw = os.path.join(confdir, 'default.nnw')

    for frame, template in zip(myframes, mytemplates):

        # read some header keywords from the template
        with fits.open(template) as f:
            header = f[0].header
            seeref = header['SEEING']
            refskybkg = header['MEDSKY']
            trefskysig = header['SKYSIG']
            tu = header['SATURATE']

        refp = os.path.basename(template)[:-5]
        newp = os.path.basename(frame)[:-5]

        refdir = os.path.dirname(template)
        outdir = os.path.dirname(frame)

        subp = '_'.join([newp, refp])

        refweight = os.path.join(refdir, refp + '.weight.fits')
        refmask = os.path.join(refdir, refp + '.mask.fits')
        refcat = os.path.join(refdir, refp + '.cat')

        newweight = os.path.join(outdir, newp + '.weight.fits')
        newmask = os.path.join(outdir, newp + '.bpm.fits')
        newnoise = os.path.join(outdir, newp + '.rms.fits')
        newcat = os.path.join(outdir, newp + '.cat')
        newhead = os.path.join(outdir, newp + '.head')

        refremap = os.path.join(outdir, 'ref.%s.remap.fits' % subp)
        refremapweight = os.path.join(outdir, 'ref.%s.remap.weight.fits' % subp)
        refremapmask = os.path.join(outdir, 'ref.%s.remap.bpm.fits' % subp)
        refremapnoise = os.path.join(outdir, 'ref.%s.remap.rms.fits' % subp)
        refremaphead = os.path.join(outdir, 'ref.%s.remap.head' % subp)
        refremapcat = os.path.join(outdir, 'ref.%s.remap.cat' % subp)

        subcat = os.path.join(outdir, 'sub.%s.cat' % subp)
        sublist = os.path.join(outdir, 'sub.%s.list' % subp)
        submask = os.path.join(outdir, 'sub.%s.bpm.fits' % subp)
        apercat = os.path.join(outdir, 'ref.%s.remap.ap.cat' % subp)
        badpix = os.path.join(outdir, 'sub.%s.bpix' % subp)
        subrms = os.path.join(outdir, 'sub.%s.rms.fits' % subp)

        sub = os.path.join(outdir, 'sub.%s.fits' % subp)
        tmpnew = os.path.join(outdir, 'new.%s.fits' % subp)

        hotlog = os.path.join(outdir, 'hot.%s.log' % subp)
        hotpar = os.path.join(outdir, 'hot.%s.par' % subp)

        hotlogger = logging.getLogger('hotlog')
        hotparlogger = logging.getLogger('hotpar')

        fh = logging.FileHandler(hotlog)
        fhp = logging.FileHandler(hotpar)

        fh.setLevel(logging.DEBUG)
        fhp.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

        fh.setFormatter(formatter)
        fhp.setFormatter(formatter)

        hotparlogger.info(sub)
        hotparlogger.info(template)
        hotparlogger.info(frame)

        # read some keywords from the fits headers
        with fits.open(frame) as f:
            header = f[0].header
            seenew = header['SEEING']
            newskybkg = header['MEDSKY']
            tnewskysig = header['SKYSIG']
            iu = header['SATURATE']
            naxis1 = header['NAXIS1']
            naxis2 = header['NAXIS2']
            naxis = header['NAXIS']
            refzp = header['MAGZP']

            # make the naxis card images
            hstr = []
            for card in header.cards:
                if 'NAXIS' in card.keyword:
                    hstr.append(card.image)
            hstr = '\n'.join(hstr) + '\n'

        # Make a catalog from the reference for astrometric matching
        syscall = 'scamp -c %s -ASTREFCAT_NAME %s %s'
        syscall = syscall % (scampconfcat, refcat, newcat)
        print(syscall)
        execute(syscall, capture=False)

        # Merge header files
        with open(refremaphead, 'w') as f:
            f.write(hstr)
            with open(newhead, 'r') as nh:
                f.write(nh.read())

        # put all swarp temp files into a random dir
        swarp_rundir = f'/tmp/{uuid.uuid4().hex}'
        os.makedirs(swarp_rundir)

        # Make the remapped ref
        syscall = 'swarp -c %s %s -SUBTRACT_BACK N -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s'
        syscall = syscall % (defswarp, template, refremap, refremapweight)
        syscall += f' -VMEM_DIR {swarp_rundir} -RESAMPLE_DIR {swarp_rundir}'

        execute(syscall, capture=False)

        # Make the noise and bpm images
        make_rms(refremap, refremapweight)

        # Add the masks together to make the supermask
        cmbmask(refremapmask, newmask, submask)

        # Add the sub and new rms together in quadrature
        cmbrms(refremapnoise, newnoise, subrms)

        # Create new and reference noise images
        ntst = seenew > seeref

        seeing = seenew if ntst else seeref

        pixscal = 1.01  # TODO make this more general
        seepix = pixscal * seeing
        r = 2.5 * seepix
        rss = 6. * seepix

        hotlogger.info('r and rss %f %f' % (r, rss))

        gain = 1.0  # TODO check this assumption

        newskysig = tnewskysig * 1.48 / gain
        refskysig = trefskysig * 1.48 / gain

        il = newskybkg - 10. * newskysig
        tl = refskybkg - 10. * refskysig

        hotlogger.info('tl and il %f %f' % (tl, il))
        hotlogger.info('refskybkg and newskybkg %f %f' % (refskybkg, newskybkg))
        hotlogger.info('refskysig and newskysig %f %f' % (refskysig, newskysig))

        nsx = naxis1 / 100.
        nsy = naxis2 / 100.

        hotlogger.info('nsx nsy %f %f' % (nsx, nsy))
        hotparlogger.info(str(il))
        hotparlogger.info(str(iu))
        hotparlogger.info(str(tl))
        hotparlogger.info(str(tu))
        hotparlogger.info(str(r))
        hotparlogger.info(str(rss))
        hotparlogger.info(str(nsx))
        hotparlogger.info(str(nsy))

        convolve_target = 'i' if not ntst else 't'
        syscall = f'hotpants -inim %s -hki -n i -c {convolve_target} -tmplim %s -outim %s -tu %f -iu %f  -tl %f -il %f -r %f ' \
                  f'-rss %f -tni %s -ini %s -imi %s -nsx %f -nsy %f'
        syscall = syscall % (frame, refremap, sub, tu, iu, tl, il, r, rss, refremapnoise, newnoise,
                             submask, nsx, nsy)

        print(syscall)
        execute(syscall, capture=False)

        # Calibrate the subtraction

        with fits.open(sub, mode='update') as f, fits.open(frame) as fr:
            header = f[0].header
            subzp = fr[0].header['MAGZP']  # normalized to photometric system of new image
            header['MAGZP'] = subzp
            subpix = f[0].data
            wcs = WCS(header)

        subpixvar = (0.5 * (np.percentile(subpix, 84) - np.percentile(subpix, 16.)))**2

        # estimate the variance in psf fit fluxes
        subcorners = wcs.calc_footprint()
        subminra = subcorners[:, 0].min()
        submaxra = subcorners[:, 0].max()
        dra = submaxra - subminra
        submindec = subcorners[:, 1].min()
        submaxdec = subcorners[:, 1].max()
        ddec = submaxdec - submindec

        subminra += 0.1 * dra
        submaxra -= 0.1 * dra
        submindec += 0.1 * ddec
        submaxdec -= 0.1 * ddec

        psfimpath = frame.replace('.fits', '.psf.fits')

        fluxes = []
        for i in range(1000):
            ra = np.random.uniform(subminra, submaxra)
            dec = np.random.uniform(submindec, submaxdec)
            pobj = yao_photometry_single(sub, psfimpath, ra, dec)
            fluxes.append(pobj.Fpsf)

        subpsffluxvar = (0.5 * (np.percentile(fluxes, 84) - np.percentile(fluxes, 16.)))**2
        beta = subpsffluxvar / subpixvar

        with fits.open(sub, mode='update') as f:
            f[0].header['BETA'] = beta

        # Make the subtraction catalogs
        clargs = ' -PARAMETERS_NAME %%s -FILTER_NAME %s -STARNNW_NAME %s' % (defconv, defnnw)

        # Reference catalog
        syscall = 'sex -c %s -MAG_ZEROPOINT %f -CATALOG_NAME %s -VERBOSE_TYPE QUIET %s'
        syscall = syscall % (defsexref, refzp, refremapcat, refremap)
        syscall += clargs % defparref
        execute(syscall, capture=False)

        # Subtraction catalog
        syscall = 'sex -c %s -MAG_ZEROPOINT %f -CATALOG_NAME %s -ASSOC_NAME %s -VERBOSE_TYPE QUIET %s -WEIGHT_IMAGE %s -WEIGHT_TYPE MAP_RMS'
        syscall = syscall % (defsexsub, subzp, subcat, refremapcat, sub, subrms)
        syscall += clargs % defparsub
        syscall += f' -FLAG_IMAGE {submask}'
        print(syscall)
        execute(syscall, capture=False)

        # Aperture catalog
        syscall = 'sex -c %s -MAG_ZEROPOINT %f -CATALOG_NAME %s -VERBOSE_TYPE QUIET %s,%s'
        syscall = syscall % (defsexaper, refzp, apercat, sub, refremap)
        syscall += clargs % defparaper
        execute(syscall, capture=False)

        # now filter objects
        filter_sexcat(subcat)

        # publish to marshal

        if publish:
            goodcat = subcat.replace('cat', 'cat.out.fits')
            load_catalog(goodcat, refremap, frame, sub)

        # put fake info in header and region file if there are any fakes
        with fits.open(frame) as f, fits.open(sub, mode='update') as fsub:
            hdr = f[0].header
            wcs = WCS(hdr)
            fakecards = [c for c in hdr.cards if 'FAKE' in c.keyword and 'X' not in c.keyword and 'Y' not in c.keyword]
            fakekws = [c.keyword for c in fakecards]

            try:
                maxn = max(set(map(int, [kw[4:-2] for kw in fakekws]))) + 1
            except ValueError:
                maxn = 0

            if maxn > 0:
                subheader = fsub[0].header
                subheader.update(fakecards)

                with open(sub.replace('fits', 'fake.reg'), 'w') as o:

                    # make the region files
                    o.write("""# Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal" \
    select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 \
    delete=1 include=1 source=1
    physical
    """)

                    for i in range(maxn):
                        ra = hdr[f'FAKE{i:02d}RA']
                        dec = hdr[f'FAKE{i:02d}DC']
                        mag = hdr[f'FAKE{i:02d}MG']
                        fake = Fake(ra, dec, mag)
                        x, y = fake.xy(wcs)
                        subheader[f'FAKE{i:02d}X'] = x
                        subheader[f'FAKE{i:02d}Y'] = y

                        o.write(f'circle({x},{y},10) # width=2 color=red\n')
                        o.write(f'text({x},{y+8}  # text={{mag={mag:.2f}}}\n')



if __name__ == '__main__':

    import argparse

    #from mpi4py import MPI
    #comm = MPI.COMM_WORLD
    rank = 0 #comm.Get_rank()
    size = 1 #comm.Get_size()

    # set up the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--science-frames', dest='sciimg', required=True,
                        help='Input frames. Prefix with "@" to read from a list.', nargs='+')
    parser.add_argument('--templates', dest='template', nargs='+', required=True,
                        help='Templates to subtract. Prefix with "@" to read from a list.')
    parser.add_argument('--no-publish', dest='publish', action='store_false', default=True,
                        help='Dont publish results to the marshal.')
    args = parser.parse_args()

    # distribute the work to each processor

    if args.sciimg[0].startswith('@'):
        framelist = args.sciimg[0][1:]

        if rank == 0:
            frames = np.genfromtxt(framelist, dtype=None, encoding='ascii')
            frames = np.atleast_1d(frames).tolist()
        else:
            frames = None
        #frames = comm.bcast(frames, root=0)
    else:
        frames = args.sciimg

    if args.template[0].startswith('@'):
        templist = args.template[0][1:]

        if rank == 0:
            templates = np.genfromtxt(templist, dtype=None, encoding='ascii')
            templates = np.atleast_1d(templates).tolist()
        else:
            templates = None
        #templates = comm.bcast(templates, root=0)
    else:
        templates = args.template

    myframes = _split(frames, size)[rank]
    mytemplates = _split(templates, size)[rank]

    make_sub(myframes, mytemplates, publish=args.publish)




