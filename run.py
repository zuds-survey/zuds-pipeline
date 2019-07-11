import db
import os
import yaml
import time
from itertools import chain
import numpy
import shutil
from pathlib import Path
from argparse import ArgumentParser
from datetime import timedelta
import pandas as pd

# read/write for group
os.umask(0o007)

__whatami__ = 'Run a lensgrinder ZTF task.'
__author__ = 'Danny Goldstein <danny@caltech.edu>'


def df_from_sa_objects(sa_objects):
    try:
        return pd.DataFrame([o.to_dict() for o in sa_objects])
    except TypeError:
        return pd.DataFrame([sa_objects.to_dict()])


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('task', help='Path to yaml file describing the workflow.')
    args = parser.parse_args()

    # process the task
    task_file = args.task
    task_name = '.'.join(os.path.basename(task_file).split('.')[:-1])
    task_spec = yaml.load(open(task_file, 'r'))

    # connect to the database
    env, cfg = db.load_env()
    db.init_db(**cfg['database'])

    # prepare the output directory for this particular task
    outdir = os.getenv('OUTPUT_DIRECTORY')
    task_output = Path(outdir) / task_name

    if task_output.exists() and task_spec['clobber']:
        shutil.rmtree(task_output)
    task_output.mkdir(exist_ok=True, parents=True)

    # make all the subdirectories that will be needed

    jobscripts = task_output / 'job_scripts'
    logs = task_output / 'logs'
    framepath = task_output / 'frames'
    templates = task_output / 'templates'

    jobscripts.mkdir(exist_ok=True, parents=True)
    logs.mkdir(exist_ok=True, parents=True)
    framepath.mkdir(exist_ok=True, parents=True)
    templates.mkdir(exist_ok=True, parents=True)

    # if an object name is provided then we must infer the query
    # retrieve the images off of tape

    from retrieve import retrieve_images, full_query

    preserve_dirs = task_spec['hpss']['preserve_directories']

    if task_spec['hpss']['object_name'] is None:
        whereclause = task_spec['hpss']['whereclause']
    else:
        object = db.DBSession().query(db.models.Source).get(task_spec['hpss']['object_name'])

        # does a q3c query - should take  <10s
        images = object.images

        whereclause = f'ID IN {tuple([image.id for image in images])}'

    exclude_masks = task_spec['hpss']['exclude_masks']

    if not task_spec['hpss']['rerun']:
        hpss_dependencies, metatable = retrieve_images(whereclause, exclude_masks=exclude_masks,
                                                       job_script_destination=jobscripts,
                                                       frame_destination=framepath, log_destination=logs,
                                                       preserve_dirs=preserve_dirs)
    else:
        hpss_dependencies = {}
        query = full_query(whereclause)
        metatable = pd.read_sql(query, db.DBSession().get_bind())

    # ensure the full paths are propagated if desired
    if preserve_dirs:
        new_hpss_dependencies = {}
        for i, row in metatable.iterrows():
            np = f"{row['field']:06d}/c{row['ccdid']:02d}/q{row['qid']}/{row['filtercode']}/{row['path']}"
            metatable.loc[i, 'path'] = np
            if len(hpss_dependencies) > 0:
                new_hpss_dependencies[np] = hpss_dependencies[row['path']]
        hpss_dependencies = new_hpss_dependencies

    # check to see if hpss jobs have finished

    deps = list(set(hpss_dependencies.values()))
    if len(deps) > 0:
        while True:

            done = db.DBSession().query(db.sa.func.bool_and(db.HPSSJob.status)) \
                                 .filter(db.HPSSJob.id.in_(deps)) \
                                 .first()[0]
            if done:
                break
            else:
                time.sleep(3.)


    # make the variance maps
    options = task_spec['makevariance']
    batch_size = options['batch_size']
    from makevariance import submit_makevariance
    frames = [im for im in metatable['path'] if 'msk' not in im]
    masks = [im.replace('sciimg', 'mskimg') for im in frames]
    variance_dependencies = submit_makevariance(frames, masks, task_name=task_name,
                                                batch_size=batch_size, log_destination=logs,
                                                frame_destination=framepath,
                                                job_script_destination=jobscripts)

    metatable['full_path'] = [f'{(framepath / frame).resolve()}' for frame in frames]

    # todo: add fakes


    from makecoadd import submit_template
    options = task_spec['template']

    template_nimages = options['nimages']
    template_start_date = options['start_date']
    template_end_date = options['end_date']
    template_science_minsep_days = options['template_science_minsep_days']

    # first check to see if we have templates

    # create templates if needed

    # now go one field, chip, quad, filter at a time:

    for (field, ccdid, qid, filtercode), group in metatable.groupby(['field', 'ccdid', 'qid', 'filtercode']):

        # check for template

        # converting to python types to avoid psycopg2 coersion error

        match = db.sa.and_(db.Reference.field == int(field),
                           db.Reference.ccdid == int(ccdid),
                           db.Reference.qid == int(qid),
                           db.Reference.filtercode == str(filtercode))

        filt = db.sa.and_(db.sa.func.count(db.Image.id) >= template_nimages,
                          db.sa.func.min(db.Image.obsdate) >= template_start_date,
                          db.sa.func.max(db.Image.obsdate) <= template_end_date)

        ref = db.DBSession().query(db.Reference)\
                            .join(db.ReferenceImage)\
                            .join(db.Image)\
                            .filter(match)\
                            .group_by(db.Reference.id)\
                            .having(filt)\
                            .order_by(db.Reference.id.desc())\
                            .first()

        if ref is None:
            # we need a reference

            template_dependencies, remaining_images, ref = submit_template(variance_dependencies,
                                                                           group,
                                                                           template_destination=templates,
                                                                           task_name=task_name,
                                                                           log_destination=logs,
                                                                           job_script_destination=jobscripts,
                                                                           nimages=template_nimages,
                                                                           start_date=template_start_date,
                                                                           end_date=template_end_date,
                                                                           template_science_minsep_days=template_science_minsep_days)
        else:
            template_dependencies = {}
            mindate = template_start_date - timedelta(days=template_science_minsep_days)
            maxdate = template_end_date + timedelta(days=template_science_minsep_days)
            remaining_ids = numpy.setdiff1d(group['id'], [i.id for i in ref.images]).tolist()

            remaining_q = db.DBSession().query(db.Image)\
                                        .filter(db.sa.and_(db.Image.id.in_(remaining_ids),
                                                           db.sa.or_(db.Image.obsdate <= mindate,  # note the logic here
                                                           db.Image.obsdate >= maxdate))
                                                )

            remaining_images = pd.read_sql(remaining_q.statement, db.DBSession().get_bind())


        from makesub import submit_coaddsub
        options = task_spec['coaddsub']
        rolling = options['rolling']
        coadd_windowsize = options['coadd_windowsize']
        batch_size = options['batch_size']

        coaddsub_dependencies = submit_coaddsub(template_dependencies, variance_dependencies, remaining_images, ref,
                                                rolling=rolling, coadd_windowsize=coadd_windowsize,
                                                batch_size=batch_size, job_script_destination=jobscripts,
                                                log_destination=logs, frame_destination=framepath, task_name=task_name)

        if coadd_windowsize > 0:
            sub_dependencies = submit_coaddsub(template_dependencies, variance_dependencies, remaining_images, ref,
                                               rolling=rolling, coadd_windowsize=0,
                                               batch_size=batch_size, job_script_destination=jobscripts,
                                               log_destination=logs, frame_destination=framepath, task_name=task_name)
