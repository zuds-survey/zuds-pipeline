import os
import yaml
import shutil
from pathlib import Path
from argparse import ArgumentParser

__whatami__ = 'Run a lensgrinder ZTF task.'
__author__ = 'Danny Goldstein <danny@caltech.edu>'

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('task', help='Path to yaml file describing the workflow.')
    args = parser.parse_args()

    # process the task
    task_file = args.task
    task_name = '.'.join(os.path.basename(task_file).split('.')[:-1])
    task_spec = yaml.load(open(task_file, 'r'))


    # prepare the output directory for this particular task
    outdir = os.getenv('OUTPUT_DIRECTORY')
    task_output = Path(outdir) / task_name
    if task_output.exists():
        shutil.rmtree(task_output)
    task_output.mkdir()

    # make all the subdirectories that will be needed
    jobscripts = task_output / 'job_scripts'
    logs = task_output / 'logs'
    framepath = task_output / 'frames'
    templates = task_output / 'templates'

    jobscripts.mkdir()
    logs.mkdir()
    framepath.mkdir()
    templates.mkdir()

    # retrieve the images off of tape
    from retrieve import retrieve_images
    whereclause = task_spec['hpss']['whereclause']
    exclude_masks = task_spec['hpss']['exclude_masks']
    hpss_dependencies, metatable = retrieve_images(whereclause, exclude_masks=exclude_masks,
                                                   job_script_destination=jobscripts,
                                                   frame_destination=framepath, log_destination=logs)

    # make the variance maps
    options = task_spec['makevariance']
    batch_size = options['batch_size']
    from makevariance import submit_makevariance

    if 'frames' in options and options['frames'] is not None:
        frames = options['frames']
        dependencies = None
    elif 'hpss' in task_spec:
        frames = [im for im in hpss_dependencies if 'msk' not in im]
        dependencies = hpss_dependencies
    else:
        raise ValueError('No images specified')

    masks = [im.replace('sciimg', 'mskimg') for im in frames]
    variance_dependencies = submit_makevariance(frames, masks, task_name=task_name,
                                                batch_size=batch_size, log_destination=logs,
                                                frame_destination=framepath,
                                                job_script_destination=jobscripts)

    metatable['full_path'] = [f'{(framepath / frame).resolve()}' for frame in frames]

    # todo: add fakes

    # create templates

    from makecoadd import submit_template
    options = task_spec['template']

    template_nimages = options['nimages']
    template_start_date = options['start_date']
    template_end_date = options['end_date']

    template_dependencies, remaining_images = submit_template(variance_dependencies, metatable,
                                                              template_destination=templates,
                                                              task_name=task_name,
                                                              log_destination=logs,
                                                              job_script_destination=jobscripts,
                                                              template_nimages=template_nimages,
                                                              template_start_date=template_start_date,
                                                              template_end_date=template_end_date)


