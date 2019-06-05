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
    task_output = Path('/output') / task_name
    if task_output.exists():
        shutil.rmtree(task_output)
    task_output.mkdir()

    # make all the subdirectories that will be needed
    jobscripts = task_output / 'job_scripts'
    logs = task_output / 'logs'
    frames = task_output / 'frames'
    templates = task_output / 'templates'

    jobscripts.mkdir()
    logs.mkdir()
    frames.mkdir()
    templates.mkdir()

    # submit HPSS jobs if requested
    if 'hpss' in task_spec:

        from retrieve import retrieve_images
        whereclause = task_spec['hpss']['whereclause']
        exclude_masks = task_spec['hpss']['exclude_masks']
        hpss_dependencies = retrieve_images(whereclause, exclude_masks=exclude_masks,
                                            job_script_destination=jobscripts,
                                            frame_destination=frames)

    # submit make variance job if requested
    if 'makevariance' in task_spec:

        options = task_spec['makevariance']
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
        variance_dependencies = submit_makevariance(frames, masks, dependencies=dependencies, task_name=task_name)

    # create templates if requested
    if 'template' in task_spec:
        options = task_spec['template']
        from makecoadd import submit_makecoadd
        pass
