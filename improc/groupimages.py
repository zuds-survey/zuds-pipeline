#!/usr/bin/env python

import shutil, os
import pandas as pd
import argparse
import warnings
import glob

from astropy.io import fits


class ReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir=values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))


def make_directory(topdir, dirname, framenames, clobber):
    refdir = os.path.join(topdir, dirname)
    if os.path.exists(refdir):
        action = 'Overwriting' if clobber else 'Not overwriting'
        warnings.warn('Template directory %s already exists. %s...' % (refdir, action), UserWarning)
        if clobber:
            shutil.rmtree(refdir)
            os.mkdir(refdir)
    for frame in framenames:
        shutil.copy(frame, refdir)


def img_in_range(image, range_low, range_high):
    time = pd.datetime(fits.open(image)[0].header['SHUTOPEN'])
    return range_low <= time < range_high


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-frames', required=True, help='List of the frames to build '
                        'a reference with (one per line).', fromfile_prefix_chars='@',
                        type=str, nargs='+')
    parser.add_argument('--output-dir', help='Directory in which to write results.', action=ReadableDir,
                        dest='outdir', required=False, default='.')
    parser.add_argument('--rolling', help='Use a rolling window to make the groups.', default=False,
                        action='store_true', dest='rolling')
    parser.add_argument('--max-seeing', type=float, default=3., required=False, help='The maximum '
                        'seeing allowed for a reference frame.')
    parser.add_argument('scidate_range', required=True, type=pd.datetime, help='Date range for science images.',
                        nargs=2)
    parser.add_argument('--window-size', type=int, help='Number of days over which to coadd science images.',
                        required=False, default=10)
    parser.add_argument('--sciimg-dir', required=False, default='.', action=ReadableDir, nargs=1,
                        help='The directory to search for science images.')
    parser.add_argument('--clobber', required=False, default=False, action='store_true',
                        help='Overwrite existing output directories.')

    args = parser.parse_args()

    # make the reference directory
    make_directory(args.outdir, 'template', args.reference_frames, args.clobber)

    # make the science image groups, rolling or partitioned

    # list all the science images
    sciimgs = glob.glob(args.sciimg_dir)

    mindate, maxdate = args.daterange

    if args.rolling:
        dates = pd.date_range(mindate, maxdate, freq='1D')
        datebins = []
        for i, date in enumerate(dates):
            if i >= len(dates):
                continue
            datebins.append((date, dates[i + args.window_size]))

    else:
        datebins = pd.date_range(mindate, maxdate, freq='%dD' % args.window_size)
        datebins = [(datebins[i], datebins[i + 1]) for i in range(len(datebins) - 1)]

    for start, stop in datebins:
        startstr = start.strftime('%y%m%d')
        stopstr = stop.strftime('%y%m%d')

        dirname = '%s-%s' % (startstr, stopstr)

        framegen = filter(lambda im: img_in_range(im, start, stop), sciimgs)
        frames = list(framegen)

        make_directory(args.outdir, dirname, frames, args.clobber)
