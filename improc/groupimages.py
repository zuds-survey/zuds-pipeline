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
        warnings.warn('Directory "%s" already exists. %s...' % (refdir, action), UserWarning)
        if clobber:
            shutil.rmtree(refdir)
            os.mkdir(refdir)
    else:
        os.mkdir(refdir)
    for frame in framenames:
        shutil.copy(frame, refdir)


def img_in_range(image, range_low, range_high):
    time = pd.datetime(fits.open(image)[0].header['SHUTOPEN'])
    return range_low <= time < range_high

def get_seeing(image):
    seeing = float(fits.open(image)[0].header['SEEING'])
    return seeing


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-frames', required=True, help='List of the frames to coadd.',
                        nargs=1, dest='frames')
    parser.add_argument('--output-dir', help='Directory in which to write results.', 
                        dest='outdir', required=False, default='.')
    parser.add_argument('--associated-frames', required=False, default=None, help='Lists of frames, in '
                        'the same order as input_frames, that should be grouped with the input frames.',
                        dest='associated', nargs='*')
    parser.add_argument('--rolling', help='Use a rolling window to group input frames.', default=False,
                        action='store_true', dest='rolling')
    parser.add_argument('--max-seeing', type=float, default=3., required=False, help='The maximum '
                        'seeing allowed for an input frame.')
    parser.add_argument('--window-size', type=int, help='Number of days over which to coadd input frames.'
                        ' If zero (default), then coadd all frames.',
                        required=False, default=0)
    parser.add_argument('--clobber', required=False, default=False, action='store_true',
                        help='Overwrite existing output directories.')

    args = parser.parse_args()
    with open(args.frames[0], 'r') as f:
        framenames = [line.strip() for line in f]

    # prune the ones with bad seeing
    frames = []
    inds = []
    for i, img in enumerate(framenames):
        seeing = get_seeing(img)
        if seeing <= args.max_seeing:
            frames.append(img)
            inds.append(i)
        else:
            warnings.warn('Frame %s has seeing=%0.2f (>%0.2f). '
                          'Skipping...' % (img, seeing, args.max_seeing),
                          UserWarning)
    # make the bins

    if args.window_size > 0:
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

            myframes = []
            myinds = []
            for im, ind in zip(frames, inds):
                if img_in_range(im, start, stop):
                    myframes.append(img)
                    myinds.append(ind)

            for l in args.associated:
                with open(l, 'r') as f:
                    aframes = [line.strip() for i, line in enumerate(f) if i in myinds]
                    if len(aframes) != len(frames):
                        raise ValueError('Length of associated frame list "%s" must be the same '
                                         'as number of input frames.' % l)
                    myframes.extend(aframes)
            
            make_directory(args.outdir, dirname, myframes, args.clobber)
            
    else:
        # just do one big coadd
        for l in args.associated:
            with open(l, 'r') as f:
                aframes = [line.strip() for i, line in enumerate(f) if i in inds]
                if len(aframes) != len(frames):
                    raise ValueError('Length of associated frame list "%s" must be the same '
                                     'as number of input frames.' % l)
                frames.extend(aframes)
        make_directory(args.outdir, '', frames, args.clobber)
    
