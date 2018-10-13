#!/usr/bin/env python

import os
import pandas as pd
import argparse
import warnings

from astropy.io import fits


def make_output(outname, framenames):
    with open(outname, 'w') as f:
        for frame in framenames:
            f.write('%s\n' % frame)


def img_in_range(image, range_low, range_high):
    time = pd.to_datetime(fits.open(image)[0].header['SHUTOPEN'])
    return range_low <= time < range_high


def get_seeing(image):
    seeing = float(fits.open(image)[0].header['SEEING'])
    return seeing


def get_date(image):
    return pd.to_datetime(fits.open(image)[0].header['SHUTOPEN'])


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

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # prune the ones with bad seeing
    frames = []
    inds = []
    for i, img in enumerate(framenames):
        seeing = get_seeing(img)
        if seeing <= args.max_seeing:
            frames.append(img)
            inds.append(i)
        else:
            warnings.warn('Frame "%s" has seeing=%0.2f (>%0.2f). '
                          'Skipping...' % (img, seeing, args.max_seeing),
                          UserWarning)
    # make the bins

    if args.window_size > 0:
        dates = list(map(get_date, frames))
        mindate = min(dates)
        maxdate = max(dates)
        
        if args.rolling:
            dates = pd.date_range(mindate, maxdate, freq='1D')
            datebins = []
            for i, date in enumerate(dates):
                if i + args.window_size >= len(dates):
                    break
                datebins.append((date, dates[i + args.window_size]))

        else:
            datebins = pd.date_range(mindate, maxdate, freq='%dD' % args.window_size)
            datebins = [(datebins[i], datebins[i + 1]) for i in range(len(datebins) - 1)]

        for start, stop in datebins:
            startstr = start.strftime('%Y%m%d')
            stopstr = stop.strftime('%Y%m%d')

            dirname = '%s-%s' % (startstr, stopstr)

            myframes = []
            myinds = []
            for im, ind in zip(frames, inds):
                if img_in_range(im, start, stop):
                    myframes.append(im)
                    myinds.append(ind)

            for l in args.associated:
                with open(l, 'r') as f:
                    aframes = [line.strip() for i, line in enumerate(f) if i in myinds]
                    if len(aframes) != len(myinds):
                        raise ValueError('Length of associated frame list "%s" must be the same '
                                         'as number of input frames.' % l)
                    myframes.extend(aframes)

            if len(myframes) > 0:
                make_output('%s.%s' % (dirname, args.outfile), myframes)
    else:
        # just do one big coadd
        for l in args.associated:
            with open(l, 'r') as f:
                aframes = [line.strip() for i, line in enumerate(f) if i in inds]
                if len(aframes) != len(frames):
                    raise ValueError('Length of associated frame list "%s" must be the same '
                                     'as number of input frames.' % l)
                frames.extend(aframes)
        make_output(args.outfile, frames)
