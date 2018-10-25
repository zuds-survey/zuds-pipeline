import pandas as pd
import argparse
import warnings
from astropy.io import fits


def make_output(outname, framenames):
    with open(outname, 'w') as f:
        for frame in framenames:
            f.write('%s\n' % frame)


def img_in_range(image, range_low, range_high):
    with fits.open(image, 'r') as f:
        shutopen = f[0].header['SHUTOPEN']
    time = pd.to_datetime(shutopen)
    return range_low <= time < range_high


def get_seeing(image):
    with fits.open(image, 'r') as f:
        seeing = f[0].header['SEEING']
    return seeing


def get_date(image):
    with fits.open(image, 'r') as f:
        shutopen = f[0].header['SHUTOPEN']
    return pd.to_datetime(shutopen)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-frames', required=True, help='List of the frames to coadd.',
                        nargs=1, dest='frames')
    parser.add_argument('--outfile-name', help='Name of file in which to write results.',
                        dest='outfile', required=False, default='list')
    parser.add_argument('--associated-frames', required=False, default=[], help='Lists of frames, in '
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

            if len(myframes) > 0:
                make_output('%s.%s' % (dirname, args.outfile), myframes)
            else:
                continue

            for l in args.associated:
                with open(l, 'r') as f:
                    aframes = [line.strip() for i, line in enumerate(f) if i in myinds]
                    if len(aframes) != len(myinds):
                        raise ValueError('Length of associated frame list "%s" must be the same '
                                         'as number of input frames.' % l)
                make_output('%s.%s.%s' % (dirname, l, args.outfile), aframes)

    else:
        # just do one big coadd
        make_output(args.outfile, frames)
        
        for l in args.associated:
            with open(l, 'r') as f:
                aframes = [line.strip() for i, line in enumerate(f) if i in inds]
                if len(aframes) != len(frames):
                    raise ValueError('Length of associated frame list "%s" must be the same '
                                     'as number of input frames.' % l)
            make_output('%s.%s' % (l, args.outfile), aframes)

