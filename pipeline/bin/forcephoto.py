import sep
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import paramiko
from itertools import chain
import os


from publish import make_stamp
from astropy.visualization import ZScaleInterval

from baselayer.app.env import load_env
from skyportal.models import DBSession, Instrument, ForcedPhotometry, ForceThumb, Source, init_db

APER_RAD_FRAC_SEEING_FWHM = 0.6731
DB_FTP_DIR = '/skyportal/static/thumbnails'
DB_FTP_ENDPOINT = os.getenv('DB_FTP_ENDPOINT')
DB_FTP_USERNAME = 'root'
DB_FTP_PASSWORD = 'root'
DB_FTP_PORT = 222


# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


def post_stamps(stamps, points):

    thumbs = []
    with paramiko.Transport((DB_FTP_ENDPOINT, DB_FTP_PORT)) as transport:
        transport.connect(username=DB_FTP_USERNAME, password=DB_FTP_PASSWORD)
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            for triplet, photpoint in zip(stamps, points):
                for f in triplet:
                    remotename = os.path.join(DB_FTP_DIR, os.path.basename(f))
                    sftp.put(f, remotename)
                    uri = f'static/thumbnails/{os.path.basename(f)}'
                    thumb = ForceThumb(type=f.split('.')[-2],
                                       file_uri=uri, public_url='/' + uri,
                                       forcedphotometry_id=photpoint.id)
                    thumbs.append(thumb)
    DBSession().add_all(thumbs)
    DBSession().commit()


def force_photometry(sources, sub_list, send_stamps=True):

    instrument = DBSession().query(Instrument).filter(Instrument.name.like('%ZTF%')).first()

    stamps = []
    points = []

    for im in sub_list:

        with fits.open(im) as hdulist:
            hdu = hdulist[1]
            zeropoint = hdu.header['MAGZP']
            seeing = hdu.header['SEEING']  # FWHM of seeing
            mjd = hdu.header['OBSMJD']
            r_aper_arcsec = APER_RAD_FRAC_SEEING_FWHM * seeing
            r_aper_pix = r_aper_arcsec / 1.013 # arcsec per pixel
            wcs = WCS(hdu.header)
            maglim = hdu.header['MAGLIM']
            band = hdu.header['FILTER'][-1].lower()

            image = hdu.data
            rms = 1.4826 * np.median(np.abs(image - np.median(image)))

            interval = ZScaleInterval().get_limits(image)

        for source in sources:

            # get the RA and DEC of the source
            ra, dec = source.ra, source.dec

            # convert the ra and dec into pixel coordinates
            x, y = wcs.wcs_world2pix([[ra, dec]], 0.)[0]
            flux, fluxerr, flag = sep.sum_circle(image, x, y, r_aper_pix, err=rms)

            flux = flux[()]
            fluxerr = fluxerr[()]

            if flag == 0:

                force_point = ForcedPhotometry(mjd=mjd, flux=flux, fluxerr=fluxerr,
                                               zp=zeropoint, lim_mag=maglim, filter=band,
                                               source=source, instrument=instrument,
                                               ra=ra, dec=dec)

                DBSession().add(force_point)
                DBSession().commit()

                mystamps = []
                for key in ['sub', 'new']:
                    name = f'/stamps/{force_point.id}.force.{key}.png'
                    if key == 'new':
                        with fits.open(im.replace('scimrefdiffimg.fits.fz', 'sciimg.fits')) as hdul:
                            newimage = hdul[0].data
                            newimage = newimage.byteswap().newbyteorder()
                            newwcs = WCS(hdul[0].header)
                            newinterval = ZScaleInterval().get_limits(newimage)
                        make_stamp(name, force_point.ra, force_point.dec, newinterval[0], newinterval[1], newimage,
                                   newwcs)
                    else:
                        make_stamp(name, force_point.ra, force_point.dec, interval[0], interval[1], image,
                                   wcs)
                    mystamps.append(name)

                stamps.append(mystamps)
                points.append(force_point)

    if send_stamps:
        post_stamps(stamps, points)
    else:
        return stamps, points


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('sub_names', nargs="+")
    args = parser.parse_args()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    sub_list = args.sub_names

    env, cfg = load_env()
    init_db(**cfg['database'])

    my_names = _split(sub_list, size)[rank]

    allstamps = []
    allpoints = []

    for sub in my_names:
        with fits.open(sub) as f:
            wcs = WCS(f[1].header)
            footprint = wcs.calc_footprint().ravel()
            source_ids = DBSession().execute('SELECT ID FROM SOURCES WHERE Q3C_POLY_QUERY(RA, DEC, '
                                             '\'{%f,%f,%f,%f,%f,%f,%f,%f}\')' % tuple(footprint.tolist())).fetchall()
            source_ids = [s[0] for s in source_ids]

            sources = DBSession().query(Source).filter(Source.id.in_(source_ids)).all()

            stamps, points = force_photometry(sources, [sub], send_stamps=False)

            allstamps.extend(stamps)
            allpoints.extend(points)

    stamps = comm.gather(allstamps, root=0)
    points = comm.gather(allpoints, root=0)

    stamps = list(chain(*stamps))
    points = list(chain(*points))

    # send them
    post_stamps(stamps, points)
