from astropy.io import fits
from astropy.wcs import WCS
import os
from publish import make_stamp
from astropy.visualization import ZScaleInterval
from liblg.yao import yao_photometry_single
from baselayer.app.env import load_env
from skyportal.models import DBSession, Instrument, ForcedPhotometry, ForceThumb, Source, init_db

from uuid import uuid4


APER_RAD_FRAC_SEEING_FWHM = 0.6731
DB_FTP_DIR = '/skyportal/static/thumbnails'
DB_FTP_ENDPOINT = os.getenv('DB_FTP_ENDPOINT')
DB_FTP_USERNAME = 'root'
DB_FTP_PASSWORD = 'root'
DB_FTP_PORT = 222



# split an iterable over some processes recursively
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
             _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


def force_photometry(sources, sub_list, psf_list):

    instrument = DBSession().query(Instrument).filter(Instrument.name.like('%ZTF%')).first()

    for im, psf in zip(sub_list, psf_list):

        thumbs = []
        points = []

        with fits.open(im) as hdulist:
            hdu = hdulist[1]
            zeropoint = hdu.header['MAGZP']
            mjd = hdu.header['OBSMJD']

            wcs = WCS(hdu.header)
            maglim = hdu.header['MAGLIM']
            band = hdu.header['FILTER'][-1].lower()

            image = hdu.data
            interval = ZScaleInterval().get_limits(image)

        for source in sources:

            # get the RA and DEC of the source
            ra, dec = source.ra, source.dec

            try:
                pobj = yao_photometry_single(im, psf, ra, dec)
            except IndexError:
                continue
            if not pobj.status:
                continue

            flux = pobj.Fpsf
            fluxerr = pobj.eFpsf

            force_point = ForcedPhotometry(mjd=mjd, flux=float(flux), fluxerr=float(fluxerr),
                                           zp=zeropoint, lim_mag=maglim, filter=band,
                                           source=source, instrument=instrument,
                                           ra=ra, dec=dec)

            points.append(force_point)

        DBSession().add_all(points)
        DBSession().commit()

        for source, force_point in zip(sources, points):
            for key in ['sub', 'new']:
                name = f'/stamps/{uuid4().hex}.force.{key}.png'
                if key == 'new':
                    fname = im.replace('scimrefdiffimg.fits.fz', 'sciimg.fits')
                    if not os.path.exists(fname):
                        continue
                    with fits.open(fname) as hdul:
                        newimage = hdul[0].data
                        newimage = newimage.byteswap().newbyteorder()
                        newwcs = WCS(hdul[0].header)
                        newinterval = ZScaleInterval().get_limits(newimage)
                    make_stamp(name, force_point.ra, force_point.dec, newinterval[0], newinterval[1], newimage,
                               newwcs)
                else:
                    make_stamp(name, force_point.ra, force_point.dec, interval[0], interval[1], image,
                               wcs)

                thumb = ForceThumb(type=key, forcedphotometry_id=force_point.id, file_uri=None,
                                   public_url='http://portal.nersc.gov/project/astro250'+ name)
                thumbs.append(thumb)

        DBSession().add_all(thumbs)
        DBSession().commit()


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

    if sub_list[0].startswith('@'):
        with open(sub_list[0][1:], 'r') as f:
            sub_list = f.read().split()

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

            force_photometry(sources, [sub])
