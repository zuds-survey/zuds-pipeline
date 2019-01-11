import sep
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import paramiko
import os

from publish import make_stamp
from astropy.visualization import ZScaleInterval

from skyportal.models import DBSession, Instrument, ForcedPhotometry, ForceThumb

APER_RAD_FRAC_SEEING_FWHM = 0.6731
DB_FTP_DIR = '/skyportal/static/thumbnails'
DB_FTP_ENDPOINT = os.getenv('DB_FTP_ENDPOINT')
DB_FTP_USERNAME = 'root'
DB_FTP_PASSWORD = 'root'
DB_FTP_PORT = 222


def force_photometry(sources, sub_list):

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
