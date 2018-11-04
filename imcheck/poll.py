import os
import pika
import time
import datetime
import psycopg2
import requests
import schedule
import logging
import uuid
import json

import numpy as np
import pandas as pd
from ztfquery import query
from astropy.time import Time

ipac_root = 'http://irsa.ipac.caltech.edu/'
formula = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{year:s}/{month:s}{day:s}/{fracday:s}' \
          '/ztf_{filefracday:s}_{paddedfield:s}_{filtercode:s}_c{paddedccdid:s}_{imgtypecode:s}_q{qid:d}_sciimg.fits'
nersc_formula = '/global/cscratch1/sd/dgold/ztfcoadd/{paddedfield:s}/c{paddedccdid:s}/{qid:d}/{filtercode:s}/{fname:s}'
newt_baseurl = 'https://newt.nersc.gov/newt'
variance_batchsize = 1024

# secrets
database_uri = os.getenv('DATABASE_URI')
ipac_username = os.getenv('IPAC_USERNAME')
ipac_password = os.getenv('IPAC_PASSWORD')
nersc_username = os.getenv('NERSC_USERNAME')
nersc_password = os.getenv('NERSC_PASSWORD')


def ipac_authenticate():

    target = os.path.join(ipac_root, 'account', 'signon', 'login.do')
    payload = {'username': ipac_username,
               'password': ipac_password}

    r = requests.post(target, data=payload)

    if r.status_code != 200:
        raise ValueError('Unable to Authenticate')

    return r.cookies


def nersc_authenticate():

    target = os.path.join(newt_baseurl, 'login')
    payload = {'username':nersc_username,
               'password':nersc_password}

    r = requests.post(target, data=payload)

    if r.status_code != 200:
        raise ValueError('Unable to Authenticate')

    return r.cookies


class IPACQueryManager(object):

    def __init__(self, pipeline_schema):
        self.pipeline_schema = pipeline_schema
        self.response = None
        self._refresh_connections()

    def _generate_paths(self, df):

        # check to see if the image has been downloaded already

        ipaths = []
        npaths = []

        for i, row in df.iterrows():
            ffd = str(row['filefracday'])
            year = ffd[:4]
            month = ffd[4:6]
            day = ffd[6:8]
            fracday = ffd[8:]
            filefracday = ffd
            paddedfield = '%06d' % row['field']
            qid = row['qid']
            imgtypecode = row['imgtypecode']
            filtercode = row['filtercode']
            paddedccdid = '%02d' % row['ccdid']
            tpath = formula.format(year=year,
                                   month=month,
                                   day=day,
                                   fracday=fracday,
                                   filefracday=filefracday,
                                   paddedfield=paddedfield,
                                   qid=qid,
                                   imgtypecode=imgtypecode,
                                   filtercode=filtercode,
                                   paddedccdid=paddedccdid)

            fname = tpath.split('/')[-1]
            npath = nersc_formula.format(paddedfield=paddedfield,
                                         paddedccdid=paddedccdid,
                                         qid=qid,
                                         filtercode=filtercode,
                                         fname=fname)

            dbq = 'SELECT COUNT(*) FROM IMAGE WHERE PATH=%s'
            self.cursor.execute(dbq, npath)
            count = self.cursor.fetchone()[0]

            if count > 0:
                continue

            ipaths.append(tpath)
            npaths.append(npath)

        return ipaths, npaths

    def _refresh_connections(self):

        try:
            self.connection.close()
        except:
            pass

        try:
            self.dbc.close()
        except:
            pass

        params = pika.ConnectionParameters('msgqueue')
        self.connection = pika.BlockingConnection(params)

        # set up the work queue and the dead letter exchange for delayed requeueing

        job_channel = self.connection.channel()
        job_channel.queue_declare(queue='jobs', arguments={
            'x-dead-letter-exchange': 'jobs-retry',
        })
        job_channel.exchange_declare(exchange_type='direct',
                                 exchange='jobs',
                                 queue='jobs')

        delay_channel = self.connection.channel()
        delay_channel.queue_declare(queue='jobs-retry', arguments={
            'x-message-ttl': 300000,  # wait five minutes
            'x-dead-letter-exchange': 'jobs',
        })
        delay_channel.exchange_declare(exchange='jobs-retry',
                                       exchange_type='direct',
                                       queue='jobs-retry')

        # cache the channels
        self.job_channel = job_channel
        self.delay_channel = delay_channel

        # connect to the database
        self.dbc = psycopg2.connect(database_uri)
        self.cursor = self.dbc.cursor()

    def list_current_template_images(self, field, ccdnum, quadrant, filter):

        # list images that are in the template
        tquery = 'SELECT ID FROM TEMPLATE WHERE FIELD=%s AND CCDNUM=%s AND QUADRANT=%s AND FILTER=%s ' \
                 'AND PIPELINE_SCHEMA_ID=%s'
        self.cursor.execute(tquery, filter, quadrant, field, ccdnum, self.pipeline_schema['schema_id'])
        result = self.cursor.fetchall()

        if len(result) == 0:
            return []

        else:
            tid = result[0][0]
            iquery = 'SELECT I.PATH FROM  TEMPLATEIMAGEASSOC TA JOIN IMAGE I ON TA.IMAGE_ID = I.ID ' \
                     'WHERE TA.TEMPLATE_ID=%s'
            self.cursor.execute(iquery, tid)
            images = [r[0] for r in self.cursor.fetchall()]
            return images

    def needs_template(self, field, ccdnum, quadrant, filter):
        query = 'SELECT COUNT(*) FROM TEMPLATE WHERE FIELD=%s AND CCDNUM=%s AND QUADRANT=%s AND FILTER=%s ' \
                'AND PIPELINE_SCHEMA_ID=%s'
        self.cursor.execute(query, field, ccdnum, quadrant, filter, self.pipeline_schema['schema_id'])
        result = self.cursor.fetchall()[0][0]

        # return true if there is no template
        return result == 0

    def relay_job(self, body):
        """Send a job off to be submitted."""
        corr_id = uuid.uuid4().hex
        corrprop = pika.BasicProperties(correlation_id=corr_id)
        self.job_channel.basic_publish(exchange='',
                                       routing_key='jobs',
                                       properties=corrprop,
                                       body=body)

        jobtype = json.loads(body)['jobtype']

        # also write its existence to the database
        query = 'INSERT INTO JOB (CORR_ID, JOBTYPE) VALUES (%s, %s)'
        self.cursor.execute(query, corrprop, jobtype)
        self.dbc.commit()

        return corr_id

    def create_template_image_list(self, field, ccdnum, quadrant, filter):
        query = 'SELECT ID, PATH FROM IMAGE WHERE FIELD=%s AND CCDNUM=%s AND QUADRANT=%s AND FILTER=%s ' \
                'AND GOOD=TRUE ORDER BY OBSJD ASC LIMIT %s'
        self.cursor.execute(query, field, ccdnum, quadrant, filter,
                            self.pipeline_schema['template_minimages'])
        result = self.cursor.fetchall()
        return result

    def __call__(self):

        # refresh our connections
        self._refresh_connections()

        # authenticate to the various places
        icookies = ipac_authenticate()
        ncookies = nersc_authenticate()

        # query for the new images by getting everything after 5pm pacific time (midnight UTC)
        # from the previous day

        td = datetime.date.today()
        dt = datetime.datetime(td.year, td.month, td.day, 0, 0, 0)
        at = Time(dt)
        jd = at.jd

        zquery = query.ZTFQuery()
        zquery.load_metadata(sql_query='obsjd>=%f' % jd, auth=[ipac_username, ipac_password])
        tab = zquery.metatable
        ipaths, npaths = self._generate_paths(tab)

        sessionid = icookies.get('sessionid')
        download_script = [f'curl {ipath} --create-dirs -o {npath} --cookie "SESSIONID={sessionid}"'
                           for ipath, npath in zip(ipaths, npaths)]
        download_script = '\n'.join(download_script)

        # upload the download script to NERSC
        path = f'/global/cscratch1/sd/dgold/ztfcoadd/download_scripts/{jd}.sh'
        target = os.path.join(newt_baseurl, 'file', 'dtn02', f'{path}')

        requests.put(target, data=download_script, cookies=ncookies)

        # now run it
        target = os.path.join(newt_baseurl, 'command', 'dtn02')
        payload = {'executable': f'/bin/bash {path}', 'loginenv':True}
        r = requests.post(target, data=payload, cookies=ncookies)
        jr = r.json()

        # check that everything worked as expected

        if r.status_code != 200:
            raise RuntimeError('Error contacting NEWT')

        if jr['status'] == 'ERROR':
            error = jr['error']
            raise RuntimeError(f'Error on dtn02: {error}')
        else:
            logging.info(jr['output'])

        # now update the database with the new images
        dbq = 'INSERT INTO IMAGE (PATH, FILTER, QUADRANT, FIELD, CCDNUM, PROGRAMID, OBSJD) VALUES ' \
              '(%s, %s, %s, %s, %s, %s, %s);'

        for (i, row), npath in zip(tab.iterrows(), npaths):
            self.cursor.execute(dbq, npath, row['filtercode'][1:], int(row['qid']),
                                int(row['paddedfield']), int(row['paddedccdid']),
                                row['pid'], row['obsjd'])
        self.dbc.commit()

        # now actually determine the new jobs

        # everything needs variance. submit batches of 1024

        variance_corrids = {}
        nvariance_jobs = len(npaths) // variance_batchsize + (1 if len(npaths) % variance_batchsize > 0 else 0)
        for i in range(nvariance_jobs):

            batch = npaths[i * variance_batchsize : (i + 1) * variance_batchsize]
            body = json.dumps({'jobtype':'variance', 'dependencies':None, 'images':batch})

            # send a message to the job submission script telling it to submit the job
            correlation_id = self.relay_job(body)

            for path in batch:
                variance_corrids[path] = correlation_id

        # check to see if new templates are needed
        template_corrids = {}
        for (field, quadrant, band, ccdnum), group in tab.groupby(['paddedfield', 'qid', 'filtercode', 'paddedccdid']):

            # convert into proper values
            field = int(field)
            quadrant = int(quadrant)
            band = band[1:]
            ccdnum = int(ccdnum)

            # check if a template is needed
            if not self.needs_template(field, ccdnum, quadrant, band):
                continue

            tmplims = self.create_template_image_list(field, ccdnum, quadrant, band)

            # see what jobs need to finish before this one can run
            dependencies = []
            for id, path in tmplims:
                varcorrid = variance_corrids[path]
                dependencies.append(varcorrid)
            dependencies = set(dependencies)

            # now that we have the dependencies we can relay the coadd job for submission
            body = json.dumps({'dependencies':dependencies, 'jobtype':'coadd', 'images':tmplims})
            tmpl_corrid = self.relay_job(body)

            template_corrids[(field, quadrant, band, ccdnum)] = tmpl_corrid

        # check to see if new coadds and subtractions are needed

        # first get all the images

    def make_coadd_bins(self, field, ccdnum, quadrant, filter):

        self._refresh_connections()

        query = 'SELECT MAXDATE FROM TEMPLATE WHERE FIELD=%s AND CCDNUM=%s AND QUADRANT=%S AND FILTER=%s ' \
                'AND PIPELINE_SCHEMA_ID=%s'
        self.cursor.execute(query, field, ccdnum, quadrant, filter, self.pipeline_schema['schema_id'])
        result = self.cursor.fetchall()

        if len(result) == 0:
            raise ValueError('No template -- can\'t make coadd bins')

        date = result[0][0]
        startsci = pd.to_datetime(date) + pd.Timedelta(self.pipeline_schema['template_science_minsep_days'],
                                                       unit='d')

        if self.pipeline_schema['rolling']:
            dates = pd.date_range(startsci, pd.to_datetime(datetime.date.today()), freq='1D')
            bins = []
            for i, date in enumerate(dates):
                if i + self.pipeline_schema['scicoadd_window_size'] >= len(dates):
                    break
                bins.append((date, dates[i + self.pipeline_schema['scicoadd_window_size']]))

        else:
            binedges = pd.date_range(startsci, pd.to_datetime(datetime.datetime.today()),
                                 freq=f'{self.pipeline_schema["scicoadd_window_size"]}')

            bins = []
            for i, lbin in enumerate(binedges[:-1]):
                bins.append((lbin, binedges[i + 1]))

        return bins

    def __del__(self):
        self.dbc.close()


if __name__ == '__main__':

    glsn_schema = {'template_minimages': 100, 'template_science_minsep_days': 30,
                   'scicoadd_window_size': 10, 'rolling':False, 'schema_id': 1}

    schemas = [glsn_schema]

    for s in schemas:
        manager = IPACQueryManager(s)
        schedule.every().day.at("06:00").do(manager)

    try:
        while True:
            schedule.run_pending()
            time.sleep(600.)  # check every 10 minutes whether to run the job
    except KeyboardInterrupt:
        pass
