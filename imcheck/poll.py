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
import grequests  # for asynchronous requests


import numpy as np
import pandas as pd
from ztfquery import query as zq
from astropy.time import Time


ipac_root = 'http://irsa.ipac.caltech.edu/'
formula = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{year:s}/{month:s}{day:s}/{fracday:s}' \
          '/ztf_{filefracday:s}_{paddedfield:s}_{filtercode:s}_c{paddedccdid:s}_{imgtypecode:s}_q{qid:d}_sciimg.fits'
nersc_formula = '/global/cscratch1/sd/dgold/ztfcoadd/{paddedfield:s}/c{paddedccdid:s}/{qid:d}/{filtercode:s}/{fname:s}'
nersc_tmpform = '/global/cscratch1/sd/dgold/ztfcoadd/templates/{fname:s}'
tmp_basename_form = '{paddedfield:s}_c{paddedccdid:s}_{qid:d}_{filtercode:s}_{mindate:s}_{maxdate:s}_ztf_deepref.fits'
newt_baseurl = 'https://newt.nersc.gov/newt'
variance_batchsize = 1024
date_start = datetime.date(2018, 2, 16)

# this is the night id corresponding
# to the first science observation in the survey
first_nid = 411
ipac_query_window = 30  # days

# number of data transfer nodes to use to pull new data
ndtn = 4

# secrets
database_uri = os.getenv('DATABASE_URI')
ipac_username = os.getenv('IPAC_USERNAME')
ipac_password = os.getenv('IPAC_PASSWORD')
nersc_username = os.getenv('NERSC_USERNAME')
nersc_password = os.getenv('NERSC_PASSWORD')


# recursively partition an iterable into subgroups (py3 compatible)
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
    _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


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
        self._refresh_connections()

    def __del__(self):
        self.dbc.close()

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

            ipaths.append(tpath)
            npaths.append(npath)

        return np.asarray(ipaths), np.asarray(npaths)

    def _refresh_connections(self):

        try:
            self.connection.close()
        except NameError:
            pass

        try:
            self.dbc.close()
        except NameError:
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
        query = 'SELECT ID, PATH, OBSJD, HASVARIANCE FROM IMAGE WHERE FIELD=%s AND ' \
                'CCDNUM=%s AND QUADRANT=%s AND FILTER=%s ' \
                'AND GOOD=TRUE ORDER BY OBSJD ASC LIMIT %s'
        self.cursor.execute(query, field, ccdnum, quadrant, filter,
                            self.pipeline_schema['template_minimages'])
        result = self.cursor.fetchall()
        return list(zip(*result))

    def retrieve_new_image_paths_and_metadata(self):

        # query ipac and return the images that are in its database but not in ours

        td = datetime.date.today()
        zquery = zq.ZTFQuery()

        # need to go one month at a time to avoid overloading the ipac server
        ndays = (td - date_start).days
        curnid = first_nid + ndays

        nids = list(range(first_nid, curnid, ipac_query_window))

        if nids[-1] < curnid:
            nids.append(nids[-1] + ipac_query_window)

        # these are the r/l limits of each group in the nid-partitioned queries
        nidbins = []
        for i, n in enumerate(nids[:-1]):
            right = nids[i + 1]
            nidbins.append([n, right])

        for i, (left, right) in enumerate(nidbins[1:]):
            nidbins[i][0] = left + 1

        tab = []
        for left, right in nidbins:
            zquery.load_metadata(sql_query=' NID BETWEEN %d AND %d' % (left, right),
                                 auth=[ipac_username, ipac_password])
            df = zquery.metatable
            tab.append(df)

        # this is a table with the metadata of all the ZTF images that IPAC has
        tab = pd.concat(tab)

        # here are all their paths
        ipaths, npaths = self._generate_paths(tab)

        # now we need to prune it down to just the ones that we don't have

        # first get all the paths that we have
        query = 'SELECT PATH FROM IMAGE'
        self.cursor.execute(query)
        result = [p[0] for p in self.cursor.fetchall()]

        # now calculate the ones we dont have
        new = np.setdiff1d(npaths, result, assume_unique=True)

        # get rid of this for memory purposes
        del result

        # get the indices of the new images
        inds = np.argwhere(npaths[:, None] == new[None, :])
        inds = np.sum(inds, axis=0)[:, 0]

        # get the ipac paths
        ipaths = ipaths[inds]

        # now return
        return ipaths, new, tab.iloc[inds]

    def download_images(self, npaths, ipaths):

        # with the paths generated now we can make the download commands
        tasks = list(zip(ipaths, npaths))

        # first we must partition the tasks into separate iterables for each dtn
        spltasks = _split(tasks, ndtn)

        # store the asynchronous http requests here
        async_requests = []
        for i, (ipc, npc) in enumerate(spltasks):

            icookies = ipac_authenticate()  # get a different cookie for each DTN
            sessionid = icookies.get('sessionid')

            # authenticate to nersc
            ncookies = nersc_authenticate()

            download_script = [f'curl {ipath} --create-dirs -o {npath} --cookie "SESSIONID={sessionid}"'
                               for ipath, npath in zip(ipc, npc)]

            # upload the download script to NERSC

            download_script = '\n'.join(download_script)
            path = f'/global/cscratch1/sd/dgold/ztfcoadd/download_scripts/{jd}_{i+1}.sh'
            target = os.path.join(newt_baseurl, 'file', f'dtn{i+1:02d}', f'{path}')
            requests.put(target, data=download_script, cookies=ncookies)

            # now prepare the arguments of the multithreaded call to make the download
            target = os.path.join(newt_baseurl, 'command', f'dtn{i+1:02d}')
            payload = {'executable': f'/bin/bash {path}', 'loginenv':True}

            # now call the download asynchronously using grequests
            request = grequests.post(target, data=payload, cookies=ncookies)
            async_requests.append(request)

        # this directs all the DTNs to execute their downloads
        responses = grequests.map(async_requests)

        # check that everything worked as expected
        for r in responses:

            if r.status_code != 200:
                raise RuntimeError('Error contacting NEWT')

            jr = r.json()

            if jr['status'] == 'ERROR':
                error = jr['error']
                raise RuntimeError(f'Error on dtn02: {error}')
            else:
                logging.info(jr['output'])

    def update_database_with_new_images(self, npaths, metatable):

        # now update the database with the new images
        columns = '(PATH,FILTER,QUADRANT,FIELD,CCDNUM,PID,OBSJD,RA,' \
                  'DEC,INFOBITS,RCID,FID,PID,NID,EXPID,ITID,OBSDATE,SEEING,AIRMASS,MOONILLF' \
                  'MOONESB,MAGLIMIT,CRPIX1,CRPIX2,CRVAL1,CRVAL2,CD11,CD12,CD21,CD22,RA1' \
                  'DEC1,RA2,DEC2,RA3,DEC3,RA4,DEC4,IPAC_PUB_DATE,IPAC_GID)'

        dbq = f'INSERT INTO IMAGE {columns} VALUES (%s)'
        dbq = dbq % ','.join(['%s'] * len(columns.split(',')))

        # programatically access some rows of the table
        dfkey = [k.lower() for k in columns[1:-1].split(',')][5:]

        for (i, row), npath in zip(metatable.iterrows(), npaths):
            self.cursor.execute(dbq, npath, row['filtercode'][1:],
                                row['qid'], row['field'], row['ccdid'],
                                *row[dfkey].tolist())

        self.dbc.commit()

    def determine_and_relay_variance_jobs(self, npaths):
        # everything needs variance. submit batches of 1024

        variance_corrids = {}

        nvariance_jobs = len(npaths) // variance_batchsize + (1 if len(npaths) % variance_batchsize > 0 else 0)
        for i in range(nvariance_jobs):

            batch = npaths[i * variance_batchsize:(i + 1) * variance_batchsize]
            body = json.dumps({'jobtype':'variance', 'dependencies':None, 'images': batch})

            # send a message to the job submission script telling it to submit the job
            correlation_id = self.relay_job(body)

            for path in batch:
                variance_corrids[path] = correlation_id

        return variance_corrids

    def determine_and_relay_template_jobs(self, variance_corrids, metatable):

        # check to see if new templates are needed
        template_corrids = {}
        for (field, quadrant, band, ccdnum), group in metatable.groupby(['field',
                                                                         'qid',
                                                                         'filtercode',
                                                                         'ccdid']):

            ofield, oquadrant, oband, occdnum = field, quadrant, band, ccdnum

            # convert into proper values
            field = int(field)
            quadrant = int(quadrant)
            band = band[1:]
            ccdnum = int(ccdnum)

            # check if a template is needed
            if not self.needs_template(field, ccdnum, quadrant, band):
                continue

            tmplims, tmplids, jds, hasvar = self.create_template_image_list(field, ccdnum, quadrant, band)

            minjd = np.min(jds)
            maxjd = np.max(jds)

            mintime = Time(minjd, format='jd', scale='utc')
            maxtime = Time(maxjd, format='jd', scale='utc')

            mindatestr = mintime.iso.split()[0].replace('-', '')
            maxdatestr = maxtime.iso.split()[1].replace('-', '')

            # see what jobs need to finish before this one can run
            dependencies = []
            remake_variance = []
            for path, hv in zip(tmplims, hasvar):
                if not hv:
                    if path in variance_corrids:
                        varcorrid = variance_corrids[path]
                        dependencies.append(varcorrid)
                    else:
                        remake_variance.append(path)

            if len(remake_variance) > 0:
                moredeps = self.determine_and_relay_variance_jobs(remake_variance)
                dependencies.extend(moredeps.values())

            dependencies = set(dependencies)

            # what will this template be called?
            tmpbase = tmp_basename_form.format(paddedfield=ofield,
                                               qid=oquadrant,
                                               paddedccdid=occdnum,
                                               filtercode=oband,
                                               mintime=mindatestr,
                                               maxtime=maxdatestr)

            outfile_name = nersc_tmpform.format(fname=tmpbase)

            # now that we have the dependencies we can relay the coadd job for submission
            body = json.dumps({'dependencies':dependencies, 'jobtype':'template', 'images':tmplims,
                               'outfile_name': outfile_name, 'imids': tmplids, 'quadrant': quadrant,
                               'field': field, 'ccdnum': ccdnum, 'mindate':mindatestr,
                               'maxdate':maxdatestr, 'filter': band,
                               'pipeline_schema_id': self.pipeline_schema['schema_id'],
                               })
            tmpl_corrid = self.relay_job(body)
            template_corrids[(field, quadrant, band, ccdnum)] = (tmpl_corrid, outfile_name)

        return template_corrids

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

    def evaluate_coadd_and_sub(self, field, ccdnum, quadrant, band, binl, binr):

        query = 'SELECT ID FROM COADD WHERE FIELD=%s AND CCDNUM=%s AND QUADRANT=%s AND FILTER=%s ' \
                'ORDER BY PROC_DATE DESC LIMIT 1'
        self.cursor.execute(query, field, ccdnum, quadrant, band, binl, binr)
        cid = self.cursor.fetchone()[0]

        query = 'SELECT IMAGE_ID FROM COADDIMAGEASSOC WHERE COADD_ID=%s'
        self.cursor.execute(query, cid)
        imids = [p[0] for p in self.cursor.fetchall()]

        # now see what should be in the coadd
        query = 'SELECT ID, PATH, HAS_VARIANCE FROM IMAGE WHERE FIELD=%s AND CCDNUM=%s AND QUADRANT=%s AND FILTER=%s ' \
                'AND GOOD=TRUE'
        self.cursor.exceute(query, field, ccdnum, quadrant, band)
        oughtids, oughtpaths, hasvar = list(zip(*self.cursor.fetchall()))
        diff = np.setxor1d(oughtids, imids, assume_unique=True)

        return len(diff) > 0, oughtids, oughtpaths, hasvar

    def get_latest_template(self, field, ccdnum, quadrant, filter):
        query = 'SELECT ID, PATH FROM TEMPLATE WHERE FIELD=%s AND CCDNUM=%s ' \
                'AND QUADRANT=%s AND FILTER=%s ORDER BY PROCDATE DESC'
        self.cursor.execute(query, field, ccdnum, quadrant, filter)
        id, path = self.cursor.fetchone()
        return id, path

    def determine_and_relay_coaddsub_jobs(self, variance_corrids, template_corrids, metatable):

        for (field, ccdnum, quadrant, filter), group in metatable.groupby(['field', 'ccdid',
                                                                           'qid', 'filtercode']):

            # convert into proper values
            field = int(field)
            quadrant = int(quadrant)
            band = filter[1:]
            ccdnum = int(ccdnum)

            # get the bins
            bins = self.make_coadd_bins(field, ccdnum, quadrant, band)

            for bin in bins:
                runjob, ids, paths, hasvar = self.evaluate_coadd_and_sub(field, ccdnum, quadrant, band, *bin)

                if runjob:

                    dependencies = []
                    if (field, quadrant, band, ccdnum) in template_corrids:
                        corrid, tmplpath = template_corrids[(field, quadrant, band, ccdnum)]
                        dependencies.append(corrid)
                    else:
                        _, tmplpath = self.get_latest_template(field, ccdnum, quadrant, filter)

                    # build up dependencies

                    dependencies = []
                    remake_variance = []
                    for path, hv in zip(paths, hasvar):
                        if not hv:
                            if path in variance_corrids:
                                varcorrid = variance_corrids[path]
                                dependencies.append(varcorrid)
                            else:
                                remake_variance.append(path)

                    if len(remake_variance) > 0:

                        moredeps = self.determine_and_relay_variance_jobs(remake_variance)
                        dependencies.extend(moredeps.values())

                    dependencies = set(dependencies)

                    data = {'jobtype': 'coaddsub', 'field': field, 'ccdnum': ccdnum,
                            'quadrant': quadrant, 'mindate': bin[0], 'maxdate': bin[1],
                            'images': paths, 'template': tmplpath, 'filter':band,
                            'pipeline_schema_id': self.pipeline_schema['schema_id'],
                            'dependencies': dependencies}

                    body = json.dumps(data)
                    self.relay_job(body)


    def __call__(self):

        # refresh our connections
        self._refresh_connections()

        # get the new image paths and metadata
        npaths, ipaths, metatable = self.retrieve_new_image_paths_and_metadata()

        # download the images
        self.download_images(npaths, ipaths)

        # update the database with the new images that were downloaded
        self.update_database_with_new_images(npaths, metatable)

        # now actually determine the new jobs

        # batch dispatch variance map making
        variance_corrids = self.determine_and_relay_variance_jobs(npaths)

        # then make any new templates that are needed
        template_corrids = self.determine_and_relay_template_jobs(variance_corrids, metatable)

        # finally coadd the science frames and make the corresponding subtractions
        self.determine_and_relay_coaddsub_jobs(variance_corrids, template_corrids, metatable)


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
