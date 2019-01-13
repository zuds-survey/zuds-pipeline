import os
import sys
import pika
import datetime
import psycopg2
import requests
import logging
import uuid
import json
import paramiko
import threading
import random


import numpy as np
import pandas as pd
from ztfquery import query as zq
from astropy.time import Time
from pika.exceptions import ConnectionClosed
from liblg import ipac_authenticate, nersc_authenticate, nersc_username, nersc_password


ipac_root = 'https://irsa.ipac.caltech.edu/'
formula = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{year:s}/{month:s}{day:s}/{fracday:s}' \
          '/ztf_{filefracday:s}_{paddedfield:s}_{filtercode:s}_c{paddedccdid:s}_{imgtypecode:s}_q{qid:d}_sciimg.fits'
nersc_formula = '/global/cscratch1/sd/dgold/ztfcoadd/science_frames/{paddedfield:s}/c{paddedccdid:s}/' \
                '{qid:d}/{filtercode:s}/{fname:s}'
nersc_tmpform = '/global/cscratch1/sd/dgold/ztfcoadd/templates/{fname:s}'


tmp_basename_form = '{paddedfield:06d}_c{paddedccdid:02d}_{qid:d}_' \
                    '{filtercode:s}_{mindate:s}_{maxdate:s}_ztf_deepref.fits'
coadd_basename_form = '{paddedfield:06d}_c{paddedccdid:02d}_{qid:d}_{filtercode:s}' \
                      '_{mindate:s}_{maxdate:s}_ztf_coadd.fits'

manifest = '/global/cscratch1/sd/dgold/ztfcoadd/download_scripts/.manifest'
lockfile = '/global/cscratch1/sd/dgold/ztfcoadd/download_scripts/.lockfile'

newt_baseurl = 'https://newt.nersc.gov/newt'
variance_batchsize = 1024
sub_batchsize = 32
date_start = datetime.date(2018, 2, 16)
n_concurrent_requests = 50
coadd_minframes = 3
img_batchsize = 1024
template_batchsize = 4

# this is the night id corresponding
# to the first science observation in the survey
first_nid = 411
ipac_query_window = 30  # days

# number of data transfer nodes to use to pull new data
ndtn = 4

# secrets
#database_uri = os.getenv('DATABASE_URI')
database_uri = 'host=db port=5432 dbname=ztfcoadd user=ztfcoadd_admin'
ipac_username = os.getenv('IPAC_USERNAME')
ipac_password = os.getenv('IPAC_PASSWORD')

outlock = threading.Lock()


# recursively partition an iterable into subgroups (py3 compatible)
_split = lambda iterable, n: [iterable[:len(iterable)//n]] + \
    _split(iterable[len(iterable)//n:], n - 1) if n != 0 else []


# make this a static method for threading purposes
def execute_download_on_dtn(self, download_script_lines, host):

    n_transactions = len(download_script_lines) // img_batchsize + \
                     (0 if len(download_script_lines) % img_batchsize == 0 else 1)

    # authenticate to nersc
    ncookies = nersc_authenticate()

    for i in range(n_transactions):

        # label the download
        now = datetime.datetime.utcnow()

        download_script = '\n'.join(download_script_lines[i * img_batchsize : (i + 1) * img_batchsize])
        path = f'global/cscratch1/sd/dgold/ztfcoadd/download_scripts/{host}_{now}.sh'
        path = path.replace(' ', '_')
        target = os.path.join(newt_baseurl, 'file', host, path)
        requests.put(target, data=download_script, cookies=ncookies)

        download_script_npath = f'/{path}'

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=nersc_username, password=nersc_password)
        stdin, stdout, stderr = ssh.exec_command(f'/bin/bash {download_script_npath}')
        exitcode = stdout.channel.recv_exit_status()
        errlines = stderr.readlines()

        if exitcode != 0:
            raise RuntimeError(errlines)

        ssh.close()

    with outlock:
        self.logger.info(f'Download complete on {host}.')


class IPACQueryManager(object):

    def __init__(self, pipeline_schema, logger):
        self.pipeline_schema = pipeline_schema
        self.logger = logger
        self._refresh_connections()

    def __del__(self):
        self.dbc.close()
        self.connection.close()

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
        except AttributeError:
            pass

        try:
            self.dbc.close()
        except AttributeError:
            pass

        while True:
            try:
                params = pika.ConnectionParameters('msgqueue')
                self.connection = pika.BlockingConnection(params)
            except ConnectionClosed:
                pass
            else:
                break

        # set up the work queue and the dead letter exchange for delayed requeueing

        job_channel = self.connection.channel()
        delay_channel = self.connection.channel()

        delay_channel.exchange_declare(exchange='jobs-retry',
                                       exchange_type='direct')
        job_channel.exchange_declare(exchange_type='direct',
                                     exchange='jobs')

        job_channel.queue_declare(queue='jobs', arguments={
            'x-dead-letter-exchange': 'jobs-retry',
        })
        job_channel.queue_bind(queue='jobs',
                               exchange='jobs')

        delay_channel.queue_declare(queue='jobs-retry', arguments={
            'x-message-ttl': 300000,  # wait five minutes
            'x-dead-letter-exchange': 'jobs',
        })
        delay_channel.queue_bind(exchange='jobs-retry',
                                 queue='jobs-retry')

        # cache the channels
        self.job_channel = job_channel
        self.delay_channel = delay_channel

        # connect to the database

        while True:
            try:
                self.dbc = psycopg2.connect(database_uri)
            except psycopg2.DatabaseError:
                pass
            else:
                break

        self.cursor = self.dbc.cursor()

    def list_current_template_images(self, field, ccdnum, quadrant, filter):

        # list images that are in the template
        tquery = 'SELECT ID FROM TEMPLATE WHERE FIELD=%s AND CCDNUM=%s AND QUADRANT=%s AND FILTER=%s ' \
                 'AND PIPELINE_SCHEMA_ID=%s'
        self.cursor.execute(tquery, (filter, quadrant, field, ccdnum, self.pipeline_schema['schema_id']))
        result = self.cursor.fetchall()

        if len(result) == 0:
            return []

        else:
            tid = result[0][0]
            iquery = 'SELECT I.PATH FROM  TEMPLATEIMAGEASSOC TA JOIN IMAGE I ON TA.IMAGE_ID = I.ID ' \
                     'WHERE TA.TEMPLATE_ID=%s'
            self.cursor.execute(iquery, (tid,))
            images = [r[0] for r in self.cursor.fetchall()]
            return images

    def needs_template(self, field, ccdnum, quadrant, filter):
        query = 'SELECT COUNT(*) FROM TEMPLATE WHERE FIELD=%s AND CCDNUM=%s AND QUADRANT=%s AND FILTER=%s ' \
                'AND PIPELINE_SCHEMA_ID=%s'
        self.cursor.execute(query, (field, ccdnum, quadrant, filter, self.pipeline_schema['schema_id']))
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
        self.cursor.execute(query, (corr_id, jobtype))
        self.dbc.commit()

        return corr_id

    def create_template_image_list(self, field, ccdnum, quadrant, filter):
        query = 'SELECT ID, PATH, OBSJD, HASVARIANCE FROM IMAGE WHERE FIELD=%s AND ' \
                'CCDNUM=%s AND QUADRANT=%s AND FILTER=%s ' \
                'AND GOOD=TRUE ORDER BY OBSJD ASC LIMIT %s'
        self.cursor.execute(query, (field, ccdnum, quadrant, filter,
                            self.pipeline_schema['template_minimages']))
        result = self.cursor.fetchall()
        return list(zip(*result))

    def retrieve_new_image_paths_and_metadata(self):

        # query ipac and return the images that are in its database but not in ours

        # refresh our connections
        self._refresh_connections()

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
            nidbins[i + 1][0] = left + 1

        tab = []
        self.logger.info(f'nidbins is {nidbins}')
        for left, right in nidbins:
            zquery.load_metadata(sql_query=' NID BETWEEN %d AND %d AND (FIELD=847 AND CCDID=2 AND QID=3)'% (left, right),
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
        inds = np.argwhere(npaths[:, None] == new[None, :])[:, 0]
        self.logger.info(f'Inds are: {inds}')

        # get the ipac paths
        ipaths = ipaths[inds]

        # now return
        return ipaths, new, tab.iloc[inds]

    def reset_manifest(self):
        # reset the manifest
        ncookies = nersc_authenticate()
        target = f'{newt_baseurl}/file/dtn01/{manifest[1:]}'
        payload = b''
        requests.put(target, data=payload, cookies=ncookies)

    def download_images(self, npaths, ipaths):

        # refresh our connections
        self._refresh_connections()

        # with the paths generated now we can make the download commands
        tasks = list(zip(ipaths, npaths))

        # first we must partition the tasks into separate iterables for each dtn
        spltasks = _split(tasks, ndtn)

        # use this to name download scripts
        td = datetime.date.today()
        dt = datetime.datetime(td.year, td.month, td.day, 0, 0, 0)
        at = Time(dt)
        jd = at.jd

        # store the asynchronous http requests here
        threads = []
        for i, l in enumerate(spltasks):

            ipc, npc = list(zip(*l))

            icookies = ipac_authenticate()  # get a different cookie for each DTN
            sessionid = icookies.get('JOSSO_SESSIONID')

            download_script = [f'if curl {ipath} --create-dirs -o {npath} --cookie "JOSSO_SESSIONID={sessionid}"; then'
                               f'\n( flock -x 200; echo {npath} >> {manifest} ) 200> {lockfile}; fi'
                               for ipath, npath in zip(ipc, npc)]
            mask_download_script = [p.replace('sciimg', 'mskimg') for p in download_script]
            sub_download_script = [p.replace('sciimg.fits', 'scimrefdiffimg.fits.fz') for p in download_script]
            download_script.extend(mask_download_script)
            download_script.extend(sub_download_script)
            random.shuffle(download_script)

            print(len(download_script))

            # upload the download script to NERSC
            host = f'dtn{i+1:02d}.nersc.gov'

            # make the multithreaded call to do the download
            thread = threading.Thread(target=execute_download_on_dtn, args=(self, download_script, host))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def read_manifest(self):
        ncookies = nersc_authenticate()
        target = f'{newt_baseurl}/file/dtn01/{manifest[1:]}?view=read'
        r = requests.get(target, cookies=ncookies)
        return list(filter(lambda s: 'msk' not in s and 'scimref' not in s,
                           r.content.decode('utf-8').strip().split('\n')))

    def read_sub_manifest(self):
        ncookies = nersc_authenticate()
        target = f'{newt_baseurl}/file/dtn01/{manifest[1:]}?view=read'
        r = requests.get(target, cookies=ncookies)
        return list(filter(lambda s: 'msk' not in s and 'sciimg' not in s,
                           r.content.decode('utf-8').strip().split('\n')))

    def update_database_with_new_images(self, npaths, metatable):

        # refresh our connections
        self._refresh_connections()

        # now update the database with the new images
        columns = '(PATH,FILTER,QUADRANT,FIELD,CCDNUM,PID,OBSJD,RA,' \
                  'DEC,INFOBITS,RCID,FID,NID,EXPID,ITID,OBSDATE,SEEING,AIRMASS,MOONILLF,' \
                  'MOONESB,MAGLIMIT,CRPIX1,CRPIX2,CRVAL1,CRVAL2,CD11,CD12,CD21,CD22,RA1,' \
                  'DEC1,RA2,DEC2,RA3,DEC3,RA4,DEC4,IPAC_GID)'

        dbq = f'INSERT INTO IMAGE {columns} VALUES (%s)'
        dbq = dbq % ','.join(['%s'] * len(columns.split(',')))

        # programatically access some rows of the table
        dfkey = [k.lower() for k in columns[1:-1].split(',')][5:]

        for (i, row), npath in zip(metatable.iterrows(), npaths):
            self.cursor.execute(dbq, ((npath, row['filtercode'][1:],
                                    row['qid'], row['field'], row['ccdid']) +
                                    tuple(row[dfkey].tolist())))

        query = 'UPDATE IMAGE SET GOOD=FALSE WHERE INFOBITS != 0 OR SEEING > 3. OR ' \
                '((FILTER=\'r\' OR FILTER=\'g\') AND MAGLIMIT < 19) OR (FILTER=\'i\' AND MAGLIMIT < 18.5)'
        self.cursor.execute(query)

        self.dbc.commit()

    def determine_and_relay_variance_jobs(self, npaths):
        # everything needs variance. submit batches of 1024

        # refresh our connections
        self._refresh_connections()

        variance_corrids = {}

        nvariance_jobs = len(npaths) // variance_batchsize + (1 if len(npaths) % variance_batchsize > 0 else 0)

        for i in range(nvariance_jobs):

            batch = npaths[i * variance_batchsize:(i + 1) * variance_batchsize]

            if isinstance(batch, np.ndarray):
                batch = batch.tolist()

            body = json.dumps({'jobtype':'variance', 'dependencies':None, 'images': batch})

            # send a message to the job submission script telling it to submit the job
            correlation_id = self.relay_job(body)

            for path in batch:
                variance_corrids[path] = correlation_id

        return variance_corrids

    def determine_and_relay_template_jobs(self, variance_corrids, metatable):

        # refresh our connections
        self._refresh_connections()

        # check to see if new templates are needed
        template_corrids = {}

        batch = []

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

            tmplids, tmplims, jds, hasvar = self.create_template_image_list(field, ccdnum, quadrant, band)

            if len(tmplids) < self.pipeline_schema['template_minimages']:
                # not enough images to make a template -- try again some other time
                continue

            minjd = np.min(jds)
            maxjd = np.max(jds)

            mintime = Time(minjd, format='jd', scale='utc')
            maxtime = Time(maxjd, format='jd', scale='utc')

            mindatestr = mintime.iso.split()[0].replace('-', '')
            maxdatestr = maxtime.iso.split()[0].replace('-', '')

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

            dependencies = list(set(dependencies))

            # what will this template be called?
            tmpbase = tmp_basename_form.format(paddedfield=field,
                                               qid=oquadrant,
                                               paddedccdid=ccdnum,
                                               filtercode=oband,
                                               mindate=mindatestr,
                                               maxdate=maxdatestr)

            outfile_name = nersc_tmpform.format(fname=tmpbase)

            # now that we have the dependencies we can relay the coadd job for submission
            tmpl_data = {'dependencies':dependencies, 'jobtype':'template', 'images':tmplims,
                         'outfile_name': outfile_name, 'imids': tmplids, 'quadrant': quadrant,
                         'field': field, 'ccdnum': ccdnum, 'mindate':mindatestr,
                         'maxdate':maxdatestr, 'filter': band,
                         'pipeline_schema_id': self.pipeline_schema['schema_id']}

            batch.append(tmpl_data)

            if len(batch) == template_batchsize:
                payload = {'jobtype': 'template', 'jobs': batch}
                body = json.dumps(payload)
                tmpl_corrid = self.relay_job(body)
                for d in batch:
                    template_corrids[(d['field'], d['quadrant'], d['filter'], d['ccdnum'])] = (tmpl_corrid, d)
                batch = []

        if len(batch) > 0:
            payload = {'jobtype': 'template', 'jobs': batch}
            body = json.dumps(payload)
            tmpl_corrid = self.relay_job(body)
            for d in batch:
                template_corrids[(d['field'], d['quadrant'], d['filter'], d['ccdnum'])] = (tmpl_corrid, d)

        return template_corrids

    def make_coadd_bins(self, field, ccdnum, quadrant, filter, maxdate=None):

        self._refresh_connections()

        if maxdate is None:
            query = 'SELECT MAXDATE FROM TEMPLATE WHERE FIELD=%s AND CCDNUM=%s AND QUADRANT=%s AND FILTER=%s ' \
                    'AND PIPELINE_SCHEMA_ID=%s'
            self.cursor.execute(query, (field, ccdnum, quadrant, filter, self.pipeline_schema['schema_id']))
            result = self.cursor.fetchall()

            if len(result) == 0:
                raise ValueError('No template queued or on disk -- can\'t make coadd bins')
            date = result[0][0]

        else:
            date = maxdate

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
                                     freq=f'{self.pipeline_schema["scicoadd_window_size"]}D')

            bins = []
            for i, lbin in enumerate(binedges[:-1]):
                bins.append((lbin, binedges[i + 1]))

        return bins

    def evaluate_coadd_and_sub(self, field, ccdnum, quadrant, band, binl, binr):

        # see what should be in the coadd
        query = 'SELECT ID, PATH, HASVARIANCE FROM IMAGE WHERE FIELD=%s AND CCDNUM=%s AND QUADRANT=%s AND FILTER=%s ' \
                'AND GOOD=TRUE AND OBSDATE BETWEEN %s AND %s'
        self.cursor.execute(query, (field, ccdnum, quadrant, band, binl, binr))
        result = self.cursor.fetchall()
        if len(result) == 0:
            return False, [], [], []
        else:
            oughtids, oughtpaths, hasvar = list(zip(*result))

        # see what is in the coadd
        query = 'SELECT ID FROM COADD WHERE FIELD=%s AND CCDNUM=%s AND QUADRANT=%s AND FILTER=%s ' \
                'AND MINDATE=%s AND MAXDATE=%s ORDER BY PROCDATE DESC LIMIT 1'
        self.cursor.execute(query, (field, ccdnum, quadrant, band, binl, binr))

        try:
            cid = self.cursor.fetchone()[0]
        except TypeError:  # there is no previous coadd
            return len(oughtids) >= coadd_minframes, oughtids, oughtpaths, hasvar

        query = 'SELECT IMAGE_ID FROM COADDIMAGEASSOC WHERE COADD_ID=%s'
        self.cursor.execute(query, (cid,))
        imids = [p[0] for p in self.cursor.fetchall()]

        diff = np.setxor1d(oughtids, imids, assume_unique=True)

        return len(diff) > 0 and len(oughtids) >= coadd_minframes, oughtids, oughtpaths, hasvar

    def get_latest_template(self, field, ccdnum, quadrant, filter):
        query = 'SELECT PATH FROM TEMPLATE WHERE FIELD=%s AND CCDNUM=%s ' \
                'AND QUADRANT=%s AND FILTER=%s ORDER BY PROCDATE DESC'
        self.cursor.execute(query, (field, ccdnum, quadrant, filter))

        pathtup = self.cursor.fetchone()
        try:
            return pathtup[0]
        except TypeError:  # it's none
            return None

    def determine_and_relay_coaddsub_jobs(self, variance_corrids, template_corrids, metatable):

        # refresh our connections
        self._refresh_connections()

        batch = []
        correlation_ids = []

        for (field, ccdnum, quadrant, filter), group in metatable.groupby(['field', 'ccdid',
                                                                           'qid', 'filtercode']):

            # convert into proper values
            field = int(field)
            quadrant = int(quadrant)
            band = filter[1:]
            ccdnum = int(ccdnum)

            dependencies = []
            if (field, quadrant, band, ccdnum) in template_corrids:
                corrid, tmplbody = template_corrids[(field, quadrant, band, ccdnum)]
                tmplpath = tmplbody['outfile_name']
                maxdate = pd.to_datetime(tmplbody['maxdate'])
                dependencies.append(corrid)
            else:
                tmplpath = self.get_latest_template(field, ccdnum, quadrant, band)
                maxdate = None

            if tmplpath is not None:
                # get the bins
                bins = self.make_coadd_bins(field, ccdnum, quadrant, band, maxdate=maxdate)

                for bin in bins:
                    runjob, ids, paths, hasvar = self.evaluate_coadd_and_sub(field, ccdnum, quadrant, band, *bin)

                    if runjob:

                        # build up dependencies

                        my_dependencies = []
                        remake_variance = []
                        for path, hv in zip(paths, hasvar):
                            if not hv:
                                if path in variance_corrids:
                                    varcorrid = variance_corrids[path]
                                    my_dependencies.append(varcorrid)
                                else:
                                    remake_variance.append(path)

                        if len(remake_variance) > 0:

                            moredeps = self.determine_and_relay_variance_jobs(remake_variance)
                            my_dependencies.extend(moredeps.values())

                        my_dependencies = list(set(my_dependencies))
                        my_dependencies.extend(dependencies)

                        mindatestr = bin[0].strftime('%Y%m%d')
                        maxdatestr = bin[1].strftime('%Y%m%d')

                        outfile_name = coadd_basename_form.format(
                            paddedfield=field,
                            paddedccdid=ccdnum,
                            qid=quadrant,
                            filtercode=filter,
                            mindate=mindatestr,
                            maxdate=maxdatestr,
                        )

                        outfile_name = nersc_formula.format(
                            paddedfield='%06d' % field,
                            paddedccdid='%02d' % ccdnum,
                            qid=quadrant,
                            filtercode=filter,
                            fname=outfile_name
                        )

                        data = {'jobtype': 'coaddsub', 'field': field, 'ccdnum': ccdnum,
                                'quadrant': quadrant, 'mindate': mindatestr, 'maxdate': maxdatestr,
                                'images': paths, 'template': tmplpath, 'filter':band,
                                'pipeline_schema_id': self.pipeline_schema['schema_id'],
                                'dependencies': my_dependencies, 'outfile_name': outfile_name,
                                'imids': ids}

                        batch.append(data)

                        if len(batch) == sub_batchsize:
                            packet = {'jobtype': 'coaddsub', 'jobs': batch}
                            body = json.dumps(packet)
                            correlation_ids.append(self.relay_job(body))
                            batch = []

        if len(batch) > 0:
            packet = {'jobtype': 'coaddsub', 'jobs': batch}
            body = json.dumps(packet)
            correlation_ids.append(self.relay_job(body))

        return correlation_ids

    def prune_metatable(self, old_npaths, new_npaths, metatable):

        inds = []
        olist = old_npaths.tolist()
        for path in new_npaths:
            index = olist.index(path)
            inds.append(index)
        return metatable.iloc[inds]

    def determine_and_relay_forcephoto_jobs(self, coaddsub_corrids, metatable):

        subs = _split(metatable['path'], 64)
        batch = []
        for images in subs:
            data = {'jobtype': 'forcephoto', 'images': images, 'dependencies': coaddsub_corrids}
            batch.append(data)

        packet = {'jobtype': 'forcephoto', 'jobs': batch}
        body = json.dumps(packet)
        self.relay_job(body)


    def __call__(self):

        self.logger.info('Reconnecting to database and message queue...')
        self.logger.info('Connection successful.')

        self.logger.info('Retrieving new image paths and metadata...')
        # get the new image paths and metadata
        ipaths, npaths, metatable = self.retrieve_new_image_paths_and_metadata()
        self.logger.info(f'{len(npaths)} new images found.')
        self.logger.debug(f'Metatable is: {metatable}')

        if len(npaths) > 0:
            # download the images
            #self.logger.info(f'Downloading {len(npaths)} images on {ndtn} data transfer nodes...')
            #self.reset_manifest()
            #self.download_images(npaths, ipaths)

            new_npaths = self.read_manifest()
            sub_npaths = self.read_sub_manifest()

            sub_allnpaths = np.asarray([p.replace('sciimg', 'scimrefdiffimg')
                                         .replace('fits', 'fits.fz') for p in npaths])

            mtcopy = metatable.copy()
            mtcopy['path'] = sub_allnpaths

            sub_metatable = self.prune_metatable(sub_allnpaths, sub_npaths, mtcopy)
            metatable = self.prune_metatable(npaths, new_npaths, metatable)

            npaths = new_npaths

            # update the database with the new images that were downloaded
            self.update_database_with_new_images(npaths, metatable)

            # now actually determine the new jobs

            # batch dispatch variance map making
            variance_corrids = self.determine_and_relay_variance_jobs(npaths)

            # then make any new templates that are needed
            template_corrids = self.determine_and_relay_template_jobs(variance_corrids, metatable)

            # finally coadd the science frames and make the corresponding subtractions
            coaddsub_corrids = self.determine_and_relay_coaddsub_jobs(variance_corrids, template_corrids, metatable)

            # lastly do forced photometry on the detected objects
            self.determine_and_relay_forcephoto_jobs(coaddsub_corrids, sub_metatable)

            self.__del__()


if __name__ == '__main__':

    glsn_schema = {'template_minimages': 100, 'template_science_minsep_days': 10,
                   'scicoadd_window_size': 10, 'rolling':False, 'schema_id': 1}

    schemas = [glsn_schema]

    logger = logging.getLogger('poll')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    logger.addHandler(ch)

    for s in schemas:
        manager = IPACQueryManager(s, logger)
        manager()


    """
        schedule.every().day.at("06:00").do(manager)

    try:
        while True:
            schedule.run_pending()
            time.sleep(600.)  # check every 10 minutes whether to run the job
    except KeyboardInterrupt:
        pass
    """
