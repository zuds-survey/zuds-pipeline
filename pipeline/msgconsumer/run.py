import os
import pika
import psycopg2
import requests
import datetime
import logging
import json

from pika.exceptions import ConnectionClosed


# some constants
cwd = os.path.basename(__file__)
slurmd = os.path.join(cwd, '../', 'slurm')
mkcoadd_cori = os.path.join(slurmd, 'makecoadd_cori.sh')
mksub_cori = os.path.join(slurmd, 'makesub_cori.sh')
mkvar_cori = os.path.join(slurmd, 'makevariance_cori.sh')
mkcoaddsub_cori = os.path.join(slurmd, 'makecoaddsub_cori.sh')
newt_baseurl = 'https://newt.nersc.gov/newt'
nersc_username = os.getenv('NERSC_USERNAME')
nersc_password = os.getenv('NERSC_PASSWORD')
#database_uri = os.getenv('DATABASE_URI')
database_uri = 'host=db port=5432 dbname=ztfcoadd user=ztfcoadd_admin'



def is_up(host):
    """Check to see if cori or edison is up via the NEWT REST API."""

    if host not in ['cori', 'edison']:
        raise ValueError('invalid host "%s". Must be either cori or edison.' % host)

    target = os.path.join(newt_baseurl, 'status', host)

    r = requests.get(target)

    if r.status_code != 200:
        return False

    j = r.json()
    return j['status'] == 'up'


def authenticate():

    target = os.path.join(newt_baseurl, 'login')
    payload = {'username':nersc_username,
               'password':nersc_password}

    r = requests.post(target, data=payload)

    if r.status_code != 200:
        raise ValueError('Unable to Authenticate')

    return r.cookies




class TaskHandler(object):

    def __init__(self):
        self._reconnect()

    def __del__(self):
        self.connection.close()
        self.msgconnection.close()

    def close(self):
        self.__del__()

    def resolve_dependencies(self, contents, data):
        query = 'SELECT DISTINCT NERSC_ID FROM JOB WHERE CORR_ID IN %s'
        self.cursor.execute(query, (list(data['dependencies']),))
        deps = [r[0] for r in self.cursor.fetchall()]
        deps = ':'.join(deps)
        contents = contents.format(dlist=deps)
        return contents

    def submit_coadd(self, images, catalogs, obase, data, host='cori'):

        # login to nersc
        cookies = authenticate()

        # create the payload
        with open(mkcoadd_cori, 'r') as f:
            contents = f.read()
        contents = contents.replace('$1', ' '.join(images))
        contents = contents.replace('$2', ' '.join(catalogs))
        contents = contents.replace('$3', obase)
        contents = self.resolve_dependencies(contents, data)

        target = os.path.join(newt_baseurl, 'queue', host)
        payload = {'jobscript': contents}

        logging.info(payload)

        # submit the job
        r = requests.post(target, data=payload, cookies=cookies)

        # check the return code
        if r.status_code != 200:
            raise ValueError('submission did not work')

        return r.json()['jobid']

    def submit_coaddsub(self, images, catalogs, obase, data, host='cori'):

        # login to nersc
        cookies = authenticate()

        # create the payload
        with open(mkcoaddsub_cori, 'r') as f:
            contents = f.read()

        contents = contents.replace('$1', ' '.join(images))
        contents = contents.replace('$2', ' '.join(catalogs))
        contents = contents.replace('$3', obase)
        contents = self.resolve_dependencies(contents, data)

        target = os.path.join(newt_baseurl, 'queue', host)
        payload = {'jobscript': contents}

        logging.info(payload)

        # submit the job
        r = requests.post(target, data=payload, cookies=cookies)

        # check the return code
        if r.status_code != 200:
            raise ValueError('submission did not work')

        return r.json()['jobid']

    def submit_variance(self, images, masks, host='cori'):

        # login to nersc
        cookies = authenticate()

        # create the payload
        with open(mkvar_cori, 'r') as f:
            contents = f.read()

        contents = contents.replace('$1', ' '.join(images))
        contents = contents.replace('$2', ' '.join(masks))

        target = os.path.join(newt_baseurl, 'queue', host)
        payload = {'jobscript': contents}

        logging.info(payload)

        r = requests.post(target, data=payload, cookies=cookies)

        if r.status_code != 200:
            raise ValueError('submission did not work')

        return r.json()['jobid']

    def submit_sub(self, images, templates, host='cori'):

        # login to nersc
        cookies = authenticate()

        # create the payload
        with open(mksub_cori, 'r') as f:
            contents = f.read()

        contents = contents.replace('$1', ' '.join(images))
        contents = contents.replace('$2', ' '.join(templates))

        target = os.path.join(newt_baseurl, 'queue', host)
        payload = {'jobscript': contents}

        logging.info(payload)

        r = requests.post(target, data=payload, cookies=cookies)

        if r.status_code != 200:
            raise ValueError('submission did not work')

        return r.json()['jobid']

    def _reconnect(self):

        try:
            self.close()
        except AttributeError:
            pass

        # keep a connection open to the database
        self.connection = psycopg2.connect(database_uri)
        self.cursor = self.connection.cursor()

        cparams = pika.ConnectionParameters('msgqueue')
        self.msgconnection = pika.BlockingConnection(cparams)
        self.pub_channel = self.msgconnection.channel()

    def __call__(self, ch, method, properties, body):

        # messages are just lists of new images. this method
        # implements the logic to figure out what to do with
        # them

        self._reconnect()
        data = json.loads(body)

        if is_up('cori'):
            host = 'cori'
        elif is_up('edison'):
            host = 'edison'
        else:
            # send the message to the dead letter exchange where it will wait for 10 minutes
            # before attempting to be resubmitted

            query = 'UPDATE JOB SET STATUS=%s, NERSC_ID=NULL WHERE CORR_ID = %s;'
            self.cursor.execute(query, ('DEAD_LETTER', properties.correlation_id))

            return ch.basic_reject(method.delivery_tag, requeue=False)

        if data['jobtype'] == 'variance':

            new_images = data['images']
            new_masks = [p.replace('sciimg', 'mskimg') for p in new_images]
            submit_func = self.submit_variance
            args = (new_images, new_masks, host)

        elif data['jobtype'] == 'template':

            images = data['images']
            catalogs = [p.replace('fits', 'cat') for p in images]
            submit_func = self.submit_coadd
            coadd_name = data['outfile_name']
            args = (images, catalogs, coadd_name, host)

        elif data['jobtype'] == 'coaddsub':

            images = data['images']
            templates = data['templates']
            submit_func = self.submit_coaddsub
            args = (images, templates, host)

        else:
            raise ValueError('invalid task type')

        # try submitting
        try:
            jobid = submit_func(*args)
        except ValueError as e:
            logging.info(e.message)
            query = 'UPDATE JOB SET STATUS=%s, NERSC_ID=NULL WHERE CORR_ID = %s;'
            self.cursor.execute(query, ('DEAD_LETTER', properties.correlation_id))
            ch.basic_reject(method.delivery_tag, requeue=False)  # try again in 10

        else:
            query = 'UPDATE JOB SET NERSC_ID = %s, STATUS=%s, SUBMIT_TIME=%s, SYSTEM=%s WHERE CORR_ID = %s;'
            self.cursor.execute(query, (jobid, 'PENDING', datetime.datetime.now(),
                                        host, properties.correlation_id))
            ch.basic_ack(method.delivery_tag)
            self.pub_channel.basic_publish(exchange='',
                                           routing_key='monitor',
                                           properties=properties,
                                           body=body)

        self.connection.commit()



if __name__ == '__main__':

    # set up a connection to the message queue
    while True:
        try:
            cparams = pika.ConnectionParameters('msgqueue')
            connection = pika.BlockingConnection(cparams)
        except ConnectionClosed:
            pass
        else:
            break

    channel = connection.channel()

    # consume stuff from the queue
    handler = TaskHandler()
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(handler, queue='jobs')
    channel.start_consuming()
