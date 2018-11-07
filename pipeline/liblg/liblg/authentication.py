import os
import requests

newt_baseurl = 'https://newt.nersc.gov/newt'
nersc_username = os.getenv('NERSC_USERNAME')
nersc_password = os.getenv('NERSC_PASSWORD')

ipac_root = 'http://irsa.ipac.caltech.edu/'
ipac_username = os.getenv('IPAC_USERNAME')
ipac_password = os.getenv('IPAC_PASSWORD')


def ipac_authenticate():
    target = os.path.join(ipac_root, 'account', 'signon', 'login.do')
    payload = {'username': ipac_username,
               'password': ipac_password}

    r = requests.post(target, data=payload)

    if r.status_code != 200:
        raise ValueError('Unable to Authenticate')

    if r.cookies.get('SESSIONID') is None:
        raise ValueError('Unable to login to IPAC - bad credentials')

    return r.cookies


def nersc_authenticate():
    target = os.path.join(newt_baseurl, 'login')
    payload = {'username': nersc_username,
               'password': nersc_password}

    r = requests.post(target, data=payload)

    if r.status_code != 200:
        raise ValueError('Unable to Authenticate')

    rj = r.json()

    if not rj['auth']:
        raise ValueError('Unable to login to NERSC - bad credentials')

    return r.cookies