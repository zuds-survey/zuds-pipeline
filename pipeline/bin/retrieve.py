
import os
import psycopg2
import pandas as pd
from argparse import ArgumentParser
from subprocess import check_call

class HPSSDB(object):

    def __init__(self):

        dbname = os.getenv('HPSS_DBNAME')
        password = os.getenv('HPSS_DBPASSWORD')
        username = os.getenv('HPSS_DBUSERNAME')
        port = os.getenv('HPSS_DBPORT')
        host = os.getenv('HPSS_DBHOST')
        dsn = f'host={host} user={username} password={password} dbname={dbname} port={port}'

        self.connection = psycopg2.connect(dsn)
        self.cursor = self.connection.cursor()

    def __del__(self):

        del self.cursor
        del self.connection


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--whereclause", required=True, default=None, type=str,
                        help='SQL where clause that tells the program which images to '
                             'retrieve.')
    parser.add_argument("--output-dir", default=None, help='Store all files flattened in this output directory.')
    parser.add_argument('--exclude-masks', default=False, action='store_true', help='Only retrieve the science '
                                                                                    'images.')
    args = parser.parse_args()

    # interface to HPSS and database
    hpssdb = HPSSDB()

    # this is the query to get the image paths
    query = f'SELECT PATH, HPSS_SCI_PATH, HPSS_MASK_PATH FROM IMAGE WHERE {args.whereclause}'
    hpssdb.cursor.execute(query)
    results = hpssdb.cursor.fetchall()
    df = pd.DataFrame(results, columns=['path', 'hpss_sci_path', 'hpss_mask_path'])

    # first retrieve the science images
    for tarname, group in df.groupby('hpss_sci_path'):

        if args.output_dir is None:
            syscall = f'htar xf {tarname} '
            for i, row in group.iterrows():
                syscall += f'/global/project/projectdirs/ptf/ztf/data/xfer/*/*/*/*/*/{row["path"]}'
            print(syscall, flush=True)
            check_call(syscall.split())
        else:
            for i, row in group.iterrows():
                syscall = f'htar xf {tarname} -O /global/project/projectdirs/ptf/ztf/data/xfer/*/*/*/*/*/{row["path"]} ' \
                          f'> {os.path.join(args.output_dir, row["path"])}'

                print(syscall, flush=True)
                check_call(syscall.split())

    if not args.exclude_masks:
        for tarname, group in df.groupby('hpss_mask_path'):

            if args.output_dir is None:
                syscall = f'htar xf {tarname} '
                for i, row in group.iterrows():
                    syscall += f'/global/project/projectdirs/ptf/ztf/data/xfer/*/*/*/*/*/{row["path"].replace("sciimg", "mskimg")}'

                print(syscall, flush=True)
                check_call(syscall.split())
            else:
                for i, row in group.iterrows():
                    syscall = f'htar xf {tarname} -O /global/project/projectdirs/ptf/ztf/data/xfer/*/*/*/*/*/{row["path"].replace("sciimg", "mskimg")} ' \
                              f'> {os.path.join(args.output_dir, row["path"].replace("sciimg", "mskimg"))}'

                    print(syscall, flush=True)
                    check_call(syscall.split())
