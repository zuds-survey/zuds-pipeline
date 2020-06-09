import time
import textwrap
import subprocess
import sqlalchemy as sa

from .core import DBSession, Base
from .status import status
from .env import DependencyError, output
from .secrets import get_secret

from distutils.version import LooseVersion as Version

__all__ = ['drop_tables', 'create_tables', 'check_postgres_extensions',
           'init_db', 'create_database']


def run(cmd):
    return subprocess.run(cmd,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          shell=True)


def drop_tables():
    conn = DBSession.session_factory.kw['bind']
    print(f'Dropping tables on database {conn.url.database}')
    meta = sa.MetaData()
    meta.reflect(bind=conn)
    meta.drop_all(bind=conn)


def create_tables(retry=5):
    """
    Create tables for all models, retrying 5 times at intervals of 3
    seconds if the database is not reachable.
    """
    for i in range(1, retry + 1):
        try:
            conn = DBSession.session_factory.kw['bind']
            print(f'Creating tables on database {conn.url.database}')
            Base.metadata.create_all()

            print('Refreshed tables:')
            for m in Base.metadata.tables:
                print(f' - {m}')

            return

        except Exception as e:
            if (i == retry):
                raise e
            else:
                print('Could not connect to database...sleeping 3')
                print(f'  > {e}')
                time.sleep(3)


def check_postgres_extensions(deps, username, password, host, port, database):

    psql_cmd = f'psql '
    flags = f'-U {username} '

    if password:
        psql_cmd = f'PGPASSWORD="{password}" {psql_cmd}'
    flags += f' --no-password'

    if host:
        flags += f' -h {host}'

    if port:
        flags += f' -p {port}'

    def get_version(v):
        lines = v.split('\n')
        for i, line in enumerate(lines):
            if '1 row' in line.strip():
                return lines[i - 1].strip()

    fail = []
    for dep, min_version in deps:

        query = f'{dep} >= {min_version}'
        clause = f"SELECT max(extversion) FROM pg_extension WHERE extname = '{dep}';"
        cmd = psql_cmd + f' {flags} {database} -c "{clause}"'

        try:
            with status(query):
                success, out = output(cmd, shell=True)
                if not success:
                    raise ValueError(out.decode("utf-8").strip())
                try:
                    version = get_version(out.decode('utf-8').strip())
                    print(f'[{version.rjust(8)}]'.rjust(40 - len(query)),
                          end='')
                except:
                    raise ValueError('Could not parse version')

                if not (Version(version) >= Version(min_version)):
                    raise RuntimeError(
                        f'Required {min_version}, found {version}'
                    )
        except ValueError:
            print(
                f'\n[!] Sorry, but our script could not parse the output of '
                f'`{cmd.replace(password, "***") if password else cmd}`; '
                f'please file a bug, or see `zuds/core.py`\n'
            )
            raise
        except Exception as e:
            fail.append((dep, e, cmd, min_version))

    if fail:
        failstr = ''
        for (pkg, exc, cmd, min_version) in fail:
            repcmd = cmd
            if password is not None:
                repcmd = repcmd.replace(password, '***')
            failstr += f'    - {pkg}: `{repcmd}`\n'
            failstr += '     ' + str(exc).replace(password, '***') + '\n'

        msg = f'''
[!] Some system dependencies seem to be unsatisfied

The failed checks were:

{failstr}
'''
        raise DependencyError(msg)


def init_db(timeout=None):

    username = get_secret('db_username')
    password = get_secret('db_password')
    port = get_secret('db_port')
    host = get_secret('db_host')
    dbname = get_secret('db_name')

    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(username, password or '', host or '', port or '', dbname)

    kwargs = {}
    if timeout is not None:
        kwargs['connect_args'] = {"options": f"-c statement_timeout={timeout}"}

    print(f'Checking for postgres extensions:')
    deps = [('q3c', '1.5.0')]
    try:
        check_postgres_extensions(deps, username, password, host, port, dbname)
    except DependencyError:
        DBSession.remove()
        raise

    conn = sa.create_engine(url, client_encoding='utf8', **kwargs)
    DBSession.configure(bind=conn)
    Base.metadata.bind = conn


def create_database(force=False):
    db = get_secret('db_name')
    user = get_secret('db_username')
    host = get_secret('db_host')
    port = get_secret('db_port')
    password = get_secret('db_password')

    psql_cmd = 'psql'
    flags = f'-U {user}'

    if password:
        psql_cmd = f'PGPASSWORD="{password}" {psql_cmd}'
    flags += f' --no-password'

    if host:
        flags += f' -h {host}'

    if port:
        flags += f' -p {port}'

    def test_db(database):
        test_cmd = f"{psql_cmd} {flags} -c 'SELECT 0;' {database}"
        p = run(test_cmd)

        try:
            with status('Testing database connection'):
                if not p.returncode == 0:
                    raise RuntimeError()
        except:
            print(textwrap.dedent(
                f'''
                 !!! Error accessing database:
                 The most common cause of database connection errors is a
                 misconfigured `pg_hba.conf`.
                 We tried to connect to the database with the following parameters:
                   database: {db}
                   username: {user}
                   host:     {host}
                   port:     {port}
                 The postgres client exited with the following error message:
                 {'-' * 78}
                 {p.stderr.decode('utf-8').strip()}
                 {'-' * 78}
                 Please modify your `pg_hba.conf`, and use the following command to
                 check your connection:
                   {test_cmd}
                '''))

            raise


    plat = run('uname').stdout
    if b'Darwin' in plat:
        print('* Configuring MacOS postgres')
        sudo = ''
    else:
        print('* Configuring Linux postgres [may ask for sudo password]')
        sudo = 'sudo -u postgres'

    # Ask for sudo password here so that it is printed on its own line
    # (better than inside a `with status` section)
    run(f'{sudo} echo -n')

    with status(f'Creating user {user}'):
        run(f'{sudo} createuser --superuser {user}')

    if force:
        try:
            with status('Removing existing database'):
                p = run(f'{sudo} dropdb {db}')
                if p.returncode != 0:
                    raise RuntimeError()
        except:
            print('Could not delete database: \n\n'
                  f'{textwrap.indent(p.stderr.decode("utf-8").strip(), prefix="  ")}\n')
            raise

    try:
        with status(f'Creating database'):
            p = run(f'{sudo} createdb {db}')
            msg = f'{textwrap.indent(p.stderr.decode("utf-8").strip(), prefix="  ")}\n'
            if p.returncode != 0 and 'already exists' not in msg:
                raise RuntimeError()

            p = run(f'{sudo} psql -c "GRANT ALL PRIVILEGES ON DATABASE {db} TO {user};" {db}')
            msg = f'{textwrap.indent(p.stderr.decode("utf-8").strip(), prefix="  ")}\n'
            if p.returncode != 0:
                raise RuntimeError()

            p = run(f'{sudo} psql -c "ALTER USER {user} WITH PASSWORD \'{password}\';" {db}')
            msg = f'{textwrap.indent(p.stderr.decode("utf-8").strip(), prefix="  ")}\n'
            if p.returncode != 0:
                raise RuntimeError()

    except:
        print(f'Could not create database: \n\n{msg}\n')
        raise

    try:
        with status(f'Creating extensions'):
            p = run(f'{sudo} psql -c "CREATE EXTENSION q3c" {db}')
            msg = f'{textwrap.indent(p.stderr.decode("utf-8").strip(), prefix="  ")}\n'
            if p.returncode != 0 and 'already exists' not in msg:
                raise RuntimeError()
    except:
        print(f'Could not create extensions: \n\n{msg}\n')
        raise

    test_db(db)

