from .status import status
import subprocess
from distutils.version import LooseVersion as Version

__all__ = ['check_dependencies', 'DependencyError', 'output']


def output(cmd, shell=False):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=shell)
    out, err = p.communicate()
    success = (p.returncode == 0)
    return success, out


class DependencyError(Exception):
    pass


def check_dependencies(deps):
    """
    Check that the executable dependencies specified by `deps` are installed
    on the user's system, displaying a check status to the user. If the
    dependencies are not installed, raise a DependencyError.

    :param deps: dict


    """

    print('Checking system dependencies:')

    fail = []
    for dep, (cmd, get_version, min_version) in deps.items():
        try:
            query = f'{dep} >= {min_version}'
            with status(query):
                success, out = output(cmd)
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
                f'`{" ".join(cmd)}`; please file a bug, or see `zuds/env.py`\n'
            )
            raise
        except Exception as e:
            fail.append((dep, e))

    if fail:
        failstr = ''
        for (pkg, exc) in fail:
            cmd, get_version, min_version = deps[pkg]
            failstr += f'    - {pkg}: `{" ".join(cmd)}`\n'
            failstr += '     ' + str(exc) + '\n'

        msg = f'''
[!] Some system dependencies seem to be unsatisfied'

The failed checks were:

{failstr}
'''
        raise DependencyError(msg)

