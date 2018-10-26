import subprocess

__all__ = ['execute']


def execute(cmd, capture=True, merge=False):
    """Execute a shell command, log the stdout and stderr, and check
    the return code. If the return code is != 0, raise an
    exception."""


    args = cmd.split()
    process = subprocess.Popen(args, stdout=subprocess.PIPE if capture else None,
                               stderr=subprocess.STDOUT if merge else subprocess.PIPE)
    stdout, stderr = process.communicate()

    # check return code
    if process.returncode != 0:
        raise Exception(stderr)

    return stdout, stderr
