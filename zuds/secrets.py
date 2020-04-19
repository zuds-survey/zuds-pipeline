import os
import yaml
import stat
from pathlib import Path

__all__ = ['get_secret']

class SecretsFilePermissionError(Exception):
    pass

class SecretManager(object):
    """Class that manages access to secrets from the ZUDS configuration /
    secrets file. Maintains a cache of secrets and implements the
    `get_secret` function (as `__call__`) that is used elsewhere in
    the codebase."""

    def __init__(self, debug=False):
        """Read the configuration file and construct the secrets cache."""
        
        # See if an alternative path to the secrets file is specified
        # using the environment variable 'ZUDS_CONFIG'
        self.config_path = os.getenv('ZUDS_CONFIG')

        # if no config_path is specified via an environment variable,
        # use the default (~/.zuds)
        if self.config_path is None:

            # use the default
            self.config_path = Path(os.getenv('HOME')) / '.zuds'

        # check access to the file
        bits = os.stat(self.config_path)
        msgbase = '''Secrets file "%s" is accessible to %s, 
                     but must disallow access to group and world 
                     for security. Please run $chmod go-rwx %s 
                     on the command line and retry.'''

        # check for group access
        if bits & stat.S_IRWXG > 0:
            raise SecretsFilePermissionError(msg % (self.config_path,
                                                    'group',
                                                    self.config_path))

        # check for world access
        if bits & stat.S_IRWXO > 0:
            raise SecretsFilePermissionError(msg % (self.config_path,
                                                    'world',
                                                    self.config_path))
            
        # load the secrets
        self.cache = yaml.load(open(self.config_path, 'r'), 
                               Loader=yaml.FullLoader)

    def __call__(self, key):
        if key in self.cache:
            value = self.cache[key]
            if value is None:
                raise ValueError(f'Secret value for key "{key}" is undefined.')
        else:
            raise KeyError(f'Nonexistent secret requested: "{key}".')

get_secret = SecretManager()
