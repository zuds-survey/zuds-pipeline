import zuds
import copy
import yaml
import tempfile

def test_secrets():
    assert zuds.get_secret('db_host') == "localhost"
    assert zuds.get_secret('db_port') == 5432
    assert zuds.get_secret('db_username') == "admin"
    assert zuds.get_secret('db_name') == "zuds_test"
    assert zuds.get_secret('db_password') is None


def test_update():
    conf = copy.copy(zuds.get_secret.cache)
    manager = zuds.SecretManager()
    conf['db_host'] = 'abcd.efg'
    f = tempfile.TemporaryFile(mode='r+')
    yaml.dump(conf, f)
    f.seek(0)
    manager.load_config(f.name)
    assert manager.cache == conf
