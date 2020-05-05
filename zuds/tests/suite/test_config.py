import zuds

def test_secrets():
    assert zuds.get_secret('db_host') == "localhost"
    assert zuds.get_secret('db_port') == 5432
    assert zuds.get_secret('db_username') == "admin"
    assert zuds.get_secret('db_name') == "zuds_test"
    assert zuds.get_secret('db_password') is None
