import os
import pytest
import pathlib
from zuds.tests.fixtures import (ScienceImageFactory,
                                 TriangulumScienceImageFactory,
                                 TMP_DIR)
import uuid
import zuds
import shutil


# initialize the database
print('Loading test configuration from test_config.yaml')
basedir = pathlib.Path(os.path.dirname(__file__))
config = basedir / '../config/test.conf.yaml'
target = pathlib.Path(TMP_DIR) / config.name
shutil.copy(config, target)
os.chmod(target, 0o700)
zuds.load_config(target)
zuds.create_database()
zuds.init_db()

# ensure the database is empty when the test suite starts
zuds.drop_tables()
zuds.create_tables()


@pytest.fixture
def science_image():
    return ScienceImageFactory(basename=uuid.uuid4().hex)


@pytest.fixture
def triangulum_science_image():
    return TriangulumScienceImageFactory(basename=uuid.uuid4().hex)


