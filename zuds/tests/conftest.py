import os
import pytest
import pathlib
from zuds.tests.fixtures import (ScienceImageFactory,
                                 TriangulumScienceImageFactory)
import uuid
import zuds


# initialize the database
print('Loading test configuration from test_config.yaml')
basedir = pathlib.Path(os.path.dirname(__file__))
zuds.load_config(basedir / '../config/test.config.yaml')
zuds.init_db()

# make sure the database is fresh
zuds.drop_tables()
zuds.create_tables()

@pytest.fixture
def science_image():
    return ScienceImageFactory(basename=uuid.uuid4().hex)


@pytest.fixture
def triangulum_science_image():
    return TriangulumScienceImageFactory(basename=uuid.uuid4().hex)


