import zuds
import numpy as np
import pytest
import requests


@pytest.mark.xfail(raises=requests.exceptions.ConnectionError)
def test_seeing(sci_image_data_20200531):
    zuds.estimate_seeing(sci_image_data_20200531)
    np.testing.assert_allclose(sci_image_data_20200531.header['SEEING'], 2.004896)

