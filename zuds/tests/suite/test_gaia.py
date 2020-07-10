import pytest
import requests
import numpy as np


first_ten = [1367093106839534592,
1367084830438648704,
1367084207667282944,
1367039578661441920,
1367084276386759936,
1367109255918138752,
1367070536787490176,
1367086612848924160,
1367042018204647424,
1367085272818481280]


@pytest.mark.xfail(raises=requests.exceptions.ConnectionError)
def test_get_gaia_dr2_calibrators(science_image):
    stars = science_image.gaia_dr2_calibrators()
    np.testing.assert_equal(stars[:10]['source_id'], first_ten)
    assert len(stars) == 6420
