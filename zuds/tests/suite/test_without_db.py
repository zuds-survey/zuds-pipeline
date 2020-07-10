import zuds
import pytest
import requests


@pytest.mark.xfail(raises=requests.exceptions.ConnectionError)
def test_image_without_database(sci_image_data_20200531):
    assert sci_image_data_20200531.unphotometered_sources == []
    assert sci_image_data_20200531.sources_contained.all() == []


@pytest.mark.xfail(raises=requests.exceptions.ConnectionError)
def test_lookup_without_database(sci_image_data_20200531):
    assert zuds.ScienceImage.get_by_basename(
        sci_image_data_20200531.basename
    ) is None


def test_source_without_database(source):
    assert source.images() == []
    assert source.best_detection is None
    assert len(source.light_curve) == 0
    assert source.unphotometered_images == []
