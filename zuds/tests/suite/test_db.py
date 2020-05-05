
from zuds import DBSession

def test_science_image_modified(science_image):
    science_image.seeing = 2.3
    DBSession().add(science_image)
    DBSession().flush()
    modified = science_image.modified
    science_image.maglimit = 20.3
    DBSession().add(science_image)
    DBSession().flush()
    new_modified = science_image.modified
    assert new_modified > modified




