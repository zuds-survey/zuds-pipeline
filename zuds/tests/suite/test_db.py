
from zuds import DBSession

def test_science_image_modified(science_image):
    science_image.seeing = 2.3
    DBSession().add(science_image)
    DBSession().commit()
    modified = science_image.modified
    science_image.basename = 'abcd'
    DBSession().add(science_image)
    DBSession().commit()
    new_modified = science_image.modified
    assert new_modified > modified


