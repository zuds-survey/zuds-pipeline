import zuds


def test_science_image_modified(science_image):
    db = zuds.DBSession()
    science_image.seeing = 2.3
    db.add(science_image)
    db.commit()
    modified = science_image.modified
    science_image.basename = 'abcd'
    db.add(science_image)
    db.commit()
    new_modified = science_image.modified
    assert new_modified > modified


