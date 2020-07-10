import os
import zuds
import uuid
import numpy as np
import pytest
import requests


stampcent = np.array([[153.38206, 148.80536, 147.25192, 153.63702, 152.90718, 150.19846],
                      [149.69249, 154.14828, 148.3748 , 153.9903 , 151.28311, 154.92308],
                      [154.71527, 141.91301, 147.02524, 151.24019, 148.31198, 147.89133],
                      [147.88127, 151.21399, 152.23627, 145.1139 , 149.66772, 151.06998],
                      [144.1585 , 150.05353, 154.60002, 152.49254, 144.95595, 147.17065],
                      [155.59914, 146.6712 , 143.15219, 156.93697, 150.05168, 150.3548 ]])


@pytest.mark.xfail(raises=requests.exceptions.ConnectionError)
def test_stack(sci_image_data_20200531, sci_image_data_20200601):
    images = [sci_image_data_20200531, sci_image_data_20200601]
    outdir = os.path.dirname(images[0].local_path)
    outname = os.path.join(outdir, f'{uuid.uuid4().hex}.fits')
    stack = zuds.ReferenceImage.from_images(images, outname)
    naxis1, naxis2 = stack.header['NAXIS1'], stack.header['NAXIS2']
    stamp = stack.data[naxis1 // 2 - 3:naxis1 // 2 + 3,
                       naxis2 // 2 - 3:naxis2 // 2 + 3]
    assert naxis1 == 544
    assert naxis2 == 545
    np.testing.assert_allclose(stamp, stampcent)


@pytest.mark.xfail(raises=requests.exceptions.ConnectionError)
def test_stack_input_images(sci_image_data_20200531, sci_image_data_20200601):
    images = [sci_image_data_20200531, sci_image_data_20200601]
    outdir = os.path.dirname(images[0].local_path)
    outname = os.path.join(outdir, f'{uuid.uuid4().hex}.fits')
    stack = zuds.ReferenceImage.from_images(images, outname)
    zuds.DBSession().add(stack)
    zuds.DBSession().commit()
    assert len(stack.input_images) == 2
    assert sci_image_data_20200601 in stack.input_images
    assert sci_image_data_20200531 in stack.input_images
