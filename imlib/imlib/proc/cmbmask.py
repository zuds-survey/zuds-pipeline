__all = ['cmbmask']

def cmbmask(mask1, mask2, maskout):

    from astropy.io import fits as afits

    """Combine two masks and write out the result to a fits file.


    Arguments
    ---------

    mask1: Path to the first mask

    mask2: Path to the second mask

    maskout: Path to the output mask
    """

    d1 = afits.open(mask1)[0].data
    d2 = afits.open(mask2)[0].data

    # combine the masks
    d3 = d1 | d2

    outmaskhdu = afits.PrimaryHDU(d3)
    omhdul = afits.HDUList([outmaskhdu])
    omhdul.writeto(maskout)
