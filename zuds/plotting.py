import numpy as np

from astropy.wcs import WCS


from .constants import CMAP_RANDOM_SEED
from .catalog import PipelineRegionFile

__all__ = ['show_images', 'discrete_cmap', 'plot_triplet']


def plot_triplet(tr):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 2), dpi=100)
    ax = fig.add_subplot(131)
    ax.axis('off')
    ax.imshow(tr[:, :, 0], origin='upper', cmap=plt.cm.bone)
    ax2 = fig.add_subplot(132)
    ax2.axis('off')
    ax2.imshow(tr[:, :, 1], origin='upper', cmap=plt.cm.bone)
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    ax3.imshow(tr[:, :, 2], origin='upper', cmap=plt.cm.bone)
    plt.show()


def discrete_cmap(ncolors):
    """Create a ListedColorMap with `ncolors` randomly-generated colors
    that can be used to color an IntegerFITSImage.

    The first color in the list is always black."""
    from matplotlib import colors

    prng = np.random.RandomState(CMAP_RANDOM_SEED)
    h = prng.uniform(low=0.0, high=1.0, size=ncolors)
    s = prng.uniform(low=0.2, high=0.7, size=ncolors)
    v = prng.uniform(low=0.5, high=1.0, size=ncolors)
    hsv = np.dstack((h, s, v))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))
    rgb[0] = (0.,) * 3
    return colors.ListedColormap(rgb)


def show_images(image_or_images, catalog=None, titles=None, reproject=False,
                ds9=False, figsize='auto'):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    imgs = np.atleast_1d(image_or_images)
    n = len(imgs)

    if ds9:
        if catalog is not None:
            reg = PipelineRegionFile.from_catalog(catalog)
        cmd = '%ds9 -zscale '
        for img in imgs:
            img.save()
            cmd += f' {img.local_path}'
            if catalog is not None:
                cmd += f' -region {reg.local_path}'
        cmd += ' -lock frame wcs'
        print(cmd)
    else:

        if titles is not None and len(titles) != n:
            raise ValueError('len(titles) != len(images)')

        ncols = min(n, 3)
        nrows = (n - 1) // 3 + 1

        align_target = imgs[0]
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharex=True,
                               sharey=True,
                               subplot_kw={
                                   'projection': WCS(
                                       align_target.astropy_header)
                               } if reproject else None,
                               figsize=None if figsize == 'auto' else figsize)

        ax = np.atleast_1d(ax)
        for a in ax.ravel()[n:]:
            a.set_visible(False)


        for i, (im, a) in enumerate(zip(imgs, ax.ravel())):
            im.show(a, align_to=align_target if reproject else None)
            if catalog is not None:
                filtered = 'GOODCUT' in catalog.data.dtype.names
                my_xy = im.wcs.all_world2pix(list(zip(catalog.data[
                                                          'X_WORLD'],
                                                      catalog.data[
                                                          'Y_WORLD'
                                                      ])), 0)
                for row, xy in zip(catalog.data, my_xy):

                    e = Ellipse(xy=xy,
                                width=6 * row['A_IMAGE'],
                                height=6 * row['B_IMAGE'],
                                angle=row['THETA_IMAGE'] * 180. / np.pi)
                    e.set_facecolor('none')

                    if filtered and row['GOODCUT'] == 1:
                        e.set_edgecolor('lime')
                    else:
                        e.set_edgecolor('red')
                    a.add_artist(e)

            if titles is not None:
                a.set_title(titles[i])

        fig.tight_layout()
        fig.show()
        return fig
