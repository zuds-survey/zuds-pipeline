conda install --copy --yes \
    nose \
    requests \
    future \
    cython \
    numpy \
    scipy \
    matplotlib=2.1.2 \
    basemap \
    seaborn \
    pyyaml \
    astropy=2 \
    hdf5 \
    h5py \
    psutil \
    ephem \
    psycopg2 \
    pytest \
    pytest-cov \
    numba \
    sqlalchemy \
    scikit-learn \
    scikit-image \
    ipython \
    jupyter \
    ipywidgets=6.0.0 \
    bokeh \
    && mplrc="@CONDA_PREFIX@/lib/python@PYVERSION@/site-packages/matplotlib/mpl-data/matplotlibrc"; \
    cat ${mplrc} | sed -e "s#^backend.*#backend : TkAgg#" > ${mplrc}.tmp; \
    mv ${mplrc}.tmp ${mplrc} \
    && rm -rf @CONDA_PREFIX@/pkgs/*
