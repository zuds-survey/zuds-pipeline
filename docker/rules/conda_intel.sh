conda config --add channels intel \
    && conda install --copy --yes $(echo intelpython@PYVERSION@_core | sed -e "s#\.._#_#") \
    && rm -rf @CONDA_PREFIX@/pkgs/*
