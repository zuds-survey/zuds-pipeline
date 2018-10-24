curl -SL http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz \
    -o metis-5.1.0.tar.gz \
    && tar xzf metis-5.1.0.tar.gz \
    && cd metis-5.1.0 && patch -p1 < ../rules/patch_metis \
    && cd .. \
    && mkdir metis_build && cd metis_build \
    && cmake \
    -D CMAKE_C_COMPILER="@CC@" \
    -D CMAKE_C_FLAGS="@CFLAGS@" \
    -D GKLIB_PATH="../metis-5.1.0/GKlib" \
    -D CMAKE_INSTALL_PREFIX="@AUX_PREFIX@" \
    ../metis-5.1.0 \
    && make \
    && make install \
    && cd .. \
    && rm -rf metis*
