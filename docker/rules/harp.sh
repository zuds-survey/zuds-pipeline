curl -SL https://github.com/tskisner/HARP/releases/download/v1.0.5/harp-1.0.5.tar.bz2 \
    -o harp-1.0.5.tar.bz2 \
    && tar xjf harp-1.0.5.tar.bz2 \
    && cd harp-1.0.5 \
    && CC="@CC@" CXX="@CXX@" CFLAGS="@CFLAGS@" CXXFLAGS="@CXXFLAGS@" ./configure @CROSS@ \
    --disable-mpi --disable-python \
    --with-cfitsio="@AUX_PREFIX@" \
    --with-boost="@AUX_PREFIX@" \
    --with-blas="@LAPACK@ @BLAS@" \
    --prefix=@AUX_PREFIX@ \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf harp*
