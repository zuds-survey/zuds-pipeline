curl -SL https://github.com/tskisner/healpix-autotools/releases/download/v3.31.4/healpix-autotools-3.31.4.tar.bz2 \
    | tar -xjf - \
    && cd healpix-autotools-3.31.4 \
    && CC="@CC@" CXX="@CXX@" FC="@FC@" \
    CFLAGS="@CFLAGS@" CXXFLAGS="@CXXFLAGS@" FCFLAGS="@FCFLAGS@" \
    ./configure @CROSS@ --with-cfitsio="@AUX_PREFIX@" --prefix="@AUX_PREFIX@" \
    && make && make install \
    && cd .. \
    && rm -rf healpix*
