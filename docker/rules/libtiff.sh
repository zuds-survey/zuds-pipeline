curl -SL http://download.osgeo.org/libtiff/tiff-4.0.7.tar.gz \
    -o tiff-4.0.7.tar.gz \
    && tar xzf tiff-4.0.7.tar.gz \
    && cd tiff-4.0.7 \
    && CC="@CC@" CXX="@CXX@" CFLAGS="@CFLAGS@" CXXFLAGS="@CXXFLAGS@" ./configure @CROSS@ \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf tiff*
