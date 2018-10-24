curl -SL https://github.com/dstndstn/astrometry.net/archive/0.74.tar.gz -o astrometry.net-0.74.tar.gz \
    && tar xzf astrometry.net-0.74.tar.gz \
    && cd astrometry.net-0.74 \
    && CC="@CC@" CXX="@CXX@" CFLAGS="@CFLAGS@" CXXFLAGS="@CXXFLAGS@" \
    LDFLAGS="-L@AUX_PREFIX@/lib -lz" LDSHARED="@CC@ -shared" \
    WCSLIB_INC="-I@AUX_PREFIX@/include/wcslib" \
    WCSLIB_LIB="-L@AUX_PREFIX@/lib -lwcs" \
    JPEG_INC="-I@AUX_PREFIX@/include" \
    JPEG_LIB="-L@AUX_PREFIX@/lib -ljpeg" \
    CFITS_INC="-I@AUX_PREFIX@/include" \
    CFITS_LIB="-L@AUX_PREFIX@/lib -lcfitsio -lm" make \
    && CC="@CC@" CXX="@CXX@" CFLAGS="@CFLAGS@" CXXFLAGS="@CXXFLAGS@" \
    LDFLAGS="-L@AUX_PREFIX@/lib -lz" LDSHARED="@CC@ -shared" \
    WCSLIB_INC="-I@AUX_PREFIX@/include/wcslib" \
    WCSLIB_LIB="-L@AUX_PREFIX@/lib -lwcs" \
    JPEG_INC="-I@AUX_PREFIX@/include" \
    JPEG_LIB="-L@AUX_PREFIX@/lib -ljpeg" \
    CFITS_INC="-I@AUX_PREFIX@/include" \
    CFITS_LIB="-L@AUX_PREFIX@/lib -lcfitsio -lm" make py \
    && CC="@CC@" CXX="@CXX@" CFLAGS="@CFLAGS@" CXXFLAGS="@CXXFLAGS@" \
    LDFLAGS="-L@AUX_PREFIX@/lib -lz" LDSHARED="@CC@ -shared" \
    WCSLIB_INC="-I@AUX_PREFIX@/include/wcslib" \
    WCSLIB_LIB="-L@AUX_PREFIX@/lib -lwcs" \
    JPEG_INC="-I@AUX_PREFIX@/include" \
    JPEG_LIB="-L@AUX_PREFIX@/lib -ljpeg" \
    CFITS_INC="-I@AUX_PREFIX@/include" \
    CFITS_LIB="-L@AUX_PREFIX@/lib -lcfitsio -lm" make extra \
    && CC="@CC@" CXX="@CXX@" CFLAGS="@CFLAGS@" CXXFLAGS="@CXXFLAGS@" \
    LDFLAGS="-L@AUX_PREFIX@/lib -lz" LDSHARED="@CC@ -shared" \
    WCSLIB_INC="-I@AUX_PREFIX@/include/wcslib" \
    WCSLIB_LIB="-L@AUX_PREFIX@/lib -lwcs" \
    JPEG_INC="-I@AUX_PREFIX@/include" \
    JPEG_LIB="-L@AUX_PREFIX@/lib -ljpeg" \
    CFITS_INC="-I@AUX_PREFIX@/include" \
    CFITS_LIB="-L@AUX_PREFIX@/lib -lcfitsio -lm" \
    make install INSTALL_DIR="@AUX_PREFIX@" \
    && cd .. \
    && rm -rf astrometry*
