curl -SL http://prdownloads.sourceforge.net/libpng/libpng-1.6.29.tar.gz?download \
    -o libpng-1.6.29.tar.gz \
    && tar xzf libpng-1.6.29.tar.gz \
    && cd libpng-1.6.29 \
    && CC="@CC@" CFLAGS="@CFLAGS@" LDFLAGS="-L@AUX_PREFIX@/lib -lz" \
    ./configure @CROSS@ \
    --with-zlib-prefix="@AUX_PREFIX@" \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf libpng*
