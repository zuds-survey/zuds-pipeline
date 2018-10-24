curl -SL http://www.astromatic.net/download/swarp/swarp-2.38.0.tar.gz \
    -o swarp-2.38.0.tar.gz \
    && tar xzf swarp-2.38.0.tar.gz \
    && cd swarp-2.38.0 \
    && CC="@CC@" CFLAGS="@CFLAGS@" \
    ./configure @CROSS@ \
    --prefix="@AUX_PREFIX@" \
    --enable-threads \
    && make && make install \
    && cd .. \
    && rm -rf swarp*
