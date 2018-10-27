curl -SL http://cdsarc.u-strasbg.fr/ftp/pub/sw/cdsclient-3.84.tar.gz \
    -o cdsclient-3.84.tar.gz \
    && tar xzf cdsclient-3.84.tar.gz \
    && cd cdsclient-3.84 \
    && CC="@CC@" CFLAGS="@CFLAGS@" \
    ./configure @CROSS@ \
    --prefix="@AUX_PREFIX@" \
    && make && make install \
    && cd .. \
    && rm -rf cdsclient*
