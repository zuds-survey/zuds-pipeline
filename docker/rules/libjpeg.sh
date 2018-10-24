curl -SL http://sourceforge.net/projects/libjpeg/files/libjpeg/6b/jpegsrc.v6b.tar.gz/download \
    -o jpegsrc.v6b.tar.gz \
    && tar xzf jpegsrc.v6b.tar.gz \
    && cd jpeg-6b \
    && CC="@CC@" CFLAGS="@CFLAGS@" ./configure @CROSS@ \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 \
    && mkdir -p "@AUX_PREFIX@/man/man1" \
    && make install \
    && make install-lib \
    && make install-headers \
    && cd .. \
    && rm -rf jpeg*
