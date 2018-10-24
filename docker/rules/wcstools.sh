curl -SL http://tdc-www.harvard.edu/software/wcstools/wcstools-3.9.5.tar.gz \
    -o wcstools-3.9.5.tar.gz \
    && tar xzf wcstools-3.9.5.tar.gz \ 
    && cd wcstools-3.9.5 \
    && chmod -R u+w . \
    && CC="@CC@" CFLAGS="@CFLAGS@" \
    CPPFLAGS="-I@AUX_PREFIX@/include" \
    LDFLAGS="-L@AUX_PREFIX@/lib" \
    ./configure @CROSS@ \
    --disable-fortran \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf wcstools*
