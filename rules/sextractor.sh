curl -SL http://www.astromatic.net/download/sextractor/sextractor-2.19.5.tar.gz \
    -o sextractor-2.19.5.tar.gz \
    && tar xzf sextractor-2.19.5.tar.gz \
    && cd sextractor-2.19.5 \
    && patch -p1 < ../rules/patch_sextractor \
    && chmod +x autogen.sh \
    && ./autogen.sh \
    && CC="@CC@" CFLAGS="@CFLAGS@" CPPFLAGS="-I@BLAS_INCLUDE@" ./configure @CROSS@ \
    --with-lapacke="@LAPACK@ @BLAS@" --prefix="@AUX_PREFIX@" \
    && make && make install \
    && cd .. \
    && rm -rf sextractor*
