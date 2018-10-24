curl -SL http://www.astromatic.net/download/psfex/psfex-3.17.1.tar.gz \
    -o psfex-3.17.1.tar.gz \
    && tar xzf psfex-3.17.1.tar.gz \
    && cd psfex-3.17.1 \
    && patch -p1 < ../rules/patch_psfex \
    && chmod +x autogen.sh \
    && ./autogen.sh \
    && CC="@CC@" CFLAGS="@CFLAGS@" CPPFLAGS="-I@BLAS_INCLUDE@" ./configure @CROSS@ \
    --with-lapacke="@LAPACK@ @BLAS@" --prefix="@AUX_PREFIX@" \
    && make && make install \
    && cd .. \
    && rm -rf psfex*
