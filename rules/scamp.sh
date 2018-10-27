curl -SL http://www.astromatic.net/download/scamp/scamp-2.0.4.tar.gz \
    -o scamp-2.0.4.tar.gz \
    && tar xzf scamp-2.0.4.tar.gz \
    && cd scamp-2.0.4 \
    && patch -p1 < ../rules/patch_scamp \
    && chmod +x autogen.sh \
    && ./autogen.sh \
    && CC="@CC@" CFLAGS="@CFLAGS@" CPPFLAGS="-I@BLAS_INCLUDE@" ./configure @CROSS@ \
    --with-lapacke="@LAPACK@ @BLAS@" --prefix="@AUX_PREFIX@" \
    && make && make install \
    && cd .. \
    && rm -rf scamp*
