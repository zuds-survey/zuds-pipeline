curl -SL https://github.com/dstndstn/tractor/archive/dr7.0.tar.gz \
    -o tractor-dr7.0.tar.gz \
    && tar xzf tractor-dr7.0.tar.gz \
    && cd tractor-dr7.0 && patch -p1 < ../rules/patch_tractor \
    && CERES_LIB_DIR="@AUX_PREFIX@/lib" \
    CC="@CC@" CFLAGS="@CFLAGS@" LDSHARED="@CC@ -shared" \
    CXX="@CXX@" CXXFLAGS="@CXXFLAGS@" BLAS_LIB="@BLAS@" \
    python setup.py install --with-ceres --prefix="@AUX_PREFIX@" \
    && cd .. \
    && rm -rf tractor*
