curl -SL https://github.com/rmjarvis/tmv/archive/v0.75.tar.gz \
    -o tmv-0.75.tar.gz \
    && tar xzf tmv-0.75.tar.gz \
    && cd tmv-0.75 \
    && scons PREFIX="@AUX_PREFIX@" \
    CXX="@CXX@" FLAGS="@CXXFLAGS@" \
    EXTRA_INCLUDE_PATH="@BLAS_INCLUDE@" \
    EXTRA_LIB_PATH=$(echo @BLAS@ | sed -e 's#-[^L]\S\+##g' | sed -e 's#-L\(\S\+\).*#\1#g') \
    LIBS="$(echo @BLAS@ | sed -e 's#-[^l]\S\+##g' | sed -e 's#-l\(\S\+\)#\1#g')" \
    FORCE_FBLAS=true \
    && scons PREFIX="@AUX_PREFIX@" install \
    && cd .. \
    && rm -rf tmv*

