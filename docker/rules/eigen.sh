curl -SL http://bitbucket.org/eigen/eigen/get/3.3.3.tar.bz2 \
    -o eigen-eigen-67e894c6cd8f.tar.bz2 \
    && tar xjf eigen-eigen-67e894c6cd8f.tar.bz2 \
    && mkdir eigen_build && cd eigen_build \
    && cmake \
    -D CMAKE_C_COMPILER="@CC@" \
    -D CMAKE_CXX_COMPILER="@CXX@" \
    -D CMAKE_C_FLAGS="@CFLAGS@" \
    -D CMAKE_CXX_FLAGS="@CXXFLAGS@" \
    -D CMAKE_INSTALL_PREFIX="@AUX_PREFIX@" \
    ../eigen-eigen-67e894c6cd8f \
    && make \
    && make install \
    && cd .. \
    && rm -rf eigen*
