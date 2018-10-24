curl -SL http://ceres-solver.org/ceres-solver-1.12.0.tar.gz \
    -o ceres-solver-1.12.0.tar.gz \
    && tar xzf ceres-solver-1.12.0.tar.gz \
    && mkdir ceres_build && cd ceres_build \
    && cmake \
    -D CMAKE_C_COMPILER="@CC@" \
    -D CMAKE_CXX_COMPILER="@CXX@" \
    -D CMAKE_C_FLAGS="@CFLAGS@" \
    -D CMAKE_CXX_FLAGS="@CXXFLAGS@" \
    -D EIGEN_INCLUDE_DIR="@AUX_PREFIX@/include/eigen3" \
    -D SUITESPARSE_INCLUDE_DIR_HINTS="@AUX_PREFIX@/include" \
    -D SUITESPARSE_LIBRARY_DIR_HINTS="@AUX_PREFIX@/lib" \
    -D GLOG_INCLUDE_DIR="@AUX_PREFIX@/include" \
    -D GLOG_LIBRARY="@AUX_PREFIX@/lib/libglog.so" \
    -D BUILD_SHARED_LIBS=ON \
    -D BUILD_TESTING=FALSE \
    -D BUILD_EXAMPLES=FALSE \
    -D CMAKE_INSTALL_PREFIX="@AUX_PREFIX@" \
    ../ceres-solver-1.12.0 \
    && make \
    && make install \
    && cd .. \
    && rm -rf ceres*
