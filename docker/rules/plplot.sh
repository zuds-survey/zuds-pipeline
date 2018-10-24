curl -SL https://sourceforge.net/projects/plplot/files/plplot/5.12.0%20Source/plplot-5.12.0.tar.gz/download \
    -o plplot-5.12.0.tar.gz \
    && tar xzf plplot-5.12.0.tar.gz \
    && mkdir plplot_build && cd plplot_build \
    && CMAKE_INCLUDE_PATH="@AUX_PREFIX@/include" \
    CMAKE_LIBRARY_PATH="@AUX_PREFIX@/lib:@AUX_PREFIX@/lib64" \
    cmake \
    -D DEFAULT_NO_CAIRO_DEVICES:BOOL=ON \
    -D DEFAULT_NO_QT_DEVICES:BOOL=ON \
    -D CMAKE_C_COMPILER="@CC@" \
    -D CMAKE_CXX_COMPILER="@CXX@" \
    -D CMAKE_C_FLAGS="@CFLAGS@" \
    -D CMAKE_CXX_FLAGS="@CXXFLAGS@" \
    -D CMAKE_INSTALL_PREFIX="@AUX_PREFIX@" \
    ../plplot-5.12.0 \
    && make \
    && make install \
    && cd .. \
    && rm -rf plplot*
