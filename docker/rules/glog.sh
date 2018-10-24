curl -SL https://github.com/google/glog/archive/v0.3.4.tar.gz \
    -o glog-0.3.4.tar.gz \
    && tar xzf glog-0.3.4.tar.gz \
    && cd glog-0.3.4 \
    && CC="@CC@" CXX="@CXX@" CFLAGS="@CFLAGS@" CXXFLAGS="@CXXFLAGS@" ./configure @CROSS@ \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf glog*
