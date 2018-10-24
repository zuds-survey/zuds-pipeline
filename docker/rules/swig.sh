curl -SL http://prdownloads.sourceforge.net/swig/swig-3.0.12.tar.gz \
    -o swig-3.0.12.tar.gz \
    && tar xzf swig-3.0.12.tar.gz \
    && cd swig-3.0.12 \
    && CC="@CC@" CXX="@CXX@" CFLAGS="@CFLAGS@" CXXFLAGS="@CXXFLAGS@" ./configure @CROSS@ \
    --prefix=@AUX_PREFIX@ \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf swig*
