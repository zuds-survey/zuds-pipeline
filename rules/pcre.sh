curl -SL https://sourceforge.net/projects/pcre/files/pcre/8.40/pcre-8.40.tar.bz2/download \
    -o pcre-8.40.tar.bz2 \
    && tar xjf pcre-8.40.tar.bz2 \
    && cd pcre-8.40 \
    && CC="@CC@" CXX="@CXX@" CFLAGS="@CFLAGS@" CXXFLAGS="@CXXFLAGS@" ./configure @CROSS@ \
    --prefix="@AUX_PREFIX@" \
    && make -j 4 && make install \
    && cd .. \
    && rm -rf pcre*
