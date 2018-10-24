curl -SL https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/bzip2/1.0.6-8/bzip2_1.0.6.orig.tar.bz2 \
    -o bzip2-1.0.6.tar.bz2 \
    && tar xjf bzip2-1.0.6.tar.bz2 \
    && cd bzip2-1.0.6 \
    && make CC="@CC@" CFLAGS="@CFLAGS@" PREFIX="@AUX_PREFIX@" \
    && make CC="@CC@" CFLAGS="@CFLAGS@" PREFIX="@AUX_PREFIX@" install \
    && cd .. \
    && rm -rf bzip2*
