curl -SL http://prdownloads.sourceforge.net/scons/scons-2.5.1.tar.gz \
    -o scons-2.5.1.tar.gz \
    && tar xzf scons-2.5.1.tar.gz \
    && cd scons-2.5.1 \
    && python2 setup.py install --prefix="@AUX_PREFIX@" \
    && cd .. \
    && rm -rf scons* \
    && perl -i -p -e 's/usr\/bin\/env python/usr\/bin\/env python2/g' @AUX_PREFIX@/bin/scons*
