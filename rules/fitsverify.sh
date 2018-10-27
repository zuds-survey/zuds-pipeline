curl -SL https://heasarc.gsfc.nasa.gov/docs/software/ftools/fitsverify/fitsverify-4.18.tar.gz \
    -o fitsverify-4.18.tar.gz \
    && tar xzf fitsverify-4.18.tar.gz \
    && cd fitsverify \
    && @CC@ @CFLAGS@ -I@AUX_PREFIX@/include -DSTANDALONE -o fitsverify ftverify.c \
    fvrf_data.c fvrf_file.c fvrf_head.c fvrf_key.c fvrf_misc.c \
    -L@AUX_PREFIX@/lib -lcfitsio -lm \
    && cp -a fitsverify "@AUX_PREFIX@/bin/" \
    && cd .. \
    && rm -rf fitsverify*
