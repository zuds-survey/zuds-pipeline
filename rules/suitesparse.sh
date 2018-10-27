curl -SL http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-4.5.4.tar.gz \
    -o SuiteSparse-4.5.4.tar.gz \
    && tar xzf SuiteSparse-4.5.4.tar.gz \
    && cd SuiteSparse \
    && patch -p1 < ../rules/patch_suitesparse \
    && make CC="@CC@" CXX="@CXX@" CFLAGS="@CFLAGS@" AUTOCC=no \
    F77="@FC@" F77FLAGS="@FCFLAGS" \
    CFOPENMP="@OPENMP_CXXFLAGS@" LAPACK=" @LAPACK@" BLAS="@BLAS@" \
    MY_METIS_INC="@AUX_PREFIX@/include" MY_METIS_LIB="-L@AUX_PREFIX@/lib -lmetis" \
    && make install CC="@CC@" CXX="@CXX@" CFLAGS="@CFLAGS@" AUTOCC=no \
    F77="@FC@" F77FLAGS="@FCFLAGS" \
    CFOPENMP="@OPENMP_CXXFLAGS@" LAPACK=" @LAPACK@" BLAS="@BLAS@" \
    MY_METIS_INC="@AUX_PREFIX@/include" MY_METIS_LIB="-L@AUX_PREFIX@/lib -lmetis" \
    INSTALL="@AUX_PREFIX@" \
    && cd .. \
    && rm -rf SuiteSparse*
