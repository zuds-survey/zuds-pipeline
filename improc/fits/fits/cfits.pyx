# cython: c_string_type=str, c_string_encoding=ascii

__all__ = ['read_header']

cdef extern from "gethead.hh":
    void readheader(char* fname, char* key, int datatype, void* value) except +

cdef extern from "fitsio.h":
    int TFLOAT
    int TSTRING
    int TINT

def read_header(fname, key, type):
    """Read the value of a fits header keyword and return it."""

    # First declare a void pointer

    if type == int:
        cdef int val
        cdef int ctype = TINT
    elif type == str:
        cdef char* val
        cdef int ctype = TSTRING
    elif type == float:
        cdef float val
        cdef int ctype = TFLOAT
    else:
        raise ValueError
    cdef void* vp = &val

    # read the value from the header
    readheader(fname, key, ctype, vp)

    # return the coerced result
    return type(val)

