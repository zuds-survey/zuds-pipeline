# cython: c_string_type=str, c_string_encoding=ascii
import cython

__all__ = ['read_header_string', 'read_header_float', 'read_header_int']


cdef extern from "gethead.hh":
    void readheader(char* fname, char* key, int datatype, void* value) except +


cdef extern from "fitsio.h":
    int TFLOAT
    int TSTRING
    int TINT


@cython.embedsignature(True)
def read_header_string(fname, key):
    """Read the value of a fits header keyword and return it as a string."""

    # First declare a void pointer
    cdef:
        char* val;
        void* vp = &val;

    # read the value from the header
    readheader(fname, key, TSTRING, vp)

    # return the coerced result
    return val


@cython.embedsignature(True)
def read_header_int(fname, key):
    """Read the value of a fits header keyword and return it as a string."""

    # First declare a void pointer
    cdef:
        int val;
        void* vp = &val;

    # read the value from the header
    readheader(fname, key, TINT, vp)

    # return the coerced result
    return val


@cython.embedsignature(True)
def read_header_float(fname, key):
    """Read the value of a fits header keyword and return it as a string."""

    # First declare a void pointer
    cdef:
        float val;
        void* vp = &val;

    # read the value from the header
    readheader(fname, key, TFLOAT, vp)

    # return the coerced result
    return val

