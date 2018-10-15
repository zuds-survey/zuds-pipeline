from ._ffits import *

class F2PYSTOP(Exception):
    def __call__(self, status):
        raise self.__class__(status)
