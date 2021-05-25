"""
Basic loading of the fast C library.
"""
from ctypes import (
    POINTER,
    byref,
    c_char_p,
    c_double,
    c_int,
    c_size_t,
    c_uint,
    c_void_p,
    cdll,
    create_string_buffer,
)
from glob import glob

from .. import environment

try:
    cpygad = cdll.LoadLibrary(glob(environment.module_dir + "C/cpygad*.so")[0])
except:
    cpygad = cdll.LoadLibrary(glob(environment.module_dir + "../cpygad*.so")[0])

cpygad.cubic.restype = c_double
cpygad.cubic.argtypes = [c_double, c_double]
cpygad.quartic.restype = c_double
cpygad.quartic.argtypes = [c_double, c_double]
cpygad.quintic.restype = c_double
cpygad.quintic.argtypes = [c_double, c_double]
cpygad.Wendland_C2.restype = c_double
cpygad.Wendland_C2.argtypes = [c_double, c_double]
cpygad.Wendland_C4.restype = c_double
cpygad.Wendland_C4.argtypes = [c_double, c_double]
cpygad.Wendland_C6.restype = c_double
cpygad.Wendland_C6.argtypes = [c_double, c_double]

cpygad.Voigt.restype = c_double
cpygad.Voigt.argtypes = [c_double, c_double, c_double]
