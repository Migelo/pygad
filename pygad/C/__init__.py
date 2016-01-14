'''
Basic loading of the fast C library.
'''
from .. import environment
from ctypes import cdll, c_void_p, c_size_t, c_double
cpygad = cdll.LoadLibrary(environment.module_dir+'C/cpygad.so')

