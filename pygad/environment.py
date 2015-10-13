'''
Determine whether we are in interactive mode or not and provide function to
securely import h5py.
'''
__all__ = ['interactive', 'module_dir', 'can_use_h5py', 'secure_get_h5py']

import sys
import os
from utils import *

module_dir = os.path.dirname(__file__)+'/'

def git_descr(path=os.curdir):
    '''
    Get a brief description of the current git revision of a directory.

    Args:
        path (str):     The path that point to or into the git repository you want
                        to get a revision / version description of.

    Returns:
        descr (str):    A description of the current git revision of the repo.

    Raises:
        IOError:        If the given path does not exists.
        subprocess.CalledProcessError:
                        If the call of 'git describe --always --tags --dirty'
                        failed.
    '''
    if not os.path.exists(path):
        raise IOError('The path "%s" does not exist!' % path)
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    import subprocess
    cmd = ['git', 'describe', '--always', '--tags', '--dirty']
    descr = subprocess.check_output(cmd, cwd=path).strip()
    return descr

import __main__ as main
if hasattr(main, '__file__'):
    interactive = False
else:
    interactive = True

verbose = interactive
allow_parallel_conversion = interactive

class _h5py_dummy(object):
    '''
    Class holding some stuff to make pygad work even if there is no h5py
    module that can be loaded, but only this class to be returned by
    secure_get_h5py.
    '''
    class File(object):
        '''No object is of type h5py.File...'''
        def __init__(self, *args, **kwargs):
            pass
    @staticmethod
    def is_hdf5(filename):
        '''No file is detected as HDF5 file...'''
        return False

_can_use_h5py = False
def can_use_h5py():
    '''Test whether h5py was properly imported by secure_get_h5py.'''
    global _can_use_h5py
    return _can_use_h5py

@static_vars(_called=False)
def secure_get_h5py():
    '''
    Try to import h5py and return it. If this fails, return a dummy class, that
    has a class 'File' and a static function 'is_hdf5' in order to make code run,
    that, however, then can never really use h5py...
    '''
    global _can_use_h5py
    try:
        import h5py
        _can_use_h5py = True
        return h5py
    except:
        if not secure_get_h5py._called:
            print >> sys.stderr, 'WARNING: Could not import h5py -- format 3 ' + \
                                 'Gadget files are, hence, not supported.'
            secure_get_h5py._called = True
        return _h5py_dummy

