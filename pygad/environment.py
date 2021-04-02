'''
Determine whether we are in interactive mode or not and provide function to
securely import h5py.
'''
__all__ = ['git_descr', 'interactive', 'module_dir', 'can_use_h5py',
           'secure_get_h5py', 'gc_full_collect']

import sys
import os
from .utils import *
import gc

module_dir = os.path.dirname(__file__)+'/'

def git_descr(path=os.curdir, dirty='dirty', PEP440=False):
    """
    Get a brief description of the current git revision of a directory.

    Args:
        path (str):     The path that point to or into the git repository you want
                        to get a revision / version description of.
        dirty (str):    If the repository is dirty, append '-<dirty>' to the
                        string. (I.e. call `git describe [...] --dirty=-<dirty>`.)
                        If None, do not mark a dirty repository.
        PEP440 (bool):  Convert to a PEP 0440 compliant version string plus a
                        leading 'v'. It is assumed that `git describe [...]`
                        returns a string that starts with a 'v' followed parts of
                        a PEP 0440 compliant version string. This typically
                        requires that the last git tag exists and is like 'v1.23'.

    Returns:
        descr (str):    A description of the current git revision of the repo.

    Raises:
        IOError:        If the given path does not exists.
        subprocess.CalledProcessError:
                        If the call of `git describe --always --tags [--dirty]`
                        failed.
        RuntimeError:   If PEP440 is True and the string obtained by `git
                        describe` returned could not have been converted into a
                        PEP 0440 compliant version.
    """
    if not os.path.exists(os.path.expanduser(path)):
        raise IOError('The path "%s" does not exist!' % path)
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    import subprocess

    cmd = ['git', 'describe', '--always', '--tags']
    if dirty is not None:
        cmd.append('--dirty=-'+dirty)
    descr = subprocess.check_output(cmd, cwd=path).strip().decode('ascii')

    if PEP440:
        import re
        match = re.match(
                r'^(?P<vtag>(v|V)(\d+!)?\d+(\.\d+)*({a|b|rc}\d+)?(\.\d+)?)' + \
                r'(?P<rev>[a-zA-Z0-9\-]*)$',
                descr)
        if not match:
            raise RuntimeError('Description obtained by `git describe` is ' + \
                               'not PEP440 compliant. Probably git tags are ' + \
                               'missing or the last one is not a version ' +
                               'tag.')
        vtag = match.group('vtag')
        rev = match.group('rev')
        if re.match('-\d+-', rev):
            i = rev.replace('-','x',1).find('-')
            rev = rev[:i] + '+' + rev[i+1:]
        descr = vtag + rev.replace('-', '.')

    return descr

import __main__ as main
if hasattr(main, '__file__'):
    interactive = False
else:
    interactive = True

VERBOSE_QUIET       = 0
VERBOSE_TACITURN    = 1
VERBOSE_NORMAL      = 2
VERBOSE_TALKY       = 3
verbose = VERBOSE_NORMAL if interactive else VERBOSE_QUIET
allow_parallel_conversion = interactive

DEFAULT_Rvir_property = 'R200_com'      # Halo property to get virial radius
DEFAULT_gx_radius = 0.1                 # fraction of Rvir to get galaxy radius
DEFAULT_findgxfast = False              # default method to find zentral galaxy
DEFAULT_linking_length = "6 ckpc"           # linking length None for FoF from pepare_zoom, in samples '6 ckpc', gTrace 0.05
DEFAULT_linking_vel = '200 km/s'        # linking vel for FoF from prepare zoom, gtrace 100 km/s
DEFAULT_lowres_threshold = 1e-2         # theshold for FoF

# Plot parameters used by tool and utils
PLOT_figx = 11
PLOT_figy = 9
PLOT_fontsize = 18

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
            print('WARNING: Could not import h5py -- format 3 ' + \
                                 'Gadget files are, hence, not supported.', file=sys.stderr)
            secure_get_h5py._called = True
        return _h5py_dummy

def gc_full_collect():
    '''Collect all collectable garbage.'''
    # just to be sure:
    old_thresholds = gc.get_threshold()
    gc.set_threshold(1, 1, 1)
    # sometimes a single call is not enough!
    while gc.collect():
        pass
    gc.set_threshold(*old_thresholds)
    if len(gc.garbage) > 0:
        print('WARNING: there is uncollectable garbage after a ' + \
                             'call of `gc_full_collect`!', file=sys.stderr)

