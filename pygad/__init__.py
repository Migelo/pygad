'''
The pygad module is a light-weighted analysis module for Gadget files /
snapshots.

All sub-modules are imported along with a "import pygad" except the
plotting sub-module, that is only imported automatically when in
interactive mode (for more details see the documentation about the
plotting module.

doctests:
    >>> import doctest
    >>> import sys
    >>> print("ATTENTION: full doctest takes a few minutes!!!", file=sys.stderr)
    >>> print('testing module utils...', file=sys.stderr)
    >>> doctest.testmod(utils)
    TestResults(failed=0, attempted=5)
    >>> print('testing module environment...', file=sys.stderr)
    >>> doctest.testmod(environment)
    TestResults(failed=0, attempted=0)
    >>> print('testing module units...', file=sys.stderr)
    >>> doctest.testmod(units)
    TestResults(failed=0, attempted=38)
    >>> print('testing module physics...', file=sys.stderr)
    >>> doctest.testmod(physics)
    TestResults(failed=0, attempted=4)
    >>> print('testing module kernels...', file=sys.stderr)
    >>> doctest.testmod(kernels)
    TestResults(failed=0, attempted=3)
    >>> print('testing module gadget...', file=sys.stderr)
    >>> doctest.testmod(gadget)
    TestResults(failed=0, attempted=3)
    >>> print('testing module transformation...', file=sys.stderr)
    >>> doctest.testmod(transformation)
    TestResults(failed=0, attempted=54)
    >>> print('testing module snapshot...', file=sys.stderr)
    >>> doctest.testmod(snapshot)
    TestResults(failed=0, attempted=118)
    >>> print('testing module octree...', file=sys.stderr)
    >>> doctest.testmod(octree)
    TestResults(failed=0, attempted=3)
    >>> print('testing module ssp...', file=sys.stderr)
    >>> doctest.testmod(ssp)
    TestResults(failed=0, attempted=2)
    >>> print('testing module binning...', file=sys.stderr)
    >>> doctest.testmod(binning)
    TestResults(failed=0, attempted=5)
    >>> print('testing module analysis...', file=sys.stderr)
    >>> doctest.testmod(analysis)
    TestResults(failed=0, attempted=6)

    #>>> print('testing module tools...', file=sys.stderr)
    #>>> doctest.testmod(.tools)
    #TestResults(failed=0, attempted=0)

    >>> print('testing module plotting...', file=sys.stderr)
    >>> doctest.testmod(plotting)
    TestResults(failed=0, attempted=2)

For development:
    In order to make the include of pygad work:
    The submodules shall only depent globally on submodules that are listed
    previuously. That means, for instance, physics can depend on units but not
    vice versa. Global dependence on another module is when it is imported in the
    global namespace. Than implies, that including other modules within certain
    functions is allowed.
    Similar is required for the submodules.
'''
# import all modules
from . import utils
from . import environment
#from . import units            made visible directly below
from . import physics
from . import kernels
from . import gadget
#from . import transformation   made visible directly below
#from . import snapshot         made visible directly below
from . import octree
from . import ssp
from . import binning
from . import analysis
#from . import tools            prepare zoom moved to SnapshotCache, others -> Cmd-scripts
from . import cloudy
from . import plotting

# the version of this pygad
# this line gets changed by setup.py during installation, and hence is different
# for the repo and the installed module
#   - in the git repository:
#       simply calls `git_descr(module_dir, PEP440=True)`
#   - in the installed version
#       a string holding the value of above's function call at the moment of
#       installation via setup.py

# make some chosen elements directly visible
from .environment import gc_full_collect
# from .units import Unit, Units, UnitArr, UnitQty, UnitScalar, Fraction
# from .physics import cosmology
# from .physics import G, m_p, solar
# from .transformation import Translation, Rotation
# from .snapshot import Snapshot, SubSnapshot, BallMask, BoxMask, DiscMask, IDMask, ExprMask, SimArr, SnapMask
# from .binning import gridbin2d, gridbin, smooth
# from .tools import prepare_zoom, read_info_file   # prepare_zoom moved to SnapshotCache, read_info_file to SnapshotProperty
# ################################################
# only import core objects to be visible on package level
# functions to be used from sub-modules only
# ################################################
from .units import *
from .transformation import *
from .snapshot import *
from .tools import prepare_zoom, read_info_file

import gc
# default seems to be (700, 10, 10)
# pygad should more often collect garbage, since it has huge objects (SimArr and
# _Snaps), in critical cases, call `gc_full_collect`.
gc.set_threshold(50, 3, 3)
gc_full_collect()

from .environment import module_dir
import subprocess

if not os.path.exists(module_dir+'./CoolingTables/z_0.000.hdf5'):
    url = 'https://bitbucket.org/broett/pygad/downloads/'
    file = 'z_0.000_highres.tar.gz'
    subprocess.run('wget -q %s%s' % (url, file), check=True, shell=True)
    subprocess.run('tar zxvf %s -C %s/' % (file, module_dir), check=True, shell=True)
    subprocess.run('rm -f %s' % file, check=True, shell=True)

if not os.path.exists(module_dir+'./iontbls'):
    url = 'https://bitbucket.org/broett/pygad/downloads/'
    file = 'iontbls.tar.gz'
    subprocess.run('wget -q  %s%s' % (url, file), check=True, shell=True)
    subprocess.run('tar zxf %s -C %s/' % (file, module_dir), check=True, shell=True)
    subprocess.run('rm -f %s' % file, check=True, shell=True)

if not os.path.exists(module_dir+'./bc03'):
    url = 'https://bitbucket.org/broett/pygad/downloads/'
    file = 'bc03.tar.gz'
    subprocess.run('wget -q  %s%s' % (url, file), check=True, shell=True)
    subprocess.run('tar zxf %s -C %s/' % (file, module_dir), check=True, shell=True)
    subprocess.run('rm -f %s' % file, check=True, shell=True)

if not os.path.exists(module_dir+'./snaps'):
    url = 'https://bitbucket.org/broett/pygad/downloads/'
    file = 'snaps.tar.gz'
    subprocess.run('wget -q  %s%s' % (url, file), check=True, shell=True)
    subprocess.run('tar zxf %s -C %s/' % (file, module_dir), check=True, shell=True)
    subprocess.run('rm -f %s' % file, check=True, shell=True)

from ._version import get_versions
__version__ = get_versions()['version']
if environment.verbose > environment.VERBOSE_QUIET:
    print('imported pygad', __version__)
del get_versions
