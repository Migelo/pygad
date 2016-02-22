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
    >>> print >> sys.stderr, "ATTENTION: full doctest takes a few minutes!!!"
    >>> print >> sys.stderr, 'testing module utils...'
    >>> doctest.testmod(utils)
    TestResults(failed=0, attempted=3)
    >>> print >> sys.stderr, 'testing module environment...'
    >>> doctest.testmod(environment)
    TestResults(failed=0, attempted=0)
    >>> print >> sys.stderr, 'testing module units...'
    >>> doctest.testmod(units)
    TestResults(failed=0, attempted=3)
    >>> print >> sys.stderr, 'testing module physics...'
    >>> doctest.testmod(physics)
    TestResults(failed=0, attempted=3)
    >>> print >> sys.stderr, 'testing module kernels...'
    >>> doctest.testmod(kernels)
    TestResults(failed=0, attempted=3)
    >>> print >> sys.stderr, 'testing module gadget...'
    >>> doctest.testmod(gadget)
    TestResults(failed=0, attempted=3)
    >>> print >> sys.stderr, 'testing module transformation...'
    >>> doctest.testmod(transformation)
    TestResults(failed=0, attempted=2)
    >>> print >> sys.stderr, 'testing module snapshot...'
    >>> doctest.testmod(snapshot)
    TestResults(failed=0, attempted=5)
    >>> print >> sys.stderr, 'testing module octree...'
    >>> doctest.testmod(octree)
    TestResults(failed=0, attempted=3)
    >>> print >> sys.stderr, 'testing module ssp...'
    >>> doctest.testmod(ssp)
    TestResults(failed=0, attempted=2)
    >>> print >> sys.stderr, 'testing module binning...'
    >>> doctest.testmod(binning)
    TestResults(failed=0, attempted=4)
    >>> print >> sys.stderr, 'testing module analysis...'
    >>> doctest.testmod(analysis)
    TestResults(failed=0, attempted=5)
    >>> print >> sys.stderr, 'testing module tools...'
    >>> doctest.testmod(tools)
    TestResults(failed=0, attempted=0)
    >>> print >> sys.stderr, 'testing module plotting...'
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
import gc
# default seems to be (700, 10, 10)
gc.set_threshold(20, 5, 5)

# import all modules
import utils
import environment
import units
import physics
import kernels
import gadget
import transformation
import snapshot
import octree
import ssp
import binning
import analysis
import tools
if environment.interactive:
    import plotting

# the version of this pygad
# this line gets changed by setup.py during installation, and hence is different
# for the repo and the installed module
#   - in the git repository:
#       simply calls `git_descr(module_dir, PEP440=True)`
#   - in the installed version
#       a string holding the value of above's function call at the moment of
#       installation via setup.py
version = environment.git_descr( environment.module_dir, PEP440=True )
if environment.verbose:
    print 'imported pygad', version

# make some chosen elements directly visible
from environment import gc_full_collect
from units import Unit, Units, UnitArr, UnitQty, UnitScalar, dist, Fraction
from physics import cosmology
from physics import G, m_p, solar
from transformation import Translation, Rotation
from snapshot import Snap, BallMask, BoxMask, DiscMask, IDMask
from binning import gridbin2d, gridbin, smooth
from plotting import show_image
from tools import prepare_zoom, read_info_file

