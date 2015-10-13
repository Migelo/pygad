#!/usr/bin/env python

try:
    from setuptools import setup
except:
    from distutils.core import setup
#from Cython.Build import cythonize
import numpy
import os
import subprocess


# define all package data
package_data = {'pygad.units':          ['units.cfg'],
                'pygad.gadget':         ['gadget.cfg'],
                'pygad.snapshot':       ['derived.cfg'],
                'pygad.luminosities':   ['SSP-model/*']}

# find all sub-packages
modules = []
setup_dir = os.path.dirname(os.path.realpath(__file__))
for root, dirs, files in os.walk(setup_dir):
    submod = os.path.relpath(root, setup_dir).replace(os.sep,'.')
    if not submod.startswith('pygad'):
        continue
    if '__init__.py' in files:
        modules.append( submod )

# get version for setup and fix version in module's `pygad.version`
from pygad.environment import git_descr
version = git_descr(setup_dir, PEP440=True)
init_file = setup_dir + '/pygad/__init__.py'
init_tmp = '__init__.tmp'
os.rename(init_file, init_tmp)
sub = "s/environment.git_descr(.*)/'%s'/g" % version
with open(init_file, 'w') as f:
    subprocess.check_call(['sed', sub, init_tmp], stdout=f)


try:
    # actually do the setup
    setup(name = 'pygad',
          version = version,
          description = 'analysis module for Gadget',
          long_description = 'A light-weighted analysis module for galaxy ' + \
                             'simulations performed by the SPH code Gadget.',
          author = 'Bernhard Roettgers',
          author_email = 'broett@mpa-garching.mpg.de',
          url = 'https://bitbucket.org/broett/pygad',
          packages = map(str,modules),
          package_data = package_data,
         )
except:
    raise
finally:
    # restore the dynamic git version of the development's pygad
    os.rename(init_tmp, init_file)

