#!/usr/bin/env python

# from Cython.Build import cythonize
import os
import subprocess
import sys
import versioneer

from pygad.environment import git_descr

# check if just uninstalling with setuptools
# and "python setup.py develop --uninstall"
uninstall = sys.argv == ["setup.py", "develop", "--uninstall"]
if sys.argv[1] == "install" or "sdist":
    from distutils.core import setup
if sys.argv[1] == "develop":
    from setuptools import setup

# define all package data
package_data = {
    "pygad.C": ["cpygad.so"],
    "pygad.units": ["units.cfg"],
    "pygad.gadget": ["gadget.cfg"],
    "pygad.snapshot": ["derived.cfg"],
    "pygad.ssp": ["SSP-model/*"],
}
# define scripts
scripts = ["bin/ginsp", "bin/gconv", "bin/gCache3", "bin/gCatalog3", "bin/gStarform3"]
datafiles = [("commands", ["commands/CmdCatalog.py", "commands/CmdStarform.py"])]

# find all sub-packages
modules = []
setup_dir = os.path.dirname(os.path.realpath(__file__))
for root, dirs, files in os.walk(setup_dir):
    submod = os.path.relpath(root, setup_dir).replace(os.sep, ".")
    if not submod.startswith("pygad"):
        continue
    if "__init__.py" in files:
        modules.append(submod)

# compile the C part (has to be done before importing
if not uninstall:
    print("compiling the fast C parts of pygad")
    subprocess.check_call(["make"], cwd=setup_dir + "/pygad/C")

# get version for setup and fix version in module's `pygad.version`
version = git_descr(setup_dir, PEP440=True)
init_file = setup_dir + '/pygad/__init__.py'
init_tmp = '__init__.tmp'
os.rename(init_file, init_tmp)
sub = "s/environment.git_descr(.*)/'%s'/g" % version
with open(init_file, 'w') as f:
    subprocess.check_call(['sed', sub, init_tmp], stdout=f)


try:
    # actually do the setup
    setup(
        name="pygad-mpa",
        description="analysis module for Gadget",
        long_description="A light-weighted analysis module for galaxy "
        + "simulations performed by the SPH code Gadget.",
        author="Bernhard Roettgers",
        author_email="broett@mpa-garching.mpg.de",
        url="https://bitbucket.org/broett/pygad",
        packages=list(map(str, modules)),
        package_data=package_data,
        scripts=scripts,
        data_files=datafiles,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
    )
except:
    raise
# finally:
#    restore the dynamic git version of the development's pygad
#    os.rename(init_tmp, init_file)
