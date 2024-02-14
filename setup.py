#!/usr/bin/env python
import os
import subprocess
from glob import glob

from setuptools import Extension, setup

import versioneer

# define scripts
scripts = ["bin/ginsp", "bin/gconv", "bin/gCache3", "bin/gCatalog3", "bin/gStarform3"]

# find all sub-packages
modules = []
setup_dir = os.path.dirname(os.path.realpath(__file__))
for root, dirs, files in os.walk(setup_dir):
    submod = os.path.relpath(root, setup_dir).replace(os.sep, ".")
    if not submod.startswith("pygad"):
        continue
    if "__init__.py" in files:
        modules.append(submod)

# clean and make the cpygad.so library
subprocess.run(["make", "clean"], cwd=setup_dir + "/pygad/C", check=False)

gsl_include = ""
gsl_lib = ""
library_dirs = [
    "/opt/homebrew/Cellar/gsl/2.7.1/lib",
]
include_dirs = [
    "/usr/include/",
    "pygad/C/include/",
    "/opt/homebrew/opt/gsl/include/",
    "/opt/local/",
]
if os.getenv("GSL_HOME") is not None:
    gsl_include = os.getenv("GSL_HOME") + "/include"
    include_dirs.append(gsl_include)
    gsl_lib = os.getenv("GSL_HOME") + "/lib"
    library_dirs.append(gsl_lib)

ext_module = Extension(
    "pygad/C/cpygad",
    language="c++",
    sources=glob("pygad/C/src/*"),
    include_dirs=include_dirs,
    extra_compile_args=[
        "-fPIC",
        "-std=c++11",
        "-O3",
        "-fopenmp",
        "-pedantic",
        "-Wall",
        "-Wextra",
    ],
    libraries=["m", "gsl", "gslcblas", "gomp"],
    extra_link_args=["-fopenmp"],
    library_dirs=library_dirs,
)

setup(
    name="pygadmpa",
    description="analysis module for Gadget",
    long_description="A light-weighted analysis module for galaxy \
        simulations performed by the SPH code Gadget.",
    author="Bernhard Roettgers",
    author_email="broett@mpa-garching.mpg.de",
    url="https://bitbucket.org/broett/pygad",
    include_package_data=True,
    packages=list(map(str, modules)),
    scripts=scripts,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    ext_modules=[ext_module],
)
