#!/usr/bin/env python
import os
import sys
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

include_dirs = [
    "/usr/include/",
    "pygad/C/include/",
    "/opt/homebrew/opt/gsl/include/",
    "/opt/local/",
]
library_dirs = []
if os.getenv("GSL_HOME") is not None:
    include_dirs.append(os.getenv("GSL_HOME") + "/include")
    library_dirs.append(os.getenv("GSL_HOME") + "/lib")

if sys.platform == "darwin":
    # Apple clang does not support GCC's -fopenmp / -lgomp; use libomp instead
    # (install with `brew install libomp` or set LIBOMP_HOME).
    omp_candidates = ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
    if os.getenv("LIBOMP_HOME"):
        omp_candidates.insert(0, os.getenv("LIBOMP_HOME"))
    omp_prefix = next((p for p in omp_candidates if os.path.isdir(p)), None)
    if omp_prefix is None:
        raise RuntimeError(
            "libomp was not found. Install it with `brew install libomp` "
            "or point LIBOMP_HOME at its prefix."
        )
    include_dirs.append(os.path.join(omp_prefix, "include"))
    library_dirs.append(os.path.join(omp_prefix, "lib"))
    # Homebrew's GSL is keg-only and not on the default linker search path.
    for _gsl_lib_dir in ("/opt/homebrew/opt/gsl/lib", "/usr/local/opt/gsl/lib"):
        if os.path.isdir(_gsl_lib_dir):
            library_dirs.append(_gsl_lib_dir)
    omp_compile_args = ["-Xpreprocessor", "-fopenmp"]
    omp_link_args = []
    omp_libraries = ["m", "gsl", "gslcblas", "omp"]
else:
    omp_compile_args = ["-fopenmp"]
    omp_link_args = ["-fopenmp"]
    omp_libraries = ["m", "gsl", "gslcblas", "gomp"]

ext_module = Extension(
    "pygad/C/cpygad",
    language="c++",
    sources=glob("pygad/C/src/*"),
    include_dirs=include_dirs,
    extra_compile_args=[
        "-fPIC",
        "-std=c++11",
        "-O3",
        *omp_compile_args,
        "-pedantic",
        "-Wall",
        "-Wextra",
    ],
    libraries=omp_libraries,
    extra_link_args=omp_link_args,
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
