#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:19:11 2019

@author: ubuntu1804
"""

import pygad
import doctest
import warnings
import sys


# monkey patch check_output to doctest.OutputChecker.check_output
from doctest import OutputChecker
from pygad.doctest import check_output_numbers
OutputChecker.check_output = check_output_numbers

warnings.filterwarnings("ignore")
print("*********************************************************************")
print("pygad version ", pygad.__version__)
print("*********************************************************************")
pygad.environment.verbose = pygad.environment.VERBOSE_NORMAL
print("running pygad doctest...")

res = doctest.testmod(pygad.utils.geo)

print("*********************************************************************")
print("return code = ", res.failed)

sys.exit(res.failed)