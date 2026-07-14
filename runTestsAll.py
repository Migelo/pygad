#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:19:11 2019

@author: ubuntu1804
"""

import pygad
from pygad.doctest import testmod
import warnings
import sys


warnings.filterwarnings("ignore")
print("*********************************************************************")
print("pygad version ", pygad.__version__)
print("*********************************************************************")
pygad.environment.verbose = pygad.environment.VERBOSE_NORMAL
print("running pygad doctest...")

res = testmod(pygad)

print("*********************************************************************")
print("return code = ", res.failed)

sys.exit(res.failed)
