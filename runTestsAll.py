#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:19:11 2019

@author: ubuntu1804
"""

import pygad
import doctest
import sys
import warnings

warnings.filterwarnings("ignore")
print("*********************************************************************")
print("pygad version ", pygad.version)
print("*********************************************************************")
pygad.environment.verbose = pygad.environment.VERBOSE_NORMAL
print("running pygad doctest...")

res = doctest.testmod(pygad)

print("*********************************************************************")
print("return code = ", res.failed)

sys.exit(res.failed)