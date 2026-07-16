# conftest.py – monkey‑patch doctest to use Pygad's numeric checker
import doctest
from pygad.doctest import NumericOutputChecker

doctest.OutputChecker = NumericOutputChecker
