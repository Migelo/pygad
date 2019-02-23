'''
Terminal helper functions.
'''
__all__ = ['strip_ansi', 'term_len', 'get_terminal_size']

import re
import os

_ansi_re = re.compile('\033\[((?:\d|;)*)([a-zA-Z])')


def strip_ansi(value):
    return _ansi_re.sub('', value)


def term_len(txt):
    '''Determine the length of the text, if printed to the terminal.'''
    return len(strip_ansi(txt))


def get_terminal_size():
    '''
    Determine the size of the terminal.

    Uses `os.popen('stty size', 'r')`. Should work on any Unix system.

    Returns:
        columns (int):  The number of columns.
        rows (int):     The number of rows.
    '''
    rows, columns = os.popen('stty size', 'r').read().split()
    return int(columns), int(rows)

