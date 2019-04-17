'''
A collection of some general (low-level) functions.

Doctests are in the functions themselves.
'''
__all__ = ['DevNull', 'static_vars', 'nice_big_num_str', 'float_to_nice_latex',
           'perm_inv', 'periodic_distance_to', 'sane_slice', 'is_consecutive',
           'rand_dir', 'ProgressBar', 'time_from_color_str', 'time_to_nice_str',
           'sec_to_nice_str', 'weighted_percentile']

import numpy as np
import re
import scipy.spatial.distance
import sys
from .term import *
import time
import numbers


class DevNull():
    '''A always open write-only null file object.'''

    def __init__(self, *args, **kwargs):
        self.closed = False
        self.mode = "w"
        self.name = "<null>"
        self.encoding = None
        self.errors = None
        self.newlines = None
        self.softspace = 0

    def close(self):
        pass

    def flush(self):
        pass

    def __next__(self):
        raise IOError("Invalid operation")

    def read(size=0):
        raise IOError("Invalid operation")

    def readline(self):
        raise IOError("Invalid operation")

    def readlines(self):
        raise IOError("Invalid operation")

    def xreadlines(self):
        raise IOError("Invalid operation")

    def seek(self):
        raise IOError("Invalid operation")

    def tell(self):
        return 0

    def truncate(self):
        pass

    def write(self, *args, **kwargs):
        pass

    def writelines(self, *args, **kwargs):
        pass


def static_vars(**kwargs):
    '''
    Decorate a function with static variables.

    Example:
        >>> @static_vars(counter=0)
        ... def foo():
        ...     foo.counter += 1
        ...     print('foo got called the %d. time' % foo.counter)
        >>> foo()
        foo got called the 1. time
        >>> foo()
        foo got called the 2. time
        >>> foo()
        foo got called the 3. time
    '''

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def nice_big_num_str(n, separator=','):
    '''
    Convert a number to a string with inserting separator all 3 digits.

    Note:
        With the default separator ',' one can also use the standard
        "{:,}".format(n).

    Example:
        >>> nice_big_num_str(12345678)
        '12,345,678'
        >>> nice_big_num_str(-12345)
        '-12,345'
        >>> nice_big_num_str(123456, separator=' ')
        '123 456'
        >>> nice_big_num_str(0)
        '0'
    '''
    if n < 0:
        return '-' + nice_big_num_str(-n)
    s = ''
    while n >= 1000:
        n, r = divmod(n, 1000)
        s = "%s%03d%s" % (separator, r, s)
    return "%d%s" % (n, s)


def float_to_nice_latex(x, dec=None):
    '''
    Convert a number to a nice latex representation.

    Args:
        x (float):  The number to convert.
        dec (int):  The number of digits for precision.

    Returns:
        repr (string):  The LaTeX representation.

    Example:
        >>> float_to_nice_latex(1.2345e67, 2)
        '1.23 \\\\times 10^{67}'
        >>> float_to_nice_latex(1e10)
        '10^{10}'
    '''
    s = ('%g' if dec is None else '%%.%dg' % (dec + 1)) % x
    if 'e' in s:
        # two backslashes in raw-string, because it is a regex
        # replace 'e+'
        s = re.sub(r'e\+', r' \\times 10^{', s)
        # replace the 'e' in 'e-'
        s = re.sub(r'e(?=-)', r' \\times 10^{', s)
        s += '}'
        # remove potential '1 \times '
        if s.startswith(r'1 '):
            s = s[9:]
    return s


def perm_inv(perm):
    '''
    Invert a permutation.

    Args:
        perm (array-like):  The permutation in form of an array of the
                            integers 0, 1, 2, ..., N

    Returns:
        inverse (np.ndarray):   The inverse.

    Examples:
        >>> a = np.arange(5)
        >>> ind = a.copy()
        >>> np.random.shuffle(ind)
        >>> np.all( a == a[ind][perm_inv(ind)])
        True
    '''
    return perm.argsort()


def _ndarray_periodic_distance_to(pos, center, boxsize):
    '''periodic_distance_to assuming np.ndarray's as arguments. (for speed)'''
    min_dists = np.minimum((pos - center) % boxsize,
                           (center - pos) % boxsize)
    return np.sqrt((min_dists ** 2).sum(axis=1))


def periodic_distance_to(pos, center, boxsize):
    '''
    Calculate distances in a periodic box.

    Args:
        pos (array-like):       An array of points.
        center (array-like):    The reference point.
        boxsize (float, array-like):
                                The box size. Either a float, then it the box is
                                a cube, or an array-like object, defining the
                                sidelengths for each axis individually.

    Returns:
        dist (array-like):  The distance(s) between the points.

    Examples:
        >>> from .. import units
        >>> from ..units import UnitArr
        >>> from ..environment import module_dir
        >>> units.undefine_all()
        >>> units.define_from_cfg([module_dir+'config/units.cfg'])
        reading units definitions from "units.cfg"
        >>> pos = UnitArr([[1.1,2.1,3.7], [2.8,-1.4,5.4], [7.0,3.4,-5.6]],
        ...               units='m')
        >>> ref = UnitArr([10.,120.,280.], units='cm')
        >>> periodic_distance_to(pos, ref, '5 m')
        UnitArr([1.61864141, 4.1       , 3.318132  ], units="m")
    '''
    from ..units import UnitArr

    if isinstance(boxsize, str):
        from ..units import Unit
        boxsize = Unit(boxsize)

    # go through all the unit mess...
    unit = None
    if isinstance(pos, UnitArr):
        unit = pos.units
        if isinstance(center, UnitArr):
            if not hasattr(boxsize, 'in_units_of') and pos.units != center.units:
                raise ValueError('boxsize is unitless and pos and center have '
                                 'different units. Ambiguous to interpret.')
            center = center.in_units_of(unit).view(np.ndarray)
        else:
            center = np.array(center)
        if hasattr(boxsize, 'in_units_of'):
            boxsize = boxsize.in_units_of(unit)
            if isinstance(boxsize, UnitArr):
                boxsize = boxsize.view(np.ndarray)
        pos = pos.view(np.ndarray)
    elif isinstance(center, UnitArr):
        unit = center.units
        if hasattr(boxsize, 'in_units_of'):
            boxsize = boxsize.in_units_of(unit)
            if isinstance(boxsize, UnitArr):
                boxsize = boxsize.view(np.ndarray)
        center = center.view(np.ndarray)
        pos = np.array(pos)
    elif hasattr(boxsize, 'in_units_of'):
        from ..units import _UnitClass
        unit = boxsize.units
        if isinstance(boxsize, _UnitClass):
            boxsize = float(boxsize)
        elif isinstance(boxsize, UnitArr):
            boxsize = boxsize.view(np.ndarray)
        center = np.array(center)
        pos = np.array(pos)

    r = _ndarray_periodic_distance_to(pos, center, boxsize).view(UnitArr)
    r.units = unit
    return r


def sane_slice(s, N, forward=True):
    '''
    Convert a slice into an equivalent "sane" one.

    The new slice is equivalent in the sense, that it yields the same values when
    applied. These are, however, reverted with respect to the original slice, if
    it had a negative step and `forward` is True.
    "Sane" slice means: none of the attributes of the slice is None (except for a
    backwards slice that contains the first element), 0 <= start <= stop <= N, and
    if the slice is empty the result is always slice(0,0,1).

    Args:
        s (slice):      The slice to make sane.
        N (int):        The length of the sequence to apply the slice to. (Needed
                        for negative indices and None's in the slice).
        forward (bool): Convert to forward slice.

    Returns:
        simple (slice): The sane start and stop position and the (positive) step.

    Raises:
        ValueError:     If a slice with step==0 was passed.

    Examples:
        >>> sane_slice(slice(None,2,-3), 10)
        slice(3, 10, 3)
        >>> sane_slice(slice(-3,None,None), 10)
        slice(7, 10, 1)

        Test some random parameters for the requirements:
        >>> from random import randint
        >>> for i in range(0): #10000
        ...     N = randint(0,10)
        ...     start, stop = randint(-3*N, 3*N), randint(-3*N, 3*N)
        ...     step = randint(-3*N, 3*N)
        ...     if step==0: step = None     # would raise exception
        ...     a = range(N)
        ...     s = slice(start,stop,step)
        ...     ss = sane_slice(s, N)
        ...     if not ( sorted(a[s])==a[ss] and 0<=ss.start<=ss.stop<= N
        ...                 and ss.step>0 ):
        ...         print('ERROR (forward=True):')
        ...         print(N, s, ss)
        ...         print(a[s], a[ss])
        ...     if len(a[s])==0 and not (ss.start==ss.stop==0 and ss.step==1):
        ...         print('ERROR:')
        ...         print('empty slice:', ss)
        ...     ss = sane_slice(s, N, forward=False)
        ...     if not ( a[s]==a[ss] and 0<=ss.start<=N and (
        ...                 (ss.stop is None and ss.step<0) or 0<=ss.stop<=N)):
        ...         print('ERROR (forward=False):')
        ...         print(N, s, ss)
        ...         print(a[s], a[ss])
        ...     if len(a[s])==0 and not (ss.start==ss.stop==0 and ss.step==1):
        ...         print('ERROR:')
        ...         print('empty slice:', ss)
    '''
    # get rid of None's, overly large indices, and negative indices (except -1 for
    # backward slices that go down to first element)
    start, stop, step = s.indices(N)

    # get number of steps & remaining
    n, r = divmod(stop - start, step)
    if n < 0 or (n == 0 and r == 0):
        return slice(0, 0, 1)
    if r != 0:  # it's a "stop index", not the last index
        n += 1

    if step < 0:
        if forward:
            start, stop, step = start + (n - 1) * step, start - step, -step
            stop = min(stop, N)
        else:
            stop = start + n * step
            if stop < 0:
                stop = None
    else:  # step > 0, step == 0 is not allowed
        stop = min(start + n * step, N)

    return slice(start, stop, step)


def is_consecutive(l):
    '''
    Test whether an iterable that supports indexing has consecutive elements.

    Args:
        l (iterable):   Some iterable, that supports indexing for which
                        list(l) == [l[i] for i in range(len(l))]

    Examples:
        >>> is_consecutive([1,2,3])
        True
        >>> is_consecutive([-2,-1,0,1])
        True
        >>> is_consecutive([1,3,2])
        False
        >>> is_consecutive({1:'some', 2:'dict'})
        Traceback (most recent call last):
        ...
        TypeError: Cannot check whether a dict has consecutive values!
    '''
    if isinstance(l, dict):
        raise TypeError('Cannot check whether a dict has consecutive values!')
    return np.all(np.arange(len(l)) + l[0] == list(l))


def rand_dir(dim=3):
    '''
    Create a vectors with uniform spherical distribution.

    Args:
        dim (int):      The number of dimensions the vector shall have.

    Returns:
        r (np.ndarray): A vector of shape (dim,) with length of one pointing into
                        a random direction.

    Examples:
        >>> N = 1000
        >>> for dim in [2,3,4]:
        ...     for n in range(10):
        ...         assert abs( np.linalg.norm(rand_dir(dim=dim)) - 1 ) < 1e-4
        ...     v = np.empty([N,dim])
        ...     for n in range(N):
        ...         v[n] = rand_dir(dim=dim)
        ...     assert np.linalg.norm(v.sum(axis=0))/N < 3.0/np.sqrt(N)
    '''
    if dim < 1:
        raise ValueError('Can only create vectors with at least one dimension!')
    length = 2.
    while length > 1.0 or length < 1e-4:
        r = np.random.uniform(-1., 1., size=dim)
        length = np.linalg.norm(r)
    return r / length


class ProgressBar(object):
    '''
    This function creates an iterable context manager that can be used to iterate
    over something while showing a progress bar.

    It will either iterate over the `iterable` or `length` items (that are counted
    up). While iteration happens, this function will print a rendered progress bar
    to the given `file` (defaults to stdout) and will attempt to calculate
    remaining time and more.

    The context manager creates the progress bar. When the context manager is
    entered the progress bar is already displayed. With every iteration over the
    progress bar, the iterable passed to the bar is advanced and the bar is
    updated. When the context manager exits, a newline is printed and the progress
    bar is finalized on screen.

    Note:
        No printing must happen or the progress bar will be unintentionally
        destroyed!

    Example usage:

    >>> with ProgressBar(range(100)) as pbar:
    ...     for i in pbar:
    ...         time.sleep(0.005)

    Alternatively, if no iterable is specified, one can manually update the
    progress bar through the `update()` method instead of directly iterating over
    the progress bar. The update method accepts the number of steps to increment
    the bar with:

    >>> chunks = [123, 32, 72, 201, 173, 88, 64]
    >>> with ProgressBar(length=sum(chunks)) as pbar:
    ...     for chunk in chunks:
    ...         pass # process_chunk chunk...
    ...         pbar.update(chunk)
    ...         time.sleep(0.1)

    Args:
        iterable (iterable):    An iterable to iterate over. If not provided the
                                length is required, however, it can also just be a
                                number, then the iterable will be
                                `range(iterable)`.
        length (int):           The number of items to iterate over. By default
                                the progress bar will attempt to ask the iterator
                                about its length with `len(iterable)`. Providing
                                the length, overwrites this.
                                If the iterable is None, length is required!
        label (str):            The label to show left to the progress bar.
        show_eta (bool):        Enables or disables the estimated time display.
        show_percent(bool):     Enables or disables the percentage display.
        show_iteration (bool):  Enables or disables the absolute iteration
                                display.
        item_show_func (function):
                                A function called with the current item which can
                                return a string to show the current item next to
                                the progress bar. Note that the current item can
                                be `None`!
        fill_char (str):        The character to use to show the filled part of
                                the progress bar (needs to be a single character).
        empty_char (str):       The character to use to show the non-filled part
                                of the progress bar (needs to be a single
                                character).
        bar_template (str):     The format string to use as template for the bar.
                                The parameters in it are 'label' for the label,
                                'bar' for the progress bar and 'info' for the info
                                section.
        info_sep (str):         The separator between multiple info items (eta etc.).
        width (int):            The width of the progress bar in characters, None
                                means full terminal width.
        file (file):            The file to write to.
    '''
    BEFORE_BAR = '\r\033[?25l'
    AFTER_BAR = '\033[?25h\n'

    def __init__(self, iterable=None, length=None,
                 show_eta=True, show_percent=True, show_iteration=True,
                 item_show_func=None,
                 fill_char='#', empty_char='.',
                 bar_template='%(label)s  [%(bar)s]  %(info)s',
                 info_sep='  ', file=sys.stdout, label=None,
                 width=36, eta_average_over=20):
        if iterable is None and isinstance(length, int):
            self._length = length
            self._iterable = range(length)
        else:
            if isinstance(iterable, int):
                iterable = range(iterable)
            self._length = int(length) if length else len(iterable)
            self._iterable = iter(iterable)
        self.show_eta = bool(show_eta)
        self.show_percent = bool(show_percent)
        self.show_iteration = bool(show_iteration)
        self.fill_char = str(fill_char)
        self.empty_char = str(empty_char)
        if len(self.fill_char) != 1 or len(self.empty_char) != 1:
            raise ValueError("The fill and the empty char both must have length 1.")
        self.bar_template = str(bar_template)
        self.info_sep = str(info_sep)
        self._file = file
        self.label = '' if label is None else str(label)
        self._auto_width = width is None
        self._width = 0 if width is None else width
        self._item_show_func = item_show_func
        self._eta_average_over = eta_average_over

        self._start = self._last_eta = time.time()
        self._last_line = None
        self._it_times = []
        self._it_time_its = []
        self._eta_known = False
        self._finished = False
        self._entered = False
        self._it = 0
        self._current_item = None

    def __enter__(self):
        self._entered = True
        self._start = time.time()
        self._it = 0
        self.render()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self._render_finish()

    def __iter__(self):
        if not self._entered:
            raise RuntimeError('You need to use progress bars in a with block.')
        self.render()
        return self

    def __next__(self):
        try:
            rv = next(self._iterable)
            self._current_item = rv
        except StopIteration:
            self._finish()
            self.render()
            raise StopIteration()
        else:
            self.update(1)
            return rv

    def __len__(self):
        return self._length

    @property
    def auto_width(self):
        return self._auto_width

    @auto_width.setter
    def auto_width(self, value):
        self._auto_width = bool(value)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value is None:
            self._auto_width = True
        else:
            self._width = max(0, int(value))

    @property
    def iteration(self):
        '''The iteration being processed.'''
        return self._it

    @property
    def pct(self):
        '''The percent done (as fraction, i.e. finished is 1.0).'''
        return min(float(self._it) / self._length, 1.0)

    @property
    def time_per_iteration(self):
        if len(self._it_times) < 2:
            return 0.0
        return np.ptp(self._it_times) / float(np.ptp(self._it_time_its))

    @property
    def eta(self):
        if not self._finished:
            return self.time_per_iteration * (self._length - self._it)
        return 0.0

    def format_iteration(self):
        '''The iteration formatted to a string.'''
        return '%s/%s' % (self._it, self._length)

    def format_pct(self):
        '''The percent done formatted to a string.'''
        return '%3d%%' % int(self.pct * 100.)

    def format_eta(self):
        '''The ETA formatted to a string (if unknown return '').'''
        if self._eta_known:
            t = int(self.eta + 1)
            seconds = t % 60
            t /= 60
            minutes = t % 60
            t /= 60
            hours = t % 24
            t /= 24
            if t > 0:
                days = t
                return '%dd %d:%02d:%02d' % (days, hours, minutes, seconds)
            else:
                return '%d:%02d:%02d' % (hours, minutes, seconds)
        return ''

    def format_progress_line(self):
        '''Create the progress bar string (for printing use `render`).'''
        info_bits = []

        bar_length = int(self.pct * self._width)
        bar = self.fill_char * bar_length
        bar += self.empty_char * (self._width - bar_length)

        if self.show_iteration:
            info_bits.append(self.format_iteration())
        if self.show_percent:
            info_bits.append(self.format_pct())
        if self.show_eta and self._eta_known and not self._finished:
            info_bits.append(self.format_eta())
        if self._item_show_func is not None:
            item_info = self._item_show_func(self._current_item)
            if item_info is not None:
                info_bits.append(str(item_info))

        return (self.bar_template % {
            'label': self.label,
            'bar': bar,
            'info': self.info_sep.join(info_bits)
        }).rstrip()

    def render(self):
        '''Render the progress bar.'''
        buf = []

        if self.auto_width:
            old_width = self._width
            self._width = 0
            clutter_length = term_len(self.format_progress_line())
            new_width = max(0, get_terminal_size()[0] - clutter_length)
            if new_width < old_width:
                buf.append(self.BEFORE_BAR)
                buf.append(' ' * old_width)
            self._width = new_width

        clear_width = self._width

        buf.append(self.BEFORE_BAR)
        line = self.format_progress_line()
        line_len = term_len(line)
        buf.append(line)

        buf.append(' ' * (clear_width - line_len))
        line = ''.join(buf)

        if line != self._last_line:
            self._last_line = line
            self._file.write(line)
            self._file.flush()

    def _render_finish(self):
        self._file.write(self.AFTER_BAR)
        self._file.flush()

    def _finish(self):
        self._eta_known = False
        self._current_item = None
        self._finished = True

    def update(self, n_steps):
        '''
        Advance the progress bar by the given number of steps and re-render it.

        Args:
            n_steps (int):  The number of steps to advance.
        '''
        self._make_step(n_steps)
        self.render()

    def _make_step(self, n_steps):
        self._it += n_steps
        if self._it >= self._length:
            self._finished = True

        t = time.time()
        if (t - self._last_eta) < 1.0:
            # not sufficiently good estimate
            return

        self._last_eta = t
        self._it_times = self._it_times[-self._eta_average_over:] + [t]
        self._it_time_its = self._it_time_its[-self._eta_average_over:] + \
                            [self._it]
        self._eta_known = True


def time_from_color_str(time, units='s'):
    '''
    TODO!

    Examples:
        >>> time_from_color_str('6:12')
        UnitArr(372.0, units="s")
        >>> time_from_color_str('1:20:00', units='h')
        UnitArr(1.3333333333333333, units="h")
    '''
    from ..units import UnitArr
    nums = time.split(':')
    t = 0
    for val in nums:
        t = 60 * t + float(val)
    return UnitArr(t, 's').in_units_of(units)


def time_to_nice_str(time, colon=False, opt_hour=False, prec=None, days=None):
    '''
    Convert a number of seconds to a nice string.

    Args:
        time (UnitScalar):  The time unit scalar to format into a nice string.
        colon (bool):       Use colons to seperate hours, minutes, and seconds,
                            not letters.
        opt_hour (bool):    Do not print the number of hours (and days) if they
                            are zero.
        prec (int):         The precision of the seconds. If not specified, it
                            will either just be integer precision for integral
                            values of secs, or it defaults to a precision of
                            three otherwise.
        days (bool):        Whether to print the number of days or just break it
                            down to hours. When not specified / None, the number
                            of days is only printed, it the duration (`secs`)
                            exceeds 24h.

    Returns:
        repr (str):         A human readable string representation of the time in
                            (days,) hours, minutes, and seconds.

    Examples:
        >>> time_to_nice_str(62*60+3)
        '1h 2m 3s'
        >>> time_to_nice_str( '73 min' )
        '1h 13m 0s'
        >>> time_to_nice_str( '73 min', colon=True )
        '01:13:00'
        >>> time_to_nice_str(int(123e4), prec=1)
        '14d 5h 40m 0.0s'
        >>> time_to_nice_str( time_from_color_str('3:11'), colon=True, opt_hour=True )
        '3:11'
    '''
    from ..units import UnitScalar
    time = UnitScalar(time, 's')
    if prec is None:
        if abs(time) > 1 and abs(time - int(time)) < 1e-12:
            prec = 0
    return sec_to_nice_str(float(time), colon=colon, opt_hour=opt_hour, prec=prec, days=days)


def sec_to_nice_str(secs, colon=False, opt_hour=False, prec=None, days=None):
    '''
    Convert a number of seconds to a nice string.

    Args:
        secs (int, float):  The number of seconds to format into a nice string.
        colon (bool):       Use colons to seperate hours, minutes, and seconds,
                            not letters.
        opt_hour (bool):    Do not print the number of hours (and days) if they
                            are zero.
        prec (int):         The precision of the seconds. If not specified, it
                            will either just be integer precision for integral
                            values of secs, or it defaults to a precision of
                            three otherwise.
        days (bool):        Whether to print the number of days or just break it
                            down to hours. When not specified / None, the number
                            of days is only printed, it the duration (`secs`)
                            exceeds 24h.

    Returns:
        repr (str):         A human readable string representation of the time in
                            (days,) hours, minutes, and seconds.

    Examples:
        >>> sec_to_nice_str(12345)
        '3h 25m 45s'
        >>> sec_to_nice_str(12345, colon=True)
        '03:25:45'
        >>> sec_to_nice_str(123e4)
        '14d 5h 40m 0.000s'
        >>> sec_to_nice_str(123e4, colon=True)
        '14d 05:40:00.000'
        >>> sec_to_nice_str(int(123e4))
        '14d 5h 40m 0s'
        >>> sec_to_nice_str(int(123e4), days=False)
        '341h 40m 0s'
        >>> sec_to_nice_str(int(123e4), colon=True)
        '14d 05:40:00'
        >>> sec_to_nice_str(int(123e4), colon=True, days=False)
        '341:40:00'
        >>> sec_to_nice_str(123, opt_hour=True)
        '2m 3s'
        >>> sec_to_nice_str(123, colon=True, opt_hour=True)
        '2:03'
        >>> sec_to_nice_str(180, days=True)
        '0d 0h 3m 0s'
        >>> sec_to_nice_str(123.456)
        '0h 2m 3.456s'
        >>> sec_to_nice_str(123.456, prec=1)
        '0h 2m 3.5s'
        >>> sec_to_nice_str(1.23e-7)
        '0h 0m 0.000000123s'
        >>> sec_to_nice_str(1.23e-12)
        '0h 0m 0.000000000s'
    '''
    if secs < 0:
        raise ValueError("Negative durations are not supported!")
    if not (prec is None or (isinstance(prec, int) and prec >= 0)):
        raise ValueError("Precision needs to be a positive integer!")

    m, s = divmod(secs, 60)
    m = int(m)
    h, m = divmod(m, 60)

    if isinstance(secs, numbers.Integral):
        tformat = '%dh %dm %ds'
        prec = 0
    else:
        if prec is None:
            prec = 3 if secs > 1 else min(9, 3 - np.log10(secs))
        tformat = '%%dh %%dm %%.%dfs' % prec
    if colon:
        tformat = tformat[:-1].replace('h ', ':').replace('m ', ':')
        tformat = tformat.replace('%d', '%02d')
        tformat = tformat.replace('%.', '%%0%d.' % (2 + (prec > 0) + prec))

    if days or (days is None and h >= 24):
        d, h = divmod(h, 24)
        string = ('%dd ' + tformat) % (d, h, m, s)
    else:
        d = None
        string = tformat % (h, m, s)

    if opt_hour and (d is None or d == 0) and h == 0:
        if colon:
            string = string[string.find(':') + 1:]
        else:
            string = string[string.find(' ') + 1:]
        string = string.lstrip('0')
        if string[0] == ':':
            string = '0' + string

    return string


def weighted_percentile(qty, perc, weights=None, values_sorted=False,
                        ignore_nan=False, new_style=False):
    '''
    Very close to np.percentile, but supports weights.

    Args:
        qty (array-like):       The data to calculate the percentiles for.
        perc (array-like):      The percentiles to calculate. Must be in the range
                                [0,100], of course.
        weights (array-like):   The weights (or counts) for the values in `qty`.
                                Needs to be the same length as `qty`, with a
                                one-to-one correspondence to the values.
        values_sorted (bool):   If True, it will be assumed that `qty` already is
                                a sorted array.
        ignore_nan (bool):      Whether to ignore values of NaN in both `qty` and
                                `weights`, i.e. to cut them out in advance.
        new_style (bool):       If False, the output will be consistent with
                                np.percentile; if True, the interpolation is
                                slightly different: the 33-percentile of [0,0.5,1]
                                no longer is 0.333..., but 0.25 (simply the mean
                                of the two first elements).

    Returns:
        perc (array-like):      The percentiles.

    Examples:
        >>> weighted_percentile( [0,0.5,1], [100./3], new_style=False)
        array([0.33333333])
        >>> np.percentile( [0,0.5,1], [100./3] )[0] # doctest: +ELLIPSIS
        0.333333...
        >>> weighted_percentile( [0,0.5,1], [100./3], new_style=True)
        array([0.25])
        >>> weighted_percentile( [4,5,1,10,6,2,3], [25,50,75],
        ...                      weights=[12,8,7,10,9,3,4])
        array([3.328125  , 4.575     , 5.80882353])
        >>> weighted_percentile( [3,4,2,7,4,1], [0,54.21,100],
        ...                      weights=[1,2,1,2,1,2])
        array([1.        , 3.86313333, 7.        ])
    '''
    if not isinstance(qty, np.ndarray):
        qty = np.array(qty)
    perc = np.array(perc, dtype=float) / 100.
    if np.any(perc < 0) or np.any(perc > 100):
        raise ValueError('Percentiles need to be be in [0,100]!')
    if weights is None:
        weights = np.ones(len(qty))
    weights = np.array(weights, dtype=float)
    if len(weights) != len(qty):
        raise ValueError('Weights need to be the same length as `qty`!')

    if ignore_nan:
        not_nan = ~(np.isnan(qty) | np.isnan(weights))
        qty = qty[not_nan]
        weights = weights[not_nan]

    if not values_sorted:
        sorter = np.argsort(qty)
        qty = qty[sorter]
        weights = weights[sorter]

    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    if new_style:
        weighted_quantiles /= np.sum(weights)
    else:
        # consistent with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    return np.interp(perc, weighted_quantiles, qty)

