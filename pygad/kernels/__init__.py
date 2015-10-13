'''
All about kernels

Also doctest other parts of this sub-module:
    >>> import doctest
    >>> doctest.testmod(definitions)
    TestResults(failed=0, attempted=5)

    >>> f = project_kernel('Wendland C4')
    >>> integ = 0.0; dx=0.1
    >>> for x in np.arange(-1,1,dx):
    ...     for y in np.arange(-1,1,dx):
    ...         r = np.sqrt(x**2+y**2)
    ...         integ += f(r)
    >>> integ *= dx**2
    >>> if not abs(integ-1.0) < 1e-3:
    ...     print integ
'''

from definitions import *

def project_kernel(kernel, N=100, inter_kind='quadratic'):
    '''
    Create an interpolation function for the integrated kernel.

    Args:
        kernel (str):   The name of the kernel to project (as in kernels.kernel).
        N (int):        The number of points to use for the interpolation.

    Returns:
        proj_kernel (function):
                        A function of R (cylindrical radius) of the projected
                        (integrated along one coordinate) kernel.
    '''
    from scipy.interpolate import interp1d
    from scipy.integrate import quad

    kernel = kernels[kernel]
    Rs = np.arange(N+1,dtype=float) / N
    table = []
    for R in Rs:
        zmax = np.sqrt(1.0-R**2)
        w, err = quad(lambda z: kernel(np.sqrt(R**2+z**2)), -zmax, zmax)
        if err > 1e-6:
            raise RuntimeError('Encountered large error (%g) in integration!' %
                               err)
        table.append( w )
    return interp1d(Rs, table, copy=False, bounds_error=False, fill_value=0.0,
                    assume_sorted=True)

