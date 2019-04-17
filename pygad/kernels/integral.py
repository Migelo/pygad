'''
Functions for kernels that can be derived from the definitions.

Note:
    Outside the kernels' support, the function return undefined values! If
    arbitrary values of radius are use, it has to be taken care of those outside
    the support manually.
                                                                              r
    Furthermore the arguments are always normed by the smoothing length: u = --- .
                                                                             2 h

Examples and doctests:
    Test some properties of the projected kernels:
    >>> f = project_kernel('Wendland C4')
    >>> integ = 0.0; dx=0.1
    >>> for x in np.arange(-1,1,dx):
    ...     for y in np.arange(-1,1,dx):
    ...         r = np.sqrt(x**2+y**2)
    ...         integ += f(r)
    >>> integ *= dx**2
    >>> if not abs(integ-1.0) < 1e-3:
    ...     print(integ)

    Test some properties of the integrated kernels:
    >>> for ikernel in kernels.items():
    ...     kernel = ikernel[0]
    ...     integ_kernel = integrate_kernel(kernel)
    ...     if abs(integ_kernel(1.0) - 1.0) > 1e-6:
    ...         print('Kernel %s did not got integrated to one!', name)
    ...     if abs(integ_kernel(0.0)) > 1e-6:
    ...         print('Integrated kernel %s does not start at zero!', name)
    ...     for r in [0.1, 0.3, 0.5, 0.8, 0.9]:
    ...         assert 0 < integ_kernel(r) < 1

    Test some properties of the random radii:
    >>> N = 1000
    >>> rs = np.empty(N)
    >>> for ikernel in kernels.items():
    ...     kernel = ikernel[0]
    ...     for n in range(N):
    ...         rs[n] = rand_kernel_r(kernel)
    ...     pdf, edges = np.histogram(rs, bins=np.linspace(0,1,30+1))
    ...     integ_rand = np.cumsum(pdf) / float(N)
    ...     mids = (edges[:-1]+edges[1:]) / 2.
    ...     integ_kernel = integrate_kernel(kernel)
    ...     for i,r in enumerate(mids):
    ...         ref_val = integ_kernel(r)
    ...         err = abs(ref_val - integ_rand[i])
    ...         if err>0.1 and err/ref_val>0.1:
    ...             print('Kernel "%s" was wrongly sampled at r=%f: err=%f' % (kernel, r, err))
'''
__all__ = ['project_kernel', 'integrate_kernel', 'rand_kernel_r']

import numpy as np
from .definitions import *
from ..utils import static_vars

def project_kernel(kernel, N=100, inter_kind='quadratic'):
    '''
    Create an interpolation function for the projected kernel.

    Args:
        kernel (str):       The name of the kernel to project (as in
                            kernels.kernel).
        N (int):            The number of points to use for the interpolation.
        inter_kind (str):   Specifies the kind of interpolation. For more info see
                            `scipy.interpolate.interp1d`.

    Returns:
        proj_kernel (function):
                        A function of R (cylindrical radius / 'impact parameter')
                        of the projected (integrated along one coordinate) kernel.
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
                    kind=inter_kind)

def integrate_kernel(kernel, N=100, inter_kind='quadratic'):
    '''
    Create an interpolation function for the 3D integrated kernel.

    The kernel w gets integrated as:

      /\
      |            2
      |  dx  4 pi x  w(x)
      |
    \/ [0,r]

    and this function (of r) gets returned.

    Args:
        kernel (str):       The name of the kernel to integrate (as in
                            kernels.kernel).
        N (int):            The number of points to use for the interpolation.
        inter_kind (str):   Specifies the kind of interpolation. For more info see
                            `scipy.interpolate.interp1d`.

    Returns:
        integ_kernel (function):    A function of r (radius) of the integrated
                                    kernel.
    '''
    from scipy.interpolate import interp1d
    from scipy.integrate import quad

    kernel = kernels[kernel]
    rs = np.linspace(0,1)
    table = []
    for r in rs:
        I, err = quad(lambda x: 4.*np.pi*x**2 * kernel(x), 0.0, r,
                      epsabs=1e-10, epsrel=1e-10)
        if err > 1e-6:
            raise RuntimeError('Encountered large error (%g) in integration!' %
                               err)
        table.append( I )
    return interp1d(rs, table, copy=False, bounds_error=False, fill_value=1.0,
                    kind=inter_kind)

#old version: needs to use 'intergrate_kernel' now (cache it?!)...
@static_vars( integs={} )
def rand_kernel_r(kernel=None):
    '''
    Get a random radius, with a PDF (propability density function) of the
    specified 3D kernel.

    Not the radii themselves have the PDF of the kernel, but if one would multiply
    them with vectors of length one and random direction in 3D, the resulting
    points in 3D space would have the PDF of the (3D-)kernel.

    Args:
        kernel (str):   The name of the kernel to use. (By default use the kernel
                        defined in `gadget.cfg`.)

    Returns:
        r (float):      A random radius.
    '''
    from scipy.optimize import brentq

    if kernel is None:
        from ..gadget import general
        kernel = general['kernel']

    # this caching gives a speed-up of a factor of an order of magnitude!
    if kernel not in rand_kernel_r.integs:
        rand_kernel_r.integs[kernel] = integrate_kernel(kernel)
    F = rand_kernel_r.integs[kernel]
    q = np.random.random()
    return brentq(lambda x: F(x)-q, 0,1)
