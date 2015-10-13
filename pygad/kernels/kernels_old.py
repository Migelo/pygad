'''
Defining some SPH kernels and helpful functions for them (3D only).

Example:
    Test whether all kernels are normed, non-negative, and zero above 1.

    >>> from scipy.integrate import quad
    >>> for name_kernel, name_integ_kernel in zip(_kernel.items(),
    ...                                           _integ_kernel.items()):
    ...     name = name_kernel[0]
    ...     assert name_integ_kernel[0] == name
    ...     kernel = name_kernel[1]
    ...     integ_kernel = name_integ_kernel[1]
    ...     for u in np.linspace(0,1,100):
    ...         if kernel(u) < 0:
    ...             print 'kernel "%s" is negative at u=%g' % (kernel, u)
    ...         quad_kernel, err = quad(lambda x: 4.*np.pi*x**2 * kernel(x),
    ...                                 0, u, epsabs=1e-10, epsrel=1e-10)
    ...         if abs(quad_kernel - integ_kernel(u)) > 2.*max(err,1e-10):
    ...             print 'integrated kernel "%s" is off by' % name,
    ...             print '%g at' % (quad_kernel - integ_kernel(u)),
    ...             print 'u=%g (err=%g)' % (u, err)
    ...     for u in [1.01, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0]:
    ...         if kernel(u) != 0:
    ...             print 'kernel "%s" is not zero' % name
    ...             print '(%g) at u=%g' % (kernel(u), u)
    ...         if abs(integ_kernel(u) - 1) > 1e-10:
    ...             print 'integrated kernel "%s" is not normed:' % name,
    ...             print 'W(%g)=%g' % (kernel(u), u)
'''
__all__ = ['kernel', 'rand_r_kernel']

import numpy as np

_kernel = {}
_integ_kernel = {}

def cubic_kernel(u):
    '''The cubic kernel.'''
    if u < 0.5:
        return 8./np.pi * (1.+6.*(u-1.)*u**2)
    elif u < 1.0:
        return 8./np.pi * (2.*(1.-u)**3)
    else:
        return 0.0
_kernel['cubic'] = cubic_kernel
def cubic_kernel_integ(u):
    '''The integrated cubic kernel.'''
    if u < 0.5:
        return 32.*u**6 - 38.4*u**5 + 10.6666666666667*u**3
    elif u < 1.0:
        return -10.6666666666667*u**6 + 38.4*u**5 - 48.*u**4 + \
                21.3333333333333*u**3 - 0.0666666666666665
    else:
        return 1.0
_integ_kernel['cubic'] = cubic_kernel_integ

def quintic_kernel(u):
    '''The quintic kernel.'''
    if u < 1./3.:
        return 2187./(40.*np.pi) * ((1.-u)**5-6.*(2./3.-u)**5+15.*(1./3.-u)**5)
    elif u < 2./3.:
        return 2187./(40.*np.pi) * ((1.-u)**5-6.*(2./3.-u)**5)
    elif u < 1.:
        return 2187./(40.*np.pi) * ((1.-u)**5)
    else:
        return 0.0
_kernel['quintic'] = quintic_kernel
def quintic_kernel_integ(u):
    '''The integrated quintic kernel.'''
    if u < 1./3.:
        return -273.375*u**8 + 312.428571428571*u**7 - 97.2*u**5 + 19.8*u**3
    elif u < 2./3.:
        return 136.6875*u**8 - 468.642857142857*u**7 + 607.5*u**6 - \
                340.2*u**5 + 50.625*u**4 + 15.3*u**3 + 0.00297619047619013
    elif u < 1.:
        return -27.3375*u**8 + 156.214285714286*u**7 - 364.5*u**6 + \
                437.4*u**5 - 273.375*u**4 + 72.9*u**3 - 0.301785714285712
    else:
        return 1.0
_integ_kernel['quintic'] = quintic_kernel_integ

def Wendland_C4_kernel(u):
    '''The Wendland C4 kernel.'''
    if u < 1.:
        return 495./(32.*np.pi) * ((1.-u)**6*(1.+6.*u+35./3.*u**2))
    else:
        return 0.0
_kernel['Wendland C4'] = Wendland_C4_kernel
def Wendland_C4_kernel_integ(u):
    '''The integrated Wendland C4 kernel.'''
    if u < 1.:
        return 65.625*u**11 - 396.*u**10 + 962.5*u**9 - 1155.*u**8 + \
                618.75*u**7 - 115.5*u**5 + 20.625*u**3
    else:
        return 1.0
_integ_kernel['Wendland C4'] = Wendland_C4_kernel_integ

def Wendland_C6_kernel(u):
    '''The Wendland C6 kernel.'''
    if u < 1.:
        return 1365./(64.*np.pi) * ((1.-u)**8*(1.+8.*u+25.*u**2+32.*u**3))
    else:
        return 0.0
_kernel['Wendland C6'] = Wendland_C6_kernel
def Wendland_C6_kernel_integ(u):
    '''The integrated Wendland C6 kernel.'''
    if u < 1.:
        return 195.*u**14 - 1515.9375*u**13 + 5005.*u**12 - 8957.8125*u**11 + \
                9009.*u**10 - 4379.375*u**9 + 804.375*u**7 - 187.6875*u**5 + \
                28.4375*u**3
    else:
        return 1.0
_integ_kernel['Wendland C6'] = Wendland_C6_kernel_integ


def kernel(u, name='Wendland C4'):
    '''
    Return the value of the specified kernel at u=r/(2h).

    Note:
        The smoothing length h is defined, such that the kernel is positive for
        r <= 2h and zero above.

    Args:
        u (float):      The normalized radius, i.e. u = r / (2*h).
        name (str):     The name of the kernel.

    Returns:
        kernel (float): The value of the kernel at the given radius.
    '''
    return _kernel[name](u)

def rand_r_kernel(name='Wendland C4'):
    '''
    Get a random radius, with a PDF (propability density function) of the
    specified 3D kernel.

    Args:
        name (str):     The name of the kernel to use.

    Returns:
        r (float):      A random radius.
    '''
    # this import is suspended until here, because it takes some time
    from scipy.optimize import brentq

    q = np.random.random()
    F = _integ_kernel[name]
    return brentq(lambda x: F(x)-q, 0,1)
