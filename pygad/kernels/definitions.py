'''
Definitions of the kernels.

Note:
    Outside the kernels' support, the function return undefined values! If
    arbitrary values of radius are use, it has to be taken care of those outside
    the support manually.
                                                                              r
    Furthermore the arguments are always normed by the smoothing length: u = --- .
                                                                             2 h

Examples and doctests:
    >>> cubic(0.0)
    2.5464790894703255
    >>> cubic(0.5)
    0.6366197723675
    >>> cubic(1.0)
    0.0
    >>> Wendland_C4_vec(np.linspace(0,1,12))
    array([4.92385605e+00, 4.56340351e+00, 3.65811505e+00, 2.55311070e+00,
           1.54488080e+00, 7.95912455e-01, 3.36297380e-01, 1.08637670e-01,
           2.33707743e-02, 2.44034893e-03, 4.47381991e-05, 0.00000000e+00])

    Test for the equality of all the vector versions and for normation:
    >>> for name in sorted(kernels):
    ...     kernel = kernels[name]
    ...     for vec_kernels in [vector_kernels]:
    ...         if name not in vec_kernels:
    ...             continue
    ...         c_kernel = getattr(C.cpygad,name.replace(' ','_'),None)
    ...         if c_kernel:
    ...             print('found C version of "%s" kernel' % name)
    ...         kernel_vec = vec_kernels[name]
    ...         u = np.linspace(0,1,100)
    ...         w = kernel_vec(u)
    ...         for uu,ww in zip(u,w):
    ...             if not np.all(np.abs(ww - kernel(uu)) < 1e-6):
    ...                 print(name, 'not equal to its vector version')
    ...                 break
    ...             if c_kernel:
    ...                 if not np.all(np.abs(ww - c_kernel(uu,1.0)) < 1e-3):
    ...                     print(ww, c_kernel(uu,1.0))
    ...                     print(name, 'not equal to its C version')
    ...                     break
    ...         I = np.sum(4*np.pi*u**2 * w) / (len(w)-1)
    ...         if not abs(I-1.0) < 1e-6:
    ...             print(name, 'is not normed to 1, but to', I)
    found C version of "Wendland C2" kernel
    found C version of "Wendland C4" kernel
    found C version of "Wendland C6" kernel
    found C version of "cubic" kernel
    found C version of "quartic" kernel
    found C version of "quintic" kernel
'''
__all__ = [
    'kernels', 'vector_kernels',
    'cubic', 'cubic_vec',
    'quartic', 'quartic_vec',
    'quintic', 'quintic_vec',
    'Wendland_C2', 'Wendland_C2_vec',
    'Wendland_C4', 'Wendland_C4_vec',
    'Wendland_C6', 'Wendland_C6_vec',
]

import numpy as np
from .. import C


def cubic(u):
    '''
    The cubic kernel:

        8   /   2           \               1
        -- | 6 u (u - 1) + 1 |      if  u < -
        pi  \               /               2

          16  /     \ 3
        - -- | 1 - u |              otherwise
          pi  \     /

    where u = r/(2h) <= 1.
    '''
    if u < 0.5:
        return 2.5464790894703255 * (1. + 6. * (u - 1.) * u ** 2)
    else:
        return 5.09295817894 * (1. - u) ** 3


def cubic_vec(u):
    '''The vector version of the cubic kernel.'''
    u = np.asarray(u)
    w = np.empty(len(u))
    mask = u < 0.5
    u_masked = u[mask]
    w[mask] = 2.5464790894703255 * (1. + 6. * (u_masked - 1.) * u_masked ** 2)
    mask = ~mask
    w[mask] = 5.09295817894 * (1. - u[mask]) ** 3
    return w


def quartic(u):
    '''
    The quartic kernel:

         15625  / /    \ 4     / 2    \ 4     / 1    \ 4 \               1
        ------ | |1 - u | - 5 |  - - u | + 10 | - - u |   |      if  u < -
        512 pi  \ \    /       \ 3    /       \ 3    /   /               5

         15625  / /    \ 4     / 2    \ 4 \                          1        3
        ------ | |1 - u | - 5 |  - - u | + |                     if  - <= u < -
        512 pi  \ \    /       \ 3    /   /                          5        5

         15625    /    \ 4
        ------   |1 - u |                                        otherwise
        512 pi    \    /

    where u = r/(2h) <= 1.
    '''
    if u < 0.2:
        return 9.71404681957369 * ((1. - u) ** 4 - 5. * (0.6 - u) ** 4 + 10. * (0.2 - u) ** 4)
    elif u < 0.6:
        return 9.71404681957369 * ((1. - u) ** 4 - 5. * (0.6 - u) ** 4)
    else:
        return 9.71404681957369 * ((1. - u) ** 4)


def quartic_vec(u):
    '''The vector version of the quartic kernel.'''
    u = np.asarray(u)
    w = (1. - u) ** 4
    mask = u < 0.6
    u_masked = u[mask]
    w[mask] -= 5. * (0.6 - u_masked) ** 4
    mask = u < 0.2
    u_masked = u[mask]
    w[mask] += 10. * (0.2 - u_masked) ** 4
    return 9.71404681957369 * w


def quintic(u):
    '''
    The quintic kernel:

         2187  / /    \ 5     / 2    \ 5     / 1    \ 5 \               1
        ----- | |1 - u | - 6 |  - - u | + 15 | - - u |   |      if  u < -
        40 pi  \ \    /       \ 3    /       \ 3    /   /               3

         2187  / /    \ 5     / 2    \ 5 \                          1        2
        ----- | |1 - u | - 6 |  - - u | + |                     if  - <= u < -
        40 pi  \ \    /       \ 3    /   /                          3        3

         2187    /    \ 5
        -----   |1 - u |                                        otherwise
        40 pi    \    /

    where u = r/(2h) <= 1.
    '''
    if u < 0.3333333333333333:
        return 17.403593027098754 * (
                    (1. - u) ** 5 - 6. * (0.6666666666666667 - u) ** 5 + 15. * (0.3333333333333333 - u) ** 5)
    elif u < 0.6666666666666667:
        return 17.403593027098754 * ((1. - u) ** 5 - 6. * (0.6666666666666667 - u) ** 5)
    else:
        return 17.403593027098754 * ((1. - u) ** 5)


def quintic_vec(u):
    '''The vector version of the quintic kernel.'''
    u = np.asarray(u)
    w = (1. - u) ** 5
    mask = u < 0.6666666666666667
    u_masked = u[mask]
    w[mask] -= 6. * (0.6666666666666667 - u_masked) ** 5
    mask = u < 0.3333333333333333
    u_masked = u[mask]
    w[mask] += 15. * (0.3333333333333333 - u_masked) ** 5
    return 17.403593027098754 * w


def Wendland_C2(u):
    '''
    The Wendland C2 kernel:

                  4 /       \
        21 (1 - u) |         |
        ---------- | 1 + 4 u |
           2 pi     \       /

    where u = r/(2h) <= 1.
    '''
    return 3.3422538049298023 * (1. - u) ** 4 * (1. + 4.0 * u)


def Wendland_C2_vec(u):
    '''The vector version of the Wendland C2 kernel.'''
    u = np.asarray(u)
    return 3.3422538049298023 * (1. - u) ** 4 * (1. + 4.0 * u)


def Wendland_C4(u):
    '''
    The Wendland C4 kernel:

                   6  /    2          \
        495 (1 - u)  | 35 u            |
        ------------ | ----- + 6 u + 1 |
           32 pi      \  3            /

    where u = r/(2h) <= 1.
    '''
    return 4.923856051905513 * (1. - u) ** 6 * (1. + (6. + 11.6666666667 * u) * u)


def Wendland_C4_vec(u):
    '''The vector version of the Wendland C4 kernel.'''
    u = np.asarray(u)
    return 4.923856051905513 * ((1. - u) ** 6 * (1. + (6. + 11.6666666667 * u) * u))


def Wendland_C6(u):
    '''
    The Wendland C6 kernel:

                    8
        1365 (1 - u)   /    3      2         \
        ------------- | 32 u + 25 u + 8 u + 1 |
            64 pi      \                     /

    where u = r/(2h) <= 1.
    '''
    return 6.78895304126366 * (1. - u) ** 8 * (1. + (8. + (25. + 32. * u) * u) * u)


def Wendland_C6_vec(u):
    '''The vector version of the Wendland C6 kernel.'''
    u = np.asarray(u)
    return 6.78895304126366 * ((1. - u) ** 8 * (1. + (8. + (25. + 32. * u) * u) * u))


kernels = {
    'cubic': cubic,
    'quartic': quartic,
    'quintic': quintic,
    'Wendland C2': Wendland_C2,
    'Wendland C4': Wendland_C4,
    'Wendland C6': Wendland_C6,
}
vector_kernels = {
    'cubic': cubic_vec,
    'quartic': quartic_vec,
    'quintic': quintic_vec,
    'Wendland C2': Wendland_C2_vec,
    'Wendland C4': Wendland_C4_vec,
    'Wendland C6': Wendland_C6_vec,
}


