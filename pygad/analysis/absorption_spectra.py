"""
Produce mock absorption spectra for given line transition(s) and line-of-sight(s).

Doctests:
    >>> from ..environment import module_dir
    >>> from ..snapshot import Snapshot
    >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_320', physical=False)

    >>> vs = UnitArr([1,1e3,1e5], 'km/s')
    >>> print(velocities_to_redshifts(vs, 0.1))
    [0.10000367 0.10366921 0.4669205 ]
    >>> print(redshifts_to_velocities(velocities_to_redshifts(vs,0.1), 0.1))
    [1.e+03 1.e+06 1.e+08] [m s**-1]
    >>> for z0 in (0.0, 0.123, 1.0, 2.34, 10.0):
    ...     zs = velocities_to_redshifts(vs, z0=z0)
    ...     vs_back = redshifts_to_velocities(zs, z0=z0)
    ...     if np.max(np.abs( ((vs-vs_back)/vs).in_units_of(1) )) > 1e-10:
    ...         print(vs)
    ...         print(vs_back)
    ...         print(np.max(np.abs( ((vs-vs_back)/vs).in_units_of(1) )))

    >>> los_arr = UnitArr([[ 34800.,  35566.],
    ...                    [ 34700.,  35600.],
    ...                    [ 35000.,  35600.]], 'ckpc/h_0')
    >>> environment.verbose = environment.VERBOSE_QUIET

    >>> thermal_b_param('H1215', '1e4 K')   # doctest: +ELLIPSIS
    UnitArr(12.8444..., units="km s**-1")
    >>> l, tau = line_profile(line='H1215', N='1e16 cm**-2', T='1e4 K')
    >>> tau[543]    # doctest: +ELLIPSIS
    0.00172...
    >>> EW(tau, l[1]-l[0])  # doctest: +ELLIPSIS
    UnitArr(0.2787..., units="Angstrom")

    Broadly following Oppenheimer & Dave (2009) for the OVI turbulent broadening
    (adding the minimum of 100 km/s):
    >>> nH = s.gas['nH'].in_units_of('cm**-3')
    >>> b_turb = UnitArr( np.sqrt( np.maximum(1405.*np.log10(nH**2) +
    ...                     15674.*np.log10(nH) + 43610., 100.**2 ) ), 'km/s')
    >>> for los in los_arr:
    ...     print('l.o.s.:', los)
    ...     for line in ['H1215', 'OVI1031']:
    ...         print('  ', line)
    ...         for method in ['particles', 'line', 'column']:
    ...             tau, dens, temp, v_edges, restr_column = mock_absorption_spectrum_of(
    ...                 s, los, line=line,
    ...                 vel_extent=UnitArr([2000.,3500.], 'km/s'),
    ...                 method=method,
    ...                 v_turb=b_turb if line=='OVI1031' else '0 km/s',
    ...             )
    ...             N = dens.sum()
    ...             print('    N = %.3e %s' % (N, N.units), end='')
    ...             if method == 'particles':
    ...                 N_restr = np.sum(restr_column)
    ...                 N_restr.convert_to('cm**-2', subs=s)
    ...                 if np.abs((N_restr - N) / N) > 0.01:
    ...                     print('; N = %.3e %s' % (N_restr, N_restr.units)),
    ...             z_edges = velocities_to_redshifts(v_edges, z0=s.redshift)
    ...             l = UnitArr(lines[line]['l'])
    ...             l_edges = l * (1.0 + z_edges)
    ...             EW_l = EW(tau, l_edges)
    ...             print('; EW = %.3f %s' % (EW_l, EW_l.units))
    l.o.s.: [34800. 35566.] [ckpc h_0**-1]
       H1215
        N = 7.506e+17 [cm**-2]; EW = 1.879 [Angstrom]
        N = 7.501e+17 [cm**-2]; EW = 1.439 [Angstrom]
        N = 7.509e+17 [cm**-2]; EW = 1.412 [Angstrom]
       OVI1031
        N = 8.693e+14 [cm**-2]; EW = 0.990 [Angstrom]
        N = 8.686e+14 [cm**-2]; EW = 0.655 [Angstrom]
        N = 8.700e+14 [cm**-2]; EW = 0.648 [Angstrom]
    l.o.s.: [34700. 35600.] [ckpc h_0**-1]
       H1215
        N = 2.815e+15 [cm**-2]; EW = 1.435 [Angstrom]
        N = 2.811e+15 [cm**-2]; EW = 1.156 [Angstrom]
        N = 2.818e+15 [cm**-2]; EW = 1.139 [Angstrom]
       OVI1031
        N = 7.536e+14 [cm**-2]; EW = 0.909 [Angstrom]
        N = 7.528e+14 [cm**-2]; EW = 0.538 [Angstrom]
        N = 7.543e+14 [cm**-2]; EW = 0.530 [Angstrom]
    l.o.s.: [35000. 35600.] [ckpc h_0**-1]
       H1215
        N = 4.816e+13 [cm**-2]; EW = 0.312 [Angstrom]
        N = 4.811e+13 [cm**-2]; EW = 0.306 [Angstrom]
        N = 4.821e+13 [cm**-2]; EW = 0.306 [Angstrom]
       OVI1031
        N = 1.098e+14 [cm**-2]; EW = 0.184 [Angstrom]
        N = 1.097e+14 [cm**-2]; EW = 0.167 [Angstrom]
        N = 1.099e+14 [cm**-2]; EW = 0.167 [Angstrom]
    >>> environment.verbose = environment.VERBOSE_NORMAL
"""
__all__ = [
    "mock_absorption_spectrum_of",
    "mock_absorption_spectrum",
    "EW",
    "Voigt",
    "Gaussian",
    "Lorentzian",
    "thermal_b_param",
    "line_profile",
    "curve_of_growth",
    "fit_Voigt",
    "find_line_contributers",
    "velocities_to_redshifts",
    "redshifts_to_velocities",
]

from ..units import Unit, UnitArr, UnitQty, UnitScalar
from ..physics import kB, m_H, c, q_e, m_e, epsilon0
from ..kernels import *
from .. import gadget
from .. import utils
from .. import C
from .. import environment
import numpy as np
from scipy.special import wofz
from numbers import Number

lines = {
    "H1215": {
        "ion": "HI",
        "l": "1215.6701 Angstrom",
        "f": 0.4164,
        "atomwt": m_H,
        "A_ki": "4.6986e+08 s**-1",
        "element": "H",
    },
    "H1025": {
        "ion": "HI",
        "l": "1025.722 Angstrom",
        "f": 0.079121,
        "atomwt": m_H,
        "A_ki": "5.5751e+07 s**-1",
        "element": "H",
    },
    "H972": {
        "ion": "HI",
        "l": "972.5368 Angstrom",
        "f": 2.900e-02,
        "atomwt": m_H,
        "A_ki": "1.2785e+07 s**-1",
        "element": "H",
    },
    "HeII": {
        "ion": "HeII",
        "l": "303.918 Angstrom",
        "f": 0.4173,
        "atomwt": "3.971 u",
        "element": "He",
    },
    "CII1036": {
        "ion": "CII",
        "l": "1036.337 Angstrom",
        "f": 0.1270,
        "atomwt": "12.011 u",
        "element": "C",
    },
    "CII1334": {
        "ion": "CII",
        "l": "1334.532 Angstrom",
        "f": 0.1270,
        "atomwt": "12.011 u",
        "element": "C",
    },
    "CII1335": {
        "ion": "CII",
        "l": "1335.708 Angstrom",
        "f": 0.1140,
        "atomwt": "12.011 u",
        "element": "C",
    },
    "CIII977": {
        "ion": "CIII",
        "l": "977.020 Angstrom",
        "f": 0.7620,
        "atomwt": "12.011 u",
        "element": "C",
    },
    "CIV1548": {
        "ion": "CIV",
        "l": "1548.195 Angstrom",
        "f": 0.1908,
        "atomwt": "12.011 u",
        "element": "C",
    },
    "CIV1550": {
        "ion": "CIV",
        "l": "1550.777 Angstrom",
        "f": 0.09520,
        "atomwt": "12.011 u",
        "element": "C",
    },
    "NI1199": {
        "ion": "NI",
        "l": "1199.550 Angstrom",
        "f": 0.1300,
        "atomwt": "14.0067 u",
        "element": "N",
    },
    "NI1200": {
        "ion": "NI",
        "l": "1200.223 Angstrom",
        "f": 0.0862,
        "atomwt": "14.0067 u",
        "element": "N",
    },
    "NI1201": {
        "ion": "NI",
        "l": "1200.710 Angstrom",
        "f": 0.043,
        "atomwt": "14.0067 u",
        "element": "N",
    },
    "NII1083": {
        "ion": "NII",
        "l": "1083.994 Angstrom",
        "f": 0.1150,
        "atomwt": "14.0067 u",
        "element": "N",
    },
    "NV1238": {
        "ion": "NV",
        "l": "1238.821 Angstrom",
        "f": 0.1560,
        "atomwt": "14.0067 u",
        "element": "N",
    },
    "NV1242": {
        "ion": "NV",
        "l": "1242.804 Angstrom",
        "f": 0.0780,
        "atomwt": "14.0067 u",
        "element": "N",
    },
    "OI1302": {
        "ion": "OI",
        "l": "1302.168 Angstrom",
        "f": 0.05190,
        "atomwt": "15.9994 u",
        "element": "O",
    },
    "OI1304": {
        "ion": "OI",
        "l": "1304.858 Angstrom",
        "f": 0.04877,
        "atomwt": "15.9994 u",
        "element": "O",
    },
    "OI1306": {
        "ion": "OI",
        "l": "1306.029 Angstrom",
        "f": 0.04873,
        "atomwt": "15.9994 u",
        "element": "O",
    },
    "OIV787": {
        "ion": "OIV",
        "l": "787.711 Angstrom",
        "f": 0.110,
        "atomwt": "15.9994 u",
        "element": "O",
    },
    "OVI1031": {
        "ion": "OVI",
        "l": "1031.927 Angstrom",
        "f": 0.1329,
        "atomwt": "15.9994 u",
        "element": "O",
    },
    "OVI1037": {
        "ion": "OVI",
        "l": "1037.617 Angstrom",
        "f": 0.06590,
        "atomwt": "15.9994 u",
        "element": "O",
    },
    "OVII21": {
        "ion": "OVII",
        "l": "21.602 Angstrom",
        "f": 0.696,
        "atomwt": "15.9994 u",
        "element": "O",
    },
    "OVIII19": {
        "ion": "OVIII",
        "l": "18.969 Angstrom",
        "f": 0.416,
        "atomwt": "15.9994 u",
        "element": "O",
    },
    "NeVIII770": {
        "ion": "NeVIII",
        "l": "770.409 Angstrom",
        "f": 0.103,
        "atomwt": "20.180 u",
        "element": "Ne",
    },
    "MgII2796": {
        "ion": "MgII",
        "l": "2796.352 Angstrom",
        "f": 0.6123,
        "atomwt": "24.305 u",
        "element": "Mg",
    },
    "SiII1190": {
        "ion": "SiII",
        "l": "1190.416 Angstrom",
        "f": 0.2930,
        "atomwt": "28.086 u",
        "element": "Si",
    },
    "SiII1193": {
        "ion": "SiII",
        "l": "1193.290 Angstrom",
        "f": 0.5850,
        "atomwt": "28.086 u",
        "element": "Si",
    },
    "SiII1260": {
        "ion": "SiII",
        "l": "1260.522 Angstrom",
        "f": 1.180,
        "atomwt": "28.086 u",
        "element": "Si",
    },
    "SiIII1206": {
        "ion": "SiIII",
        "l": "1206.500 Angstrom",
        "f": 1.669,
        "atomwt": "28.086 u",
        "element": "Si",
    },
    "SiIV1393": {
        "ion": "SiIV",
        "l": "1393.755 Angstrom",
        "f": 0.5140,
        "atomwt": "28.086 u",
        "element": "Si",
    },
    "SiIV1402": {
        "ion": "SiIV",
        "l": "1402.770 Angstrom",
        "f": 0.2553,
        "atomwt": "28.086 u",
        "element": "Si",
    },
    "SI1295": {
        "ion": "SI",
        "l": "1295.653 Angstrom",
        "f": 0.08700,
        "atomwt": "32.065 u",
        "element": "S",
    },
    "SI1425": {
        "ion": "SI",
        "l": "1425.030 Angstrom",
        "f": 0.1250,
        "atomwt": "32.065 u",
        "element": "S",
    },
    "SI1473": {
        "ion": "SI",
        "l": "1473.994 Angstrom",
        "f": 0.08280,
        "atomwt": "32.065 u",
        "element": "S",
    },
    "SII1250": {
        "ion": "SII",
        "l": "1250.584 Angstrom",
        "f": 0.00543,
        "atomwt": "32.065 u",
        "element": "S",
    },
    "SII1253": {
        "ion": "SII",
        "l": "1253.811 Angstrom",
        "f": 0.01090,
        "atomwt": "32.065 u",
        "element": "S",
    },
    "SII1259": {
        "ion": "SII",
        "l": "1259.519 Angstrom",
        "f": 0.01660,
        "atomwt": "32.065 u",
        "element": "S",
    },
    "SIII1190": {
        "ion": "SIII",
        "l": "1190.203 Angstrom",
        "f": 0.02310,
        "atomwt": "32.065 u",
        "element": "S",
    },
}
lines["Lyman_alpha"] = lines["H1215"]
lines["Lyman_beta"] = lines["H1025"]
lines["Lyman_gamma"] = lines["H972"]


def line_quantity(line, qty):
    return lines[line][str(qty)]


def Gaussian(x, sigma):
    """
    The Gaussian function:

                              1                /    1     x^2    \
    G (x; sigma)  =  --------------------  exp | - --- --------- |
                      sigma sqrt( 2 pi )       \    2   sigma^2  /

    which is normed to 1.

    Args:
        x (float, np.ndarray):  The argument of the Gaussian function.
        sigma (float):          The standard deviation of the Gaussian.

    Returns:
        y (float, np.ndarray):  The value(s) of the Gaussian.
    """
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def Lorentzian(x, gamma):
    """
    The Lorentz function:

                             gamma
    L (x; gamma)  =  ----------------------
                      pi ( x^2 + gamma^2 )

    which is normed to 1.

    Args:
        x (float, np.ndarray):  The argument of the Gauss function.
        gamma (float):          The standard deviation of the Gaussian.

    Returns:
        y (float, np.ndarray):  The value(s) of the Gaussian.
    """
    return gamma / (np.pi * (x ** 2 + gamma ** 2))


def Voigt(x, sigma, gamma):
    """
    The Voigt function.

    It is defined as a convolution of a Gaussian and a Lorentz function:

                           oo
                           /\
    V (x; gamma, sigma) =  |  dx' G(x';sigma) L(x-x';gamma)
                          \/
                         -oo

    which is normed to one, since G and L are normed to 1.

    Args:
        x (float, np.ndarray):  The argument of the Voigt function.
        sigma (float):          The standard deviation of the Gaussian.
        gamma (float):          The gamma value of the Lorentz function.

    Returns:
        y (float, np.ndarray):  The value(s) of the Voigt profile.
    """
    z = (x + 1j * gamma) / (sigma * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def thermal_b_param(line, T, units="km/s"):
    """Calculate the thermal Doppler b-parameter for given line and temperature."""
    if isinstance(line, str):
        line = lines[line]
    atomwt = UnitScalar(line["atomwt"])
    b = np.sqrt(2.0 * kB * T / atomwt).in_units_of(units)
    return b


def line_profile(
    line, N, T=None, b=None, l0=None, l=None, lim=None, bins=1000, mode="Voigt"
):
    """
    Calculate the theoretical line profile of a non-moving slice of
    homogenous gas with constant temperature and given column density.

    Args:
        line (str, dict):
                        The line name as listed in
                        `analysis.absorption_spectra.lines` or an analogous
                        dictionary.
        N (UnitScalar): The column density of the slice. Can either bin in
                        particles per area (e.g. 'cm**-2') or in mass per
                        area (e.g. 'g cm**-2').
        T (UnitScalar): The temperature of the slice.
        b (UnitScalar): The b-parameter. If give, `T` is ignore, which
                        otherwise would translate into a thermal b-parameter.
                        (Units default to 'km/s'.)
        l0 (UnitScalar):The line centroid. Defaults to the restframe wavelength
                        of the given line.
        l (UnitQty):    The wavelengths to calculate the optical depths for. If
                        this is None, create the wavelength array from the
                        following two parameters.
        lim (UnitQty):  The limits of the spectrum in wavelength. If one of
                        the values is negative, they are taken to be realtive
                        to the line center.
                        Defaults to +-5 Angstrom around the line center.
        bins (int):     The number of evenly spread points in the
                        spectrum.
        mode (str):     What profile to use:
                        'Gaussian', 'thermal', 'Doppler':
                            Take just the thermal broadening into account,
                            which produces a Gaussian profile of the optical
                            depths.
                        'Lorentzian', 'natural', 'intrinsic':
                            Take just the natural/instrinsic line width into
                            account, which produces a Lorentzian profile of
                            the optical depth.
                        'Voigt', 'full':
                            Generate the full Voigt profile of the convolution
                            of the above two profiles.

    Returns:
        l (UnitArr):        The wavelengths of the sprectrum (is input `l` if
                            given).
        tau (np.ndarray):   The optical depth at the given wavelengths.
    """
    if isinstance(line, str):
        line = lines[line]
    f = float(line["f"])
    atomwt = UnitScalar(line["atomwt"]) if "atomwt" in line else None
    A_ki = UnitScalar(line.get("A_ki", 0.0), "Hz")
    if l0 is None:
        l0 = line["l"]
    l0 = UnitScalar(l0, "Angstrom")
    lim = UnitQty([-5, 5] if lim is None else lim, "Angstrom", dtype=float)
    if np.any(lim < 0):
        lim += l0
    N = UnitScalar(N)
    try:
        N = N.in_units_of("cm**-2")
    except:
        N = (N / atomwt).in_units_of("cm**-2")
    sigma0 = f * q_e ** 2 / (4.0 * epsilon0 * m_e * c)
    sigma0.convert_to("cm**2 / s")
    if b is None:
        T = UnitScalar(T, "K")
        b = thermal_b_param(line, T, "km/s")
    else:
        b = UnitScalar(b, "km/s")

    if l is None:
        l = UnitArr(np.linspace(float(lim[0]), float(lim[1]), bins), lim.units)
    else:
        l = UnitQty(l, "Angstrom")
    nu = (c / l).in_units_of("Hz")
    nu0 = (c / l0).in_units_of("Hz")
    x = (nu - nu0).view(np.ndarray)
    sigma = (b / l0 / np.sqrt(2)).in_units_of("Hz").view(np.ndarray)
    gamma = (A_ki / (4.0 * np.pi)).in_units_of("Hz").view(np.ndarray)
    if mode in ["Voigt", "full"]:
        phi = Voigt(x, sigma, gamma)
    elif mode in ["Gaussian", "thermal", "Doppler"]:
        phi = Gaussian(x, sigma)
    elif mode in ["Lorentzian", "natural", "intrinsic"]:
        phi = Lorentzian(x, gamma)
    else:
        raise ValueError('Unknown mode "%s"' % mode)
    phi = UnitArr(phi, "1/Hz")
    tau = sigma0 * phi * N
    tau.convert_to(1)

    return l, tau.view(np.ndarray)


def curve_of_growth(line, b, Nlim=(10, 21), bins=30, mode="Voigt"):
    """
    Calculate the curve of growth for homogeneous gas.

    Args:
        line (str, dict):   The line to calculate the curve of growth for
                            as listed in `analysis.absorption_spectra.lines` or
                            an analogous dictionary.
        b (UnitScalar):     The b-parameter which is relevant in the flat part.
                            You might want to use `thermal_b_param` here.
        Nlim (arraly-like): The limits in logarithmic column density in units of
                            particles per cm**2.
        bins (int):         The number of evaltuation points in the given range
                            of column densities (equi-distant in log-space).
        mode (str):         What profile to use:
                            'Gaussian', 'thermal', 'Doppler':
                                Take just the thermal broadening into account,
                                which produces a Gaussian profile of the optical
                                depths.
                            'Lorentzian', 'natural', 'intrinsic':
                                Take just the natural/instrinsic line width into
                                account, which produces a Lorentzian profile of
                                the optical depth.
                            'Voigt', 'full':
                                Generate the full Voigt profile of the
                                convolution of the above two profiles.

    Returns:
        N (UnitQty):        The (linear) column densities at which the EW were
                            evaluated.
        EW (UnitQty):       The equivalent widths at the given column densities.
    """
    if isinstance(line, str):
        line = lines[line]
    l0 = UnitScalar(line["l"])
    N = UnitArr(np.logspace(float(Nlim[0]), float(Nlim[1]), int(bins)), "cm**-2")
    ew = UnitArr(np.empty(len(N)), "Angstrom")
    for i, N_ in enumerate(N):
        lim, bins = [-0.5, 0.5], 300
        tau = [1, 1]
        while tau[0] > 1e-3 or tau[-1] > 1e-3:
            lim_ = l0 + np.array(lim)
            l, tau = line_profile(
                line, "%g cm**-2" % N_, b=b, lim=lim_, bins=bins, mode=mode
            )
            lim, bins = [2 * lim[0], 2 * lim[1]], 2 * bins
        dl = UnitArr(l[1] - l[0], l.units)
        l_edges = UnitArr(np.empty(len(l) + 1), l.units)
        l_edges[:-1] = l - dl / 2.0
        l_edges[-1] = l_edges[-2] + dl
        ew[i] = EW(tau, l_edges)
    return N, ew


def fit_Voigt(l, flux, line, Nlim=(8, 22), blim=(0, 200), bins=(57, 41)):
    """
    Fit a single Voigt profile to the given line.

    Args:
        l (UnitQty):        The wavelengths of the given relative fluxes.
                            Shall be equi-distant positions.
        flux (array-like):  The relative flux at the given wavelengths, i.e.
                            the spectrum to fit.
        line (str):         The line to fit as listed in
                            `analysis.absorption_spectra.lines`.
        Nlim (array-like):  The limits in logarithmic column density to test
                            in units of log10(cm**-2).
        blim (UnitQty):     The b-parameter limits to test.
        bins (int, tuple):  The number of bins in column density (log-spaced)
                            and b-parameter (lin-spaced).

    Returns:
        N (UnitScalar):     The best fit column density.
        b (UnitScalar):     The best fit b-parameter (of the Voigt profile).
        dl (UnitScalar):    The shift from the rest-frame wavelength of the fit.
        errs (UnitQty):     The integrateed squared differences of fit
                            and given spectrum.
    """
    if isinstance(line, str):
        line = lines[line]
    Nlim = np.array(Nlim)
    blim = UnitQty(blim, "km/s")
    l = UnitQty(l, "Angstrom")
    flux = np.array(flux, dtype=float)
    if isinstance(bins, int):
        bins = [bins, bins]
    bins = np.array(bins, dtype=int)
    if not bins.shape == (2,):
        raise ValueError("`bins` needs to be scalar or 2-tuple.")

    l0_line = UnitArr(line["l"], l.units)
    l0 = np.average(l, weights=1.0 - flux)
    dl = l0_line - l0
    l_lim = UnitArr([l[0] + dl, l[-1] + dl], l.units)

    def err(N, b):
        _, tau = line_profile(line, N, b=b, lim=l_lim, bins=len(flux))
        return np.sum((np.exp(-tau) - flux) ** 2)

    errs = np.empty(tuple(bins), dtype=float)
    Ns = np.logspace(float(Nlim[0]), float(Nlim[1]), bins[0])
    bs = np.linspace(float(blim[0]), float(blim[1]), bins[1])
    for i, N in enumerate(Ns):
        for j, b in enumerate(bs):
            errs[i, j] = err(N, b)
    finite = np.isfinite(errs)
    errs[~finite] = np.inf

    i = np.unravel_index(np.argmin(errs), errs.shape)
    N, b = Ns[i[0]], bs[i[1]]

    return UnitArr(N, "cm**-2"), UnitArr(b, "km/s"), dl, errs * dl


def find_line_contributers(
    s, los, line, vel_extent, threshold=0.95, EW_space="wavelength", **kwargs
):
    """
    Find the minimally required particles to generate the specified line without
    having the equivalent width falling below a threshold.

    Args:
        s (Snap):               The snapshot to use for the line creation.
        los (UnitQty):          The position of the l.o.s.. By default understood
                                as in units of s['pos'], if not explicitly
                                specified.
        line (str, dict):       The line to generate. It can either be a name in
                                `analysis.absorption_spectra.lines` of a
                                dictionary alike one of these.
        vel_extent (UnitQty):   The limits of the spectrum in (rest frame)
                                velocity space for the spectrum to create. Units
                                default to 'km/s'.
        threshold (float):      The threshold which the EW must not fall below in
                                fractions of the EW of the line created by all
                                particles.
        EW_space (str):         The space in which to calculate the equivalent
                                width, i.e. in which space it shall be integrated
                                over exp(-tau) in order to calculate EW. Possible
                                choices are:
                                'wavelength', 'frequency', 'redshift', 'velocity'.
        kwargs:                 Further arguments are passed to
                                `mock_absorption_spectrum_of`.

    Returns:
        contributing (nd.ndarray<bool>):
                                A booling mask for all the (gas) particles in `s`
                                that are needed for the EW not falling below the
                                specified threshold.
    """
    if isinstance(line, str):
        line = str(line)
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print(
            "find all necessary particles, beginning with those that have "
            + "the highest column density along the line of sight, that are "
            + "needed for getting %.1f%% of the total EW" % (100.0 * threshold)
        )
        if isinstance(line, str):
            print('  line "%s" at %s' % (line, los))

    if isinstance(line, str):
        line = lines[line]
    taus, dens, temp, v_edges, restr_column = mock_absorption_spectrum_of(
        s.gas, los=los, line=line, vel_extent=vel_extent, **kwargs
    )
    N_intersecting = np.sum(restr_column > 0)

    z_edges = velocities_to_redshifts(v_edges, z0=s.redshift)
    l_edges = UnitScalar(line["l"]) * (1.0 + z_edges)
    if EW_space == "wavelength":
        edges = l_edges
    elif EW_space == "frequency":
        edges = -(c / l_edges).in_units_of("Hz")
    elif EW_space == "redshift":
        edges = z_edges
    elif EW_space == "velocity":
        edges = v_edges
    else:
        raise ValueError('Unknown `EW_space`: "%s"!' % EW_space)
    EW_full = EW(taus, edges)
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print("in %s space EW = %s" % (EW_space, EW_full))

    # bisect by percentiles
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print("finding the necessary particles...")
    low, mid, high = 0.0, 50.0, 100.0
    Nlow, Nmid, Nhigh = [np.percentile(restr_column, x) for x in [low, mid, high]]
    verbosity = environment.verbose
    environment.verbose = environment.VERBOSE_QUIET
    while np.sum(restr_column > Nlow) > np.sum(restr_column > Nhigh) + 1:
        mid = (low + high) / 2.0
        Nmid = np.percentile(restr_column, mid)
        taus, _, _, _, _ = mock_absorption_spectrum_of(
            s.gas[restr_column > Nmid],
            los=los,
            line=line,
            vel_extent=vel_extent,
            **kwargs
        )
        E = EW(taus, edges)
        if E < threshold * EW_full:
            high, Nhigh = mid, Nmid
        else:
            low, Nlow = mid, Nmid
    environment.verbose = verbosity
    contributing = np.array((restr_column > Nmid), dtype=bool)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print(
            "%s of the %s N_intersecting particles needed "
            % (
                utils.nice_big_num_str(np.sum(contributing)),
                utils.nice_big_num_str(N_intersecting),
            )
            + "for a line with >= %.1f%% of the EW" % (100.0 * threshold)
        )

    return contributing


def mock_absorption_spectrum_of(s, los, line, vel_extent, **kwargs):
    """
    Create a mock absorption spectrum for the given line of sight (l.o.s.) for the
    given line transition.

    This function basically just calls `mock_absorption_spectrum` for the given
    line:

        if isinstance(line, str):
            line = lines[line]
        return mock_absorption_spectrum(s, los, line['ion'],
                                        l=line['l'], f=line['f'],
                                        atomwt=line['atomwt'],
                                        vel_extent=vel_extent,
                                        A_ki=line.get('A_ki',0.0),
                                        **kwargs)
    """
    if isinstance(line, str):
        line = str(line)
    try:
        if isinstance(line, str):
            line = lines[line]
        elif not isinstance(line, dict):
            raise ValueError(
                "`line` needs to be a string or a dictionary, " + "not %s!" % type(line)
            )
    except KeyError:
        raise KeyError(
            "unkown line '%s' -- " % line
            + "see `analysis.absorption_spectra.lines.keys()`"
        )
    return mock_absorption_spectrum(
        s,
        los,
        line["ion"],
        l=line["l"],
        f=line["f"],
        atomwt=line["atomwt"],
        vel_extent=vel_extent,
        A_ki=line.get("A_ki", 0.0),
        element=line["element"],
        **kwargs
    )


def mock_absorption_spectrum(
    s,
    los,
    ion,
    l,
    f,
    atomwt,
    vel_extent,
    Nbins=1000,
    A_ki="0 s**-1",
    element=None,
    v_turb=None,
    method="particles",
    spatial_extent=None,
    spatial_res=None,
    col_width=None,
    pad=7,
    hsml="hsml",
    kernel=None,
    restr_column_lims=None,
    zero_Hubble_flow_at=0,
    xaxis=0,
    yaxis=1,
    return_los_phys=False,
):
    """
    Create a mock absorption spectrum for the given line of sight (l.o.s.) for the
    given line transition.

    Credits to Neal Katz and Romeel Dave, who wrote a code taken as a basis for
    this one, first called 'specexbin' and later 'specexsnap' that did the same
    as in the method 'line', and who helped me with the gist of this one.

    TODO:
        Check the "redshift correction of the wavelength difference"! Cf.
        Formular (5) in Churchill+(2015).

    Args:
        s (Snap):               The snapshot to shoot the l.o.s. though.
        los (UnitQty):          The position of the l.o.s.. By default understood
                                as in units of s['pos'], if not explicitly
                                specified.
        ion (str, UnitQty):     The block for the masses of the ion that generates
                                the line asked for (e.g. HI for Lyman alpha or CIV
                                for CIV1548).
                                If given as a UnitQty without units, they default
                                to those of the 'mass' block.
        l (UnitScalar):         The wavelength of the line transition. By default
                                understood in Angstrom.
        f (float):              The oscillatr strength of the line transition.
        atomwt (UnitScalar):    The atomic weight. By default interpreted in
                                atomic mass units.
        vel_extent (UnitQty):   The limits of the spectrum in (rest frame)
                                velocity space. Units default to 'km/s'.
        Nbins (int):            The number of bins for the spectrum.
        A_ki (UnitScalar):      The transition probability / Einstein coefficient.
                                The units default to 's**-1'.
        element (str):          The element of the ion, used to find metal mass
                                fraction.
        v_turb (UnitScalar, str, UnitQty):
                                A turbulent velocity (constant or individiual per
                                particle) which adds to the thermal broadening. If
                                it is not a scalar (or None) it has to be
                                block-like (array-like of appropiate size or a
                                string) for the particles.
                                This argument only is used for the 'particles'
                                method and is ignored otherwise.
        method (str):           How to do the binning into velocity space. The
                                available choices are:
                                * 'particles':  Create a line for each particle
                                                individually and then add them up.
                                * 'line':       First create a infinitesimal line
                                                along the l.o.s. in position space
                                                and bin the SPH quantities onto
                                                that, then for each of these bins
                                                create a line and add those up.
                                * 'column':     Same as the 'line' method, but use
                                                a square column with finite
                                                thickness (of `col_width`).
                                Note that the ionisation fractions are always
                                calculated on the particle basis, which yields to
                                inconsitencies (eps. in the thermal broadening,
                                which is done with the ion mass-weighted
                                temperature of the spatial bins, not the particle
                                temperature). This is a problem in multi-phase SPH
                                and with wind particles.
                                The particle variant, however, does not properly
                                capture sheer flows and the Hubble flow within a
                                particle.
        spatial_extent (UnitQty):
                                Ignored, if method=='particles'.
                                The extent in the spatial bins along the l.o.s..
                                If not units are provided, it is assumed it is
                                given in those of s['pos'].
        spatial_res (UnitScalar):
                                Ignored, if method=='particles'.
                                The resolution of the spatial bins. If not units
                                are provided, it is assumed it is given in those
                                of s['pos']. Defaults to the 0.1% quantile of the
                                smoothing lengths of the given snapshot.
        col_width (UnitScalar): Ignored, if method!='column'.
                                Defines the side length of the column.
        pad (int):              Ignored, if method!='column'.
                                Pad this number of voxels into each direction
                                around the column. Needed in order to make use of
                                the S-normation in the 3D binning (ensuring
                                integral conservation); see SPH_to_3Dgrid for more
                                information.
        hsml (str, UnitQty, Unit):
                                The smoothing lengths to use. Can be a block name,
                                a block itself or a Unit that is taken as constant
                                volume for all particles.
        kernel (str):           The kernel to use for smoothing. (By default use
                                the kernel defined in `gadget.cfg`.)
        restr_column_lims (UnitQty):
                                The velocity limits for a window of interest. For
                                each particle / cell the column density that
                                contributes to absorption in this window is
                                calculated and returned as `restr_column`.
                                By default the same as `vel_extent`.
        zero_Hubble_flow_at (UnitScalar):
                                The position along the l.o.s. where there is no
                                Hubble flow. If not units are given, they are
                                assume to be those of s['pos'].
        xaxis/yaxis (int):      The x- and y-axis for the l.o.s.. The implicitly
                                defined z-axis goes along the l.o.s.. The axis
                                must be chosen from [0,1,2].
        return_los_phys (Bool): Returns additional LOS info, currently
                                los_phys_dens (optical depth-weighted density), and
                                vel (optical depth-weighted peculiar velocity)

    Returns:
        taus (np.ndarray):      The optical depths for the velocity bins.
        los_dens (UnitArr):     The column densities restricted to the velocity
                                bins (in cm^-2).
        los_dens_phys (UnitArr):The gas density for the velocity bins (in g cm^-3)
        los_temp (UnitArr):     The (mass-weighted) particle temperatures
                                restricted to the velocity bins (in K).
        vel (UnitArr):          The LOS velocities of particles (in km/s)
        v_edges (UnitArr):      The velocities at the bin edges.
        restr_column (np.ndarray):
                                The column densities of the particles/cells along
                                the line of sight that contributes to the given
                                window of interest defined by `restr_column_lims`.
    """
    # internally used units
    v_units = Unit("km/s")
    l_units = Unit("cm")

    if isinstance(ion, str):
        ion = str(ion)
    zaxis = (set([0, 1, 2]) - set([xaxis, yaxis])).pop()
    if set([xaxis, yaxis, zaxis]) != set([0, 1, 2]):
        raise ValueError("x- and y-axis must be in [0,1,2] and different!")
    los = UnitQty(los, s["pos"].units, dtype=np.float64, subs=s)
    zero_Hubble_flow_at = UnitScalar(zero_Hubble_flow_at, s["pos"].units, subs=s)
    vel_extent = UnitQty(vel_extent, "km/s", dtype=np.float64, subs=s)
    if restr_column_lims is None:
        restr_column_lims = vel_extent.copy()
    else:
        restr_column_lims = UnitQty(restr_column_lims, "km/s", dtype=np.float64, subs=s)
    l = UnitScalar(l, "Angstrom")
    if v_turb is not None:
        if isinstance(v_turb, str):
            try:
                v_turb = UnitScalar(v_turb)
            except:
                v_turb = s.gas.get(v_turb)
        v_turb = UnitQty(v_turb, "km/s", dtype=np.float64)
        if v_turb.shape == tuple():
            v_turb = UnitQty(
                float(v_turb) * np.ones(len(s.gas), dtype=np.float64),
                units=getattr(v_turb, "units", "km/s"),
            )
        assert v_turb.shape == (len(s.gas),)
        assert v_turb.units == "km/s"
        if np.all(v_turb == 0):
            v_turb = None
    if method != "particles":
        if v_turb is not None:
            v_turb = None
    A_ki = UnitScalar(A_ki, "s**-1", dtype=float)
    f = float(f)
    atomwt = UnitScalar(atomwt, "u")
    if kernel is None:
        kernel = gadget.general["kernel"]

    # natural line width in frequency space, when using a Lorentzian
    # defined as: L(f) = 1/pi * (Gamma / (f**2 + Gamma**2))
    Gamma = A_ki / (4.0 * np.pi)
    # ...and the velocity needed for an appropiate redshift
    # (i.e. a conversion of the width to velocity space)
    # Gamma = (Gamma / (c/l)) * c   # c/l = f
    Gamma = (Gamma * l).in_units_of(v_units, subs=s)

    b_0 = np.sqrt(2.0 * kB * UnitScalar("1 K") / atomwt)
    b_0.convert_to(v_units)
    s0 = q_e ** 2 / (4.0 * epsilon0 * m_e * c)
    Xsec = f * s0 * l
    Xsec = Xsec.in_units_of(l_units ** 2 * v_units, subs=s)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print("create a mock absorption spectrum:")
        print("  at", los)
        if isinstance(ion, str):
            print("  for", ion, "at lambda =", l)
        else:
            print("  at lambda =", l)
        print("  with oscillator strength f =", f)
        print("  => Xsec =", Xsec)
        print("  and atomic weight", atomwt)
        print("  => b(T=1e4K) =", b_0 * np.sqrt(1e4))
        print("  and a lifetime of 1/A_ki =", (1.0 / A_ki))
        print("  => Gamma =", Gamma)
        if v_turb is not None:
            v_perc = np.percentile(v_turb, [10, 90])
            print(
                "  and a turbulent motion per particle of v_turb ~(%.1f - %.1f) %s"
                % (v_perc[0], v_perc[-1], v_turb.units)
            )
        print('  using kernel "%s"' % kernel)

    v_edges = UnitArr(
        np.linspace(float(vel_extent[0]), float(vel_extent[1]), Nbins + 1),
        vel_extent.units,
    )

    # get ne number of ions per particle
    if isinstance(ion, str):
        ion = s.gas.get(ion)
    else:
        ion = UnitQty(ion, units=s["mass"].units, subs=s)
    # double precision needed in order not to overflow
    # 1 Msol / 1 u = 1.2e57, float max = 3.4e38, but double max = 1.8e308
    n = (ion.astype(np.float64) / atomwt).in_units_of(1, subs=s)
    # n = n.view(np.ndarray).astype(np.float64)

    if method != "particles":
        # do SPH smoothing along the l.o.s.
        if spatial_extent is None:
            spatial_extent = [
                np.min(s.gas["pos"][:, zaxis]),
                np.max(s.gas["pos"][:, zaxis]),
            ]
            spatial_extent = UnitArr(spatial_extent, spatial_extent[-1].units)
            if 1.01 * s.boxsize > spatial_extent.ptp() > 0.8 * s.boxsize:
                # the box seems to be full with gas
                missing = s.boxsize - spatial_extent.ptp()
                spatial_extent[0] -= missing / 2.0
                spatial_extent[1] += missing / 2.0
            spatial_extent.convert_to(s["pos"].units, subs=s)
        else:
            spatial_extent = UnitQty(spatial_extent, s["pos"].units, subs=s)

        if spatial_res is None:
            spatial_res = UnitArr(np.percentile(s.gas["hsml"], 1), s.gas["hsml"].units)
        spatial_res = UnitScalar(spatial_res, s["pos"].units, subs=s)
        N = int(max(1e3, (spatial_extent.ptp() / spatial_res).in_units_of(1, subs=s)))
        spatial_res == spatial_extent.ptp() / N

        if method == "column":
            # do some padding in the 3D binning in order to use the the normation
            # process
            pad = int(pad)
            Npx = (1 + 2 * pad) * np.ones(3, dtype=np.int)
            Npx[zaxis] = N
            # mask for getting the middle column of interest
            m = [pad] * 3
            m[zaxis] = slice(None)

            if col_width is None:
                col_width = spatial_res
            col_width = UnitScalar(col_width, s["pos"].units, subs=s)
            w = ((0.5 + 2.0 * pad) * col_width).in_units_of(los.units, subs=s)
            extent = UnitArr(np.empty((3, 2), dtype=float), los.units)
            extent[xaxis] = [los[0] - w, los[0] + w]
            extent[yaxis] = [los[1] - w, los[1] + w]
            extent[zaxis] = spatial_extent

            binargs = {
                "extent": extent,
                "Npx": Npx,
                "kernel": kernel,
                "dV": "dV",
                "hsml": hsml,
                "normed": True,
            }

            # restrict to particles intersecting the l.o.s. column:
            sub = s.gas[
                (s.gas["pos"][:, xaxis] - s.gas["hsml"] < los[0] + col_width)
                & (s.gas["pos"][:, xaxis] + s.gas["hsml"] > los[0] - col_width)
                & (s.gas["pos"][:, yaxis] - s.gas["hsml"] < los[1] + col_width)
                & (s.gas["pos"][:, yaxis] + s.gas["hsml"] > los[1] - col_width)
            ]

            dV = sub["dV"].in_units_of(sub["pos"].units ** 3)

            if environment.verbose >= environment.VERBOSE_NORMAL:
                print("  using an spatial extent of:", spatial_extent)
                print(
                    "  ... with %d bins of size %sx%s^2" % (N, col_width, spatial_res)
                )

            from ..binning import SPH_to_3Dgrid

            def bin_func(s, qty, **args):
                Q = SPH_to_3Dgrid(sub, qty, **args)
                Q, px = Q[m].reshape(N) * Q.vol_voxel(), Q.res()
                return Q, px

            n_parts = n[sub._mask]
            n, px = bin_func(sub, n_parts / dV, **binargs)
            non0n = n != 0
            vel, px = bin_func(sub, n_parts * sub["vel"][:, zaxis] / dV, **binargs)
            vel[non0n] = vel[non0n] / n[non0n]

            rho, px = bin_func(sub, n_parts * sub["rho"] / dV, **binargs)
            if element is not None and element != "H" and element != "He":
                metal_frac, px = bin_func(
                    sub, n_parts * sub.gas[element] / sub.gas["mass"] / dV, **binargs
                )
            else:
                metal_frac, px = bin_func(
                    sub, n_parts * sub.gas["metallicity"] / dV, **binargs
                )

            # average sqrt(T), since thats what the therm. broadening scales with
            temp, px = bin_func(sub, n_parts * np.sqrt(sub["temp"]) / dV, **binargs)
            temp[non0n] = temp[non0n] / n[non0n]
            temp **= 2
            # we actually need the column densities, not the number of particles
            n /= np.prod(px[[xaxis, yaxis]])

            # the z-coordinates for the Hubble flow
            los_pos = UnitArr(
                np.linspace(
                    float(spatial_extent[0]), float(spatial_extent[1] - px[zaxis]), N
                ),
                spatial_extent.units,
            )
        elif method == "line":
            binargs = {
                "los": los,
                "extent": spatial_extent,
                "Npx": N,
                "kernel": kernel,
                "dV": "dV",
                "hsml": hsml,
                "xaxis": xaxis,
                "yaxis": yaxis,
            }

            # restrict to particles intersecting the l.o.s.:
            sub = s.gas[
                (s.gas["pos"][:, xaxis] - s.gas["hsml"] < los[0])
                & (s.gas["pos"][:, xaxis] + s.gas["hsml"] > los[0])
                & (s.gas["pos"][:, yaxis] - s.gas["hsml"] < los[1])
                & (s.gas["pos"][:, yaxis] + s.gas["hsml"] > los[1])
            ]

            dV = sub["dV"].in_units_of(sub["pos"].units ** 3)

            if environment.verbose >= environment.VERBOSE_NORMAL:
                print("  using an spatial extent of:", spatial_extent)
                print("  ... with %d bins of length %s" % (N, spatial_res))

            from ..binning import SPH_3D_to_line

            bin_func = SPH_3D_to_line

            def bin_func(s, qty, **args):
                Q = SPH_3D_to_line(sub, qty, **args)
                Q, px = Q / Q.vol_voxel(), Q.res()
                Q.units = Q.units.gather()
                return Q, px

            n_parts = n[sub._mask]
            n, px = bin_func(sub, n_parts / dV, **binargs)
            # we actually need the column densities, not the number of particles
            n *= px
            # for averaging, we want the integral over n_parts, not its density
            n_, px = bin_func(sub, n_parts, **binargs)
            non0n = n_ != 0
            vel, px = bin_func(sub, n_parts * sub["vel"][:, zaxis], **binargs)
            vel[non0n] = vel[non0n] / n_[non0n]
            rho, px = bin_func(sub, n_parts * sub["rho"], **binargs)
            if element is not None and element != "H" and element != "He":
                metal_frac, px = bin_func(
                    sub, n_parts * sub.gas[element] / sub.gas["mass"], **binargs
                )
            else:
                metal_frac, px = bin_func(
                    sub, n_parts * sub.gas["metallicity"], **binargs
                )

            # average sqrt(T), since thats what the therm. broadening scales with
            temp, px = bin_func(sub, n_parts * np.sqrt(sub["temp"]), **binargs)
            temp[non0n] = temp[non0n] / n_[non0n]
            temp **= 2

            # the z-coordinates for the Hubble flow
            los_pos = UnitArr(
                np.linspace(float(spatial_extent[0]), float(spatial_extent[1] - px), N),
                spatial_extent.units,
            )
        else:
            raise ValueError("Unkown method '%s'!" % method)

        n.convert_to(l_units ** -2, subs=s)
        pos = None  # no use of positions in the C function
        hsml = None  # no use of smoothing lengths in the C function
        # inplace conversion possible (later conversion does not add to runtime!)
        vel.convert_to(v_units, subs=s)
        temp.convert_to("K", subs=s)
    else:
        pos = s.gas["pos"][:, (xaxis, yaxis)]
        vel = s.gas["vel"][:, zaxis]
        temp = s.gas["temp"]
        rho = s.gas["rho"]  # DS: gas density
        if element is not None and element != "H" and element != "He":
            metal_frac = s.gas[element] / s.gas["mass"]  # SA: metal mass fraction
        else:
            metal_frac = s.gas["metallicity"]

        if temp.base is not None:
            temp.copy()

        if isinstance(hsml, str):
            hsml = s.gas[hsml]
        elif isinstance(hsml, (Number, Unit)):
            hsml = UnitScalar(hsml, s["pos"].units) * np.ones(
                len(s.gas), dtype=np.float64
            )
        else:
            hsml = UnitQty(hsml, s["pos"].units, subs=s)
        if hsml.base is not None:
            hsml.copy()

        N = len(s.gas)

        # the z-coordinates for the Hubble flow
        los_pos = s.gas["pos"][:, zaxis]

    # add the Hubble flow
    zero_Hubble_flow_at.convert_to(los_pos.units, subs=s)
    H_flow = s.cosmology.H(s.redshift) * (los_pos - zero_Hubble_flow_at)
    H_flow.convert_to(vel.units, subs=s)
    vpec_z = vel  # DS: peculiar LOS velocities
    vel = vel + H_flow

    if pos is not None:
        pos = (
            pos.astype(np.float64).in_units_of(l_units, subs=s).view(np.ndarray).copy()
        )
    vel = vel.astype(np.float64).in_units_of(v_units, subs=s).view(np.ndarray).copy()
    vpec_z = (
        vpec_z.astype(np.float64).in_units_of(v_units, subs=s).view(np.ndarray).copy()
    )  # DS LOS peculiar velocities
    temp = temp.in_units_of("K", subs=s).view(np.ndarray).astype(np.float64)
    rho = (
        rho.in_units_of("g/cm**3", subs=s).view(np.ndarray).astype(np.float64)
    )  # DS: gas density
    metal_frac = metal_frac.view(np.ndarray).astype(
        np.float64
    )  # SA metal mass fraction

    if hsml is not None:
        hsml = hsml.in_units_of(l_units, subs=s).view(np.ndarray).astype(np.float64)

    los = los.in_units_of(l_units, subs=s).view(np.ndarray).astype(np.float64).copy()
    vel_extent = (
        vel_extent.in_units_of(v_units, subs=s)
        .view(np.ndarray)
        .astype(np.float64)
        .copy()
    )
    if v_turb is not None:
        v_turb = (
            v_turb.in_units_of(v_units, subs=s)
            .view(np.ndarray)
            .astype(np.float64)
            .copy()
        )

    b_0 = float(b_0.in_units_of(v_units, subs=s))
    Xsec = float(Xsec.in_units_of(l_units ** 2 * v_units, subs=s))
    Gamma = float(Gamma.in_units_of(v_units))

    taus = np.empty(Nbins, dtype=np.float64)
    los_dens = np.empty(Nbins, dtype=np.float64)
    los_dens_phys = np.empty(Nbins, dtype=np.float64)  # DS: gas density field
    los_temp = np.empty(Nbins, dtype=np.float64)
    los_vpec = np.empty(Nbins, dtype=np.float64)  # DS: LOS peculiar velocity field
    los_metal_frac = np.empty(Nbins, dtype=np.float64)  # SA: LOS metallicity field
    restr_column_lims = restr_column_lims.view(np.ndarray).astype(np.float64)
    restr_column = np.empty(N, dtype=np.float64)
    C.cpygad.absorption_spectrum(
        method == "particles",
        C.c_size_t(N),
        C.c_void_p(pos.ctypes.data) if pos is not None else None,
        C.c_void_p(vel.ctypes.data),
        C.c_void_p(vpec_z.ctypes.data),  # DS Los peculiar velocities
        C.c_void_p(hsml.ctypes.data) if hsml is not None else None,
        C.c_void_p(n.ctypes.data),
        C.c_void_p(temp.ctypes.data),
        C.c_void_p(rho.ctypes.data),  # DS: gas density
        C.c_void_p(metal_frac.ctypes.data),  # SA: metal mass fraction
        C.c_void_p(los.ctypes.data),
        C.c_void_p(vel_extent.ctypes.data),
        C.c_size_t(Nbins),
        C.c_double(b_0),
        C.c_void_p(v_turb.ctypes.data) if v_turb is not None else None,
        C.c_double(Xsec),
        C.c_double(Gamma),
        C.c_void_p(taus.ctypes.data),
        C.c_void_p(los_dens.ctypes.data),
        C.c_void_p(los_dens_phys.ctypes.data),  # DS gas density
        C.c_void_p(los_metal_frac.ctypes.data),  # SA LOS metal mass fraction
        C.c_void_p(los_temp.ctypes.data),
        C.c_void_p(los_vpec.ctypes.data),  # DS LOS peculiar velocity field
        C.c_void_p(restr_column_lims.ctypes.data),
        C.c_void_p(restr_column.ctypes.data),
        C.create_string_buffer(kernel.encode("ascii")),
        C.c_double(s.boxsize.in_units_of(l_units)),
    )

    los_dens = UnitArr(los_dens, "cm**-2")
    los_dens_phys = UnitArr(los_dens_phys, "g cm**-3")  # DS gas density field
    los_temp = UnitArr(los_temp, "K")
    los_metal_frac = UnitArr(
        los_metal_frac
    )  # SA: LOS metal mass fraction, dimensionless
    los_vpec = UnitArr(los_vpec, "km/s")  # DS LOS peculiar velocity field
    restr_column = UnitArr(restr_column, "cm**-2")

    if environment.verbose >= environment.VERBOSE_NORMAL:
        # if called with bad parameters sum(taus)==0 and, hence, no normation
        # possible:
        try:
            # calculate parameters
            z_edges = velocities_to_redshifts(v_edges, z0=s.redshift)
            l_edges = l * (1.0 + z_edges)
            EW_l = EW(taus, l_edges)
            extinct = np.exp(-np.asarray(taus))
            v_mean = UnitArr(
                np.average((v_edges[:-1] + v_edges[1:]) / 2.0, weights=extinct),
                v_edges.units,
            )
            l_mean = UnitArr(
                np.average((l_edges[:-1] + l_edges[1:]) / 2.0, weights=extinct),
                l_edges.units,
            )
            print("created line with:")
            print("  EW =", EW_l)
            print("  v0 =", v_mean)
            print("  l0 =", l_mean)
        except:
            pass

    if return_los_phys:
        return (
            taus,
            los_dens,
            los_dens_phys,
            los_temp,
            los_metal_frac,
            los_vpec,
            v_edges,
            restr_column,
        )
    else:
        return taus, los_dens, los_temp, v_edges, restr_column


def EW(taus, edges):
    """
    Calculate the equivalent width of the given line / spectrum.

    Args:
        taus (array-like):  The optical depths in the bins.
        edges (UnitQty):    The edges of the bins. May and should have units,
                            otherwise it is assumed to be in units of Angstrom.
                            If it is a scalar, it is assumed to be the constant
                            width of the bins.

    Returns:
        EW (float, UnitScalar):     The equivalent width in the given space (i.e.
                                    the units of the edges).
    """
    edges = UnitQty(edges)
    if edges.units in [1, None]:
        edges = UnitArr(edges, "Angstrom")
    if edges.shape != tuple() and len(taus) + 1 != len(edges):
        raise ValueError(
            "The length of the edges does not match the length of "
            + "the optical depths!"
        )
    if edges.shape == tuple():
        EW = edges * np.sum((1.0 - np.exp(-np.asarray(taus))))
    else:
        EW = np.sum((1.0 - np.exp(-np.asarray(taus))) * (edges[1:] - edges[:-1]))
    return EW


def velocities_to_redshifts(vs, z0=0.0):
    """
    Convert velocities to redshifts, assuming an additional cosmological redshift.

    Note:
        The inverse is `redshifts_to_velocities`.

    Args:
        vs (UnitQty):       The velocities to convert to redshifts.
        z0 (float):         The cosmological redshift of the restframe in which
                            the velocities were measured.

    Returns:
        zs (np.ndarray):    The redshifts corresponding to the given velocities.
    """
    vs = UnitQty(vs, "km/s", dtype=float)
    zs = (vs / c).in_units_of(1).view(np.ndarray)
    if z0 != 0.0:  # avoid multiplication of 1.
        zs = (1.0 + zs) * (1.0 + z0) - 1.0
    return zs


def redshifts_to_velocities(zs, z0=0.0):
    """
    Convert redshifts to velocities, assuming an additional cosmological redshift.

    Note:
        This is the inverse to `velocities_to_redshifts`.

    Args:
        zs (array-like):    The redshifts to convert to velocities.
        z0 (float):         The cosmological redshift of the restframe in which
                            the velocities were measured.

    Returns:
        vs (UnitQty):       The velocities corresponding to the given redshifts.
    """
    zs = np.array(zs, dtype=float)
    if z0 != 0.0:  # avoid division by 1.
        zs = (zs + 1.0) / (1.0 + z0) - 1.0
    vs = c * zs
    return vs


def fit_continuum(l, flux, noise, order=0, sigma_lim=2.0, tol=1.0e-4, max_iter=100):
    """
    Fits a continuum to a spectrum by fitting a curve to "unabsorbed" pixels, removing
    all "absorbed" pixels more than sigma_lim*noise below the curve, and iterating
    until the median value of the continuum fractionally varies by less than tol.
    The curve is computed as the best-fit polynomial of a specified order.
    After iterating to convergence, the curve is returned as the continuum level.
    Note that the noise is assumed to be Gaussian.
    This broadly follows what is done for observational data (e.g. Danforth+16).
    Last modified: Romeel Dave (28 Nov 2019)

    Args:
        l (UnitQty):         The wavelength array.
        flux (numpy array):  The fluxes at the given wavelengths.
        noise (numpy array): The 1-sigma noise array (>0), same length as l or flux.
        order (int):         Order of polynomial to fit to the unabsorbed portions.
        sigma_lim (float):   Pixels less than sigma_lim*noise away from polynomial are
                             considered unabsorbed.
        tol (float):         The continuum is converged if its median value changes by
                             less than this fraction relative to the previous iteration.
        max_iter (int):      The maximum number of allowed iterations.

    Returns:
        contin (array-like): Array of continuum values, of len(flux).
    """

    n_iter = 0
    med_old = 0
    diff = 2 * tol  # initialize to something larger than tol
    l_unabs = np.copy(l)  # begin with all pixels as "unabsorbed"
    f_unabs = np.copy(flux)
    n_unabs = np.copy(noise)
    while diff > tol:  # iterate to convergence
        p = np.polyfit(
            l_unabs, f_unabs, order, w=1.0 / n_unabs
        )  # fit polynomial of specified order
        contin = np.polyval(p, l_unabs)  # evaluate polynomial to get continuum guess
        med_contin = np.median(contin)  # get median value of continuum guess
        select = (
            f_unabs > contin - sigma_lim * n_unabs
        )  # select unabsorbed pixels for next iteration
        f_unabs = f_unabs[select]
        l_unabs = l_unabs[select]
        n_unabs = n_unabs[select]
        diff = abs((med_contin - med_old) / med_old)  # criteria for convergence
        med_old = med_contin
        if n_iter > max_iter:
            print(
                "warning: continuum fitting failed after %d iterations: diff=%g > tol=%g"
                % (n_iter, diff, tol)
            )
            break
        if len(l_unabs) < 0.1 * len(l):
            print(
                "warning: continuum fitting failed, too few pixels left: only %d of %d"
                % (len(l_unabs), len(l))
            )
            break
        n_iter += 1
    contin = np.polyval(p, l)
    print(
        "Continuum fit done: Median (using %d/%d pixels, %d iterations) is %g"
        % (len(l_unabs), len(l), n_iter, med_contin)
    )
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print("contin=", contin)
    return contin  # when converged, evaluate final polynomial to get full continuum


def apply_LSF(l, flux, noise, grating="COS_G130M"):
    """
    Smooth the spectrum with an instrumental line spread function.  The LSF is
    wavelength-dependent, so it uses the LSF at the closest wavelength to the spectrum.
    Last modified: Romeel Dave (28 Nov 2019)

    Args:
        l (UnitQty):         The wavelength array.
        flux (numpy array):  The fluxes at the given wavelengths.
        noise (numpy array): The 1-sigma noise array (>0), same length as l or flux.
        grating (string):    Name of grating; choices are those in LSF_data[]

    Returns:
        flux (numpy array):  LSF-convolved flux
        noise (numpy array): LSF-convolved noise
    """

    from astropy.convolution import convolve

    lsf_data = LSF_data[grating]  # load grating LSF data
    # print(np.mean(l),abs(lsf_data['channels']-np.mean(l)),abs(lsf_data['channels']-np.mean(l)).argmin())
    channel = "w" + str(
        lsf_data["channels"][abs(lsf_data["channels"] - np.mean(l)).argmin()]
    )  # COS naming convention
    rw = lsf_data["relwave"]  # relative wavelengths
    lsf = lsf_data[channel]  # line spread function
    # interpolate the LSF onto the wavelength scale of the input spectrum (approximately)
    dl_lsf = rw[1] - rw[0]
    nlsf_interp = 2 * (
        int(len(rw) * dl_lsf / (l[1] - l[0])) // 2
    )  # make sure this is an even number
    if nlsf_interp < 2:
        return flux, noise  # if pixels are too large, then LSF has no effect
    rw_interp = np.arange(rw[0], rw[-1] + 0.5 * dl_lsf, (rw[-1] - rw[0]) / nlsf_interp)
    lsf_interp = np.interp(rw_interp, rw, lsf)
    # convolve flux and noise
    flux_conv = convolve(flux, lsf_interp, boundary="wrap")
    noise_conv = convolve(noise, lsf_interp, boundary="wrap")
    print(
        "Applied LSF at <lambda>=%1g: %s, channel %s" % (np.mean(l), grating, channel)
    )  # ,np.mean(flux_conv)-np.mean(flux)))
    return flux_conv, noise_conv


LSF_data = {
    "COS_G130M": {  # HST/COS medium-res FUV grating
        "channels": np.array([1150, 1200, 1250, 1300, 1350, 1400, 1450]),
        "relwave": 9.97e-3
        * np.arange(-50, 51, 1, dtype=int),  # 9.97 mA/pixel for G130M
        "w1150": np.array(
            [
                0.0001011,
                0.0001059,
                0.0001115,
                0.0001179,
                0.0001256,
                0.0001348,
                0.000146,
                0.0001593,
                0.0001745,
                0.000192,
                0.0002122,
                0.0002354,
                0.0002608,
                0.0002879,
                0.0003169,
                0.0003502,
                0.0003903,
                0.0004392,
                0.000497,
                0.0005637,
                0.000639,
                0.0007243,
                0.0008228,
                0.000939,
                0.001079,
                0.00125,
                0.001459,
                0.001707,
                0.001993,
                0.00232,
                0.002685,
                0.00307,
                0.003454,
                0.00384,
                0.004271,
                0.004793,
                0.005426,
                0.006172,
                0.007063,
                0.008165,
                0.009567,
                0.01141,
                0.01392,
                0.01753,
                0.02298,
                0.03133,
                0.04364,
                0.05991,
                0.07782,
                0.09258,
                0.09899,
                0.09454,
                0.08097,
                0.06306,
                0.04581,
                0.03229,
                0.0231,
                0.01729,
                0.0136,
                0.01112,
                0.009343,
                0.007964,
                0.006796,
                0.005768,
                0.004887,
                0.004175,
                0.00362,
                0.003176,
                0.002799,
                0.002464,
                0.002169,
                0.001919,
                0.001714,
                0.001551,
                0.00142,
                0.00131,
                0.001205,
                0.001094,
                0.0009745,
                0.0008524,
                0.0007401,
                0.000646,
                0.0005713,
                0.0005117,
                0.0004626,
                0.0004205,
                0.0003829,
                0.0003481,
                0.0003158,
                0.0002865,
                0.0002609,
                0.0002383,
                0.0002174,
                0.0001973,
                0.0001786,
                0.0001618,
                0.0001472,
                0.0001344,
                0.0001233,
                0.0001143,
                0.0001074,
            ]
        ),
        "w1200": np.array(
            [
                9.485e-05,
                1.003e-04,
                1.065e-04,
                1.136e-04,
                1.222e-04,
                1.324e-04,
                1.441e-04,
                1.573e-04,
                1.724e-04,
                1.902e-04,
                2.106e-04,
                2.328e-04,
                2.556e-04,
                2.793e-04,
                3.066e-04,
                3.409e-04,
                3.836e-04,
                4.341e-04,
                4.915e-04,
                5.569e-04,
                6.329e-04,
                7.210e-04,
                8.229e-04,
                9.431e-04,
                1.091e-03,
                1.277e-03,
                1.501e-03,
                1.757e-03,
                2.041e-03,
                2.357e-03,
                2.708e-03,
                3.075e-03,
                3.436e-03,
                3.797e-03,
                4.200e-03,
                4.695e-03,
                5.310e-03,
                6.047e-03,
                6.902e-03,
                7.911e-03,
                9.183e-03,
                1.092e-02,
                1.341e-02,
                1.707e-02,
                2.258e-02,
                3.103e-02,
                4.357e-02,
                6.032e-02,
                7.892e-02,
                9.441e-02,
                1.013e-01,
                9.664e-02,
                8.230e-02,
                6.333e-02,
                4.522e-02,
                3.126e-02,
                2.200e-02,
                1.631e-02,
                1.276e-02,
                1.042e-02,
                8.797e-03,
                7.605e-03,
                6.622e-03,
                5.720e-03,
                4.880e-03,
                4.154e-03,
                3.579e-03,
                3.144e-03,
                2.803e-03,
                2.510e-03,
                2.239e-03,
                1.986e-03,
                1.755e-03,
                1.556e-03,
                1.395e-03,
                1.269e-03,
                1.168e-03,
                1.080e-03,
                9.928e-04,
                9.000e-04,
                8.004e-04,
                7.009e-04,
                6.111e-04,
                5.370e-04,
                4.783e-04,
                4.311e-04,
                3.915e-04,
                3.570e-04,
                3.262e-04,
                2.980e-04,
                2.716e-04,
                2.472e-04,
                2.251e-04,
                2.055e-04,
                1.879e-04,
                1.715e-04,
                1.561e-04,
                1.421e-04,
                1.298e-04,
                1.190e-04,
                1.096e-04,
            ]
        ),
        "w1250": np.array(
            [
                9.123e-05,
                9.703e-05,
                1.037e-04,
                1.117e-04,
                1.211e-04,
                1.319e-04,
                1.441e-04,
                1.582e-04,
                1.745e-04,
                1.927e-04,
                2.117e-04,
                2.309e-04,
                2.513e-04,
                2.758e-04,
                3.069e-04,
                3.448e-04,
                3.883e-04,
                4.376e-04,
                4.947e-04,
                5.623e-04,
                6.411e-04,
                7.321e-04,
                8.401e-04,
                9.735e-04,
                1.140e-03,
                1.340e-03,
                1.568e-03,
                1.817e-03,
                2.092e-03,
                2.397e-03,
                2.728e-03,
                3.064e-03,
                3.394e-03,
                3.732e-03,
                4.127e-03,
                4.627e-03,
                5.243e-03,
                5.948e-03,
                6.730e-03,
                7.649e-03,
                8.862e-03,
                1.061e-02,
                1.317e-02,
                1.694e-02,
                2.258e-02,
                3.121e-02,
                4.399e-02,
                6.105e-02,
                8.002e-02,
                9.580e-02,
                1.026e-01,
                9.760e-02,
                8.260e-02,
                6.309e-02,
                4.469e-02,
                3.062e-02,
                2.132e-02,
                1.564e-02,
                1.214e-02,
                9.866e-03,
                8.319e-03,
                7.224e-03,
                6.367e-03,
                5.594e-03,
                4.846e-03,
                4.151e-03,
                3.562e-03,
                3.108e-03,
                2.769e-03,
                2.501e-03,
                2.262e-03,
                2.031e-03,
                1.806e-03,
                1.594e-03,
                1.408e-03,
                1.256e-03,
                1.139e-03,
                1.046e-03,
                9.670e-04,
                8.936e-04,
                8.188e-04,
                7.380e-04,
                6.529e-04,
                5.710e-04,
                5.000e-04,
                4.427e-04,
                3.974e-04,
                3.602e-04,
                3.282e-04,
                2.998e-04,
                2.744e-04,
                2.512e-04,
                2.296e-04,
                2.093e-04,
                1.909e-04,
                1.748e-04,
                1.604e-04,
                1.470e-04,
                1.344e-04,
                1.227e-04,
                1.124e-04,
            ]
        ),
        "w1300": np.array(
            [
                8.844e-05,
                9.490e-05,
                1.026e-04,
                1.116e-04,
                1.217e-04,
                1.330e-04,
                1.461e-04,
                1.609e-04,
                1.768e-04,
                1.928e-04,
                2.092e-04,
                2.277e-04,
                2.508e-04,
                2.798e-04,
                3.140e-04,
                3.525e-04,
                3.962e-04,
                4.473e-04,
                5.077e-04,
                5.781e-04,
                6.599e-04,
                7.578e-04,
                8.788e-04,
                1.029e-03,
                1.207e-03,
                1.410e-03,
                1.632e-03,
                1.873e-03,
                2.138e-03,
                2.427e-03,
                2.728e-03,
                3.024e-03,
                3.317e-03,
                3.643e-03,
                4.051e-03,
                4.567e-03,
                5.166e-03,
                5.809e-03,
                6.507e-03,
                7.362e-03,
                8.560e-03,
                1.033e-02,
                1.293e-02,
                1.671e-02,
                2.237e-02,
                3.108e-02,
                4.412e-02,
                6.160e-02,
                8.110e-02,
                9.733e-02,
                1.042e-01,
                9.880e-02,
                8.310e-02,
                6.297e-02,
                4.421e-02,
                3.000e-02,
                2.068e-02,
                1.504e-02,
                1.161e-02,
                9.404e-03,
                7.911e-03,
                6.876e-03,
                6.104e-03,
                5.432e-03,
                4.777e-03,
                4.136e-03,
                3.558e-03,
                3.089e-03,
                2.735e-03,
                2.470e-03,
                2.250e-03,
                2.042e-03,
                1.835e-03,
                1.630e-03,
                1.438e-03,
                1.269e-03,
                1.131e-03,
                1.025e-03,
                9.410e-04,
                8.700e-04,
                8.055e-04,
                7.419e-04,
                6.739e-04,
                6.006e-04,
                5.271e-04,
                4.610e-04,
                4.068e-04,
                3.639e-04,
                3.293e-04,
                2.998e-04,
                2.737e-04,
                2.504e-04,
                2.296e-04,
                2.105e-04,
                1.927e-04,
                1.761e-04,
                1.611e-04,
                1.480e-04,
                1.364e-04,
                1.254e-04,
                1.150e-04,
            ]
        ),
        "w1350": np.array(
            [
                8.769e-05,
                9.526e-05,
                1.035e-04,
                1.126e-04,
                1.230e-04,
                1.352e-04,
                1.485e-04,
                1.620e-04,
                1.753e-04,
                1.894e-04,
                2.067e-04,
                2.289e-04,
                2.562e-04,
                2.875e-04,
                3.223e-04,
                3.621e-04,
                4.090e-04,
                4.641e-04,
                5.275e-04,
                6.011e-04,
                6.900e-04,
                8.004e-04,
                9.365e-04,
                1.097e-03,
                1.278e-03,
                1.474e-03,
                1.687e-03,
                1.924e-03,
                2.181e-03,
                2.442e-03,
                2.693e-03,
                2.936e-03,
                3.204e-03,
                3.542e-03,
                3.979e-03,
                4.502e-03,
                5.064e-03,
                5.639e-03,
                6.272e-03,
                7.088e-03,
                8.272e-03,
                1.003e-02,
                1.256e-02,
                1.622e-02,
                2.176e-02,
                3.047e-02,
                4.379e-02,
                6.193e-02,
                8.233e-02,
                9.940e-02,
                1.067e-01,
                1.009e-01,
                8.430e-02,
                6.307e-02,
                4.350e-02,
                2.898e-02,
                1.971e-02,
                1.427e-02,
                1.105e-02,
                8.971e-03,
                7.549e-03,
                6.562e-03,
                5.852e-03,
                5.262e-03,
                4.686e-03,
                4.100e-03,
                3.544e-03,
                3.071e-03,
                2.706e-03,
                2.433e-03,
                2.215e-03,
                2.018e-03,
                1.824e-03,
                1.633e-03,
                1.450e-03,
                1.283e-03,
                1.137e-03,
                1.018e-03,
                9.253e-04,
                8.502e-04,
                7.857e-04,
                7.269e-04,
                6.692e-04,
                6.077e-04,
                5.414e-04,
                4.749e-04,
                4.154e-04,
                3.666e-04,
                3.281e-04,
                2.970e-04,
                2.704e-04,
                2.467e-04,
                2.255e-04,
                2.067e-04,
                1.898e-04,
                1.743e-04,
                1.598e-04,
                1.464e-04,
                1.344e-04,
                1.240e-04,
                1.145e-04,
            ]
        ),
        "w1400": np.array(
            [
                8.711e-05,
                9.479e-05,
                1.037e-04,
                1.141e-04,
                1.257e-04,
                1.377e-04,
                1.492e-04,
                1.603e-04,
                1.731e-04,
                1.895e-04,
                2.107e-04,
                2.359e-04,
                2.640e-04,
                2.954e-04,
                3.321e-04,
                3.753e-04,
                4.250e-04,
                4.810e-04,
                5.457e-04,
                6.252e-04,
                7.262e-04,
                8.519e-04,
                9.983e-04,
                1.158e-03,
                1.328e-03,
                1.516e-03,
                1.730e-03,
                1.964e-03,
                2.195e-03,
                2.402e-03,
                2.596e-03,
                2.813e-03,
                3.096e-03,
                3.467e-03,
                3.921e-03,
                4.426e-03,
                4.948e-03,
                5.484e-03,
                6.093e-03,
                6.897e-03,
                8.062e-03,
                9.773e-03,
                1.224e-02,
                1.583e-02,
                2.132e-02,
                3.012e-02,
                4.372e-02,
                6.230e-02,
                8.325e-02,
                1.009e-01,
                1.085e-01,
                1.028e-01,
                8.558e-02,
                6.346e-02,
                4.309e-02,
                2.813e-02,
                1.882e-02,
                1.354e-02,
                1.051e-02,
                8.591e-03,
                7.246e-03,
                6.287e-03,
                5.603e-03,
                5.069e-03,
                4.571e-03,
                4.053e-03,
                3.535e-03,
                3.068e-03,
                2.692e-03,
                2.406e-03,
                2.180e-03,
                1.979e-03,
                1.786e-03,
                1.600e-03,
                1.427e-03,
                1.272e-03,
                1.135e-03,
                1.018e-03,
                9.197e-04,
                8.390e-04,
                7.711e-04,
                7.115e-04,
                6.564e-04,
                6.015e-04,
                5.428e-04,
                4.804e-04,
                4.195e-04,
                3.660e-04,
                3.230e-04,
                2.895e-04,
                2.626e-04,
                2.396e-04,
                2.190e-04,
                2.005e-04,
                1.839e-04,
                1.689e-04,
                1.549e-04,
                1.420e-04,
                1.302e-04,
                1.199e-04,
                1.110e-04,
            ]
        ),
        "w1450": np.array(
            [
                8.764e-05,
                9.696e-05,
                1.076e-04,
                1.185e-04,
                1.288e-04,
                1.386e-04,
                1.487e-04,
                1.607e-04,
                1.761e-04,
                1.948e-04,
                2.165e-04,
                2.414e-04,
                2.704e-04,
                3.043e-04,
                3.430e-04,
                3.865e-04,
                4.353e-04,
                4.927e-04,
                5.647e-04,
                6.576e-04,
                7.721e-04,
                9.016e-04,
                1.038e-03,
                1.184e-03,
                1.350e-03,
                1.547e-03,
                1.762e-03,
                1.963e-03,
                2.128e-03,
                2.276e-03,
                2.454e-03,
                2.702e-03,
                3.021e-03,
                3.396e-03,
                3.815e-03,
                4.278e-03,
                4.785e-03,
                5.342e-03,
                5.989e-03,
                6.813e-03,
                7.951e-03,
                9.591e-03,
                1.201e-02,
                1.565e-02,
                2.137e-02,
                3.044e-02,
                4.409e-02,
                6.238e-02,
                8.284e-02,
                1.001e-01,
                1.081e-01,
                1.030e-01,
                8.654e-02,
                6.479e-02,
                4.427e-02,
                2.883e-02,
                1.902e-02,
                1.342e-02,
                1.030e-02,
                8.413e-03,
                7.125e-03,
                6.172e-03,
                5.451e-03,
                4.891e-03,
                4.414e-03,
                3.953e-03,
                3.489e-03,
                3.052e-03,
                2.681e-03,
                2.386e-03,
                2.148e-03,
                1.934e-03,
                1.729e-03,
                1.535e-03,
                1.363e-03,
                1.216e-03,
                1.094e-03,
                9.913e-04,
                9.032e-04,
                8.244e-04,
                7.529e-04,
                6.890e-04,
                6.317e-04,
                5.778e-04,
                5.228e-04,
                4.647e-04,
                4.058e-04,
                3.517e-04,
                3.066e-04,
                2.712e-04,
                2.435e-04,
                2.210e-04,
                2.016e-04,
                1.843e-04,
                1.689e-04,
                1.554e-04,
                1.432e-04,
                1.317e-04,
                1.209e-04,
                1.113e-04,
                1.030e-04,
            ]
        ),
    },
    "COS_G140L": {  # COS low-res grating
        "channels": np.array(
            [1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800]
        ),
        "relwave": 80.3e-3
        * np.arange(-50, 51, 1, dtype=int),  # 80.3 mA/pixel for G140L
        "w1250": np.array(
            [
                8.859e-05,
                9.503e-05,
                1.030e-04,
                1.127e-04,
                1.241e-04,
                1.369e-04,
                1.511e-04,
                1.676e-04,
                1.874e-04,
                2.105e-04,
                2.361e-04,
                2.634e-04,
                2.933e-04,
                3.276e-04,
                3.672e-04,
                4.124e-04,
                4.638e-04,
                5.237e-04,
                5.952e-04,
                6.809e-04,
                7.822e-04,
                9.010e-04,
                1.042e-03,
                1.214e-03,
                1.421e-03,
                1.653e-03,
                1.891e-03,
                2.126e-03,
                2.370e-03,
                2.639e-03,
                2.932e-03,
                3.238e-03,
                3.561e-03,
                3.929e-03,
                4.364e-03,
                4.858e-03,
                5.386e-03,
                5.959e-03,
                6.650e-03,
                7.587e-03,
                8.901e-03,
                1.071e-02,
                1.321e-02,
                1.692e-02,
                2.290e-02,
                3.278e-02,
                4.804e-02,
                6.843e-02,
                9.033e-02,
                1.068e-01,
                1.109e-01,
                1.003e-01,
                7.921e-02,
                5.578e-02,
                3.647e-02,
                2.360e-02,
                1.615e-02,
                1.209e-02,
                9.788e-03,
                8.334e-03,
                7.318e-03,
                6.502e-03,
                5.709e-03,
                4.871e-03,
                4.056e-03,
                3.379e-03,
                2.902e-03,
                2.592e-03,
                2.369e-03,
                2.159e-03,
                1.932e-03,
                1.696e-03,
                1.474e-03,
                1.282e-03,
                1.121e-03,
                9.887e-04,
                8.783e-04,
                7.853e-04,
                7.054e-04,
                6.330e-04,
                5.632e-04,
                4.948e-04,
                4.316e-04,
                3.780e-04,
                3.352e-04,
                3.013e-04,
                2.738e-04,
                2.509e-04,
                2.313e-04,
                2.139e-04,
                1.979e-04,
                1.829e-04,
                1.689e-04,
                1.558e-04,
                1.436e-04,
                1.321e-04,
                1.210e-04,
                1.103e-04,
                1.005e-04,
                9.213e-05,
                8.526e-05,
            ]
        ),
        "w1300": np.array(
            [
                8.653e-05,
                9.431e-05,
                1.036e-04,
                1.142e-04,
                1.257e-04,
                1.388e-04,
                1.542e-04,
                1.725e-04,
                1.933e-04,
                2.155e-04,
                2.392e-04,
                2.657e-04,
                2.964e-04,
                3.315e-04,
                3.710e-04,
                4.164e-04,
                4.703e-04,
                5.357e-04,
                6.142e-04,
                7.057e-04,
                8.112e-04,
                9.357e-04,
                1.088e-03,
                1.274e-03,
                1.484e-03,
                1.699e-03,
                1.907e-03,
                2.117e-03,
                2.352e-03,
                2.621e-03,
                2.908e-03,
                3.198e-03,
                3.508e-03,
                3.865e-03,
                4.278e-03,
                4.732e-03,
                5.214e-03,
                5.754e-03,
                6.445e-03,
                7.402e-03,
                8.724e-03,
                1.049e-02,
                1.291e-02,
                1.655e-02,
                2.258e-02,
                3.272e-02,
                4.848e-02,
                6.952e-02,
                9.192e-02,
                1.085e-01,
                1.120e-01,
                1.006e-01,
                7.883e-02,
                5.507e-02,
                3.574e-02,
                2.297e-02,
                1.563e-02,
                1.167e-02,
                9.429e-03,
                8.013e-03,
                7.032e-03,
                6.284e-03,
                5.598e-03,
                4.871e-03,
                4.116e-03,
                3.433e-03,
                2.908e-03,
                2.558e-03,
                2.333e-03,
                2.159e-03,
                1.977e-03,
                1.769e-03,
                1.552e-03,
                1.349e-03,
                1.176e-03,
                1.031e-03,
                9.106e-04,
                8.085e-04,
                7.220e-04,
                6.489e-04,
                5.851e-04,
                5.252e-04,
                4.660e-04,
                4.092e-04,
                3.593e-04,
                3.186e-04,
                2.862e-04,
                2.598e-04,
                2.375e-04,
                2.183e-04,
                2.014e-04,
                1.858e-04,
                1.713e-04,
                1.578e-04,
                1.455e-04,
                1.343e-04,
                1.238e-04,
                1.141e-04,
                1.048e-04,
                9.587e-05,
                8.765e-05,
            ]
        ),
        "w1350": np.array(
            [
                8.788e-05,
                9.670e-05,
                1.062e-04,
                1.165e-04,
                1.284e-04,
                1.427e-04,
                1.595e-04,
                1.778e-04,
                1.971e-04,
                2.180e-04,
                2.417e-04,
                2.691e-04,
                3.001e-04,
                3.349e-04,
                3.756e-04,
                4.249e-04,
                4.854e-04,
                5.575e-04,
                6.403e-04,
                7.340e-04,
                8.437e-04,
                9.782e-04,
                1.144e-03,
                1.333e-03,
                1.528e-03,
                1.713e-03,
                1.894e-03,
                2.097e-03,
                2.340e-03,
                2.611e-03,
                2.885e-03,
                3.157e-03,
                3.448e-03,
                3.784e-03,
                4.166e-03,
                4.579e-03,
                5.026e-03,
                5.558e-03,
                6.267e-03,
                7.247e-03,
                8.555e-03,
                1.025e-02,
                1.257e-02,
                1.614e-02,
                2.221e-02,
                3.257e-02,
                4.876e-02,
                7.038e-02,
                9.334e-02,
                1.101e-01,
                1.133e-01,
                1.010e-01,
                7.861e-02,
                5.450e-02,
                3.512e-02,
                2.241e-02,
                1.516e-02,
                1.127e-02,
                9.086e-03,
                7.711e-03,
                6.759e-03,
                6.057e-03,
                5.456e-03,
                4.836e-03,
                4.166e-03,
                3.512e-03,
                2.961e-03,
                2.560e-03,
                2.301e-03,
                2.134e-03,
                1.992e-03,
                1.828e-03,
                1.633e-03,
                1.430e-03,
                1.243e-03,
                1.085e-03,
                9.548e-04,
                8.448e-04,
                7.508e-04,
                6.702e-04,
                6.022e-04,
                5.443e-04,
                4.919e-04,
                4.403e-04,
                3.893e-04,
                3.427e-04,
                3.039e-04,
                2.730e-04,
                2.480e-04,
                2.269e-04,
                2.087e-04,
                1.928e-04,
                1.782e-04,
                1.645e-04,
                1.515e-04,
                1.395e-04,
                1.283e-04,
                1.181e-04,
                1.088e-04,
                1.002e-04,
                9.223e-05,
            ]
        ),
        "w1400": np.array(
            [
                9.029e-05,
                9.877e-05,
                1.082e-04,
                1.194e-04,
                1.326e-04,
                1.476e-04,
                1.637e-04,
                1.808e-04,
                1.996e-04,
                2.213e-04,
                2.460e-04,
                2.734e-04,
                3.042e-04,
                3.409e-04,
                3.864e-04,
                4.427e-04,
                5.093e-04,
                5.842e-04,
                6.675e-04,
                7.641e-04,
                8.828e-04,
                1.030e-03,
                1.200e-03,
                1.376e-03,
                1.542e-03,
                1.699e-03,
                1.873e-03,
                2.088e-03,
                2.341e-03,
                2.605e-03,
                2.857e-03,
                3.104e-03,
                3.372e-03,
                3.682e-03,
                4.031e-03,
                4.412e-03,
                4.845e-03,
                5.391e-03,
                6.130e-03,
                7.122e-03,
                8.393e-03,
                1.001e-02,
                1.223e-02,
                1.575e-02,
                2.189e-02,
                3.246e-02,
                4.900e-02,
                7.111e-02,
                9.456e-02,
                1.115e-01,
                1.144e-01,
                1.016e-01,
                7.850e-02,
                5.405e-02,
                3.460e-02,
                2.194e-02,
                1.475e-02,
                1.090e-02,
                8.765e-03,
                7.425e-03,
                6.498e-03,
                5.827e-03,
                5.285e-03,
                4.754e-03,
                4.175e-03,
                3.576e-03,
                3.028e-03,
                2.590e-03,
                2.283e-03,
                2.088e-03,
                1.960e-03,
                1.840e-03,
                1.689e-03,
                1.506e-03,
                1.315e-03,
                1.144e-03,
                1.001e-03,
                8.826e-04,
                7.829e-04,
                6.975e-04,
                6.240e-04,
                5.616e-04,
                5.089e-04,
                4.622e-04,
                4.168e-04,
                3.710e-04,
                3.272e-04,
                2.895e-04,
                2.593e-04,
                2.353e-04,
                2.156e-04,
                1.987e-04,
                1.840e-04,
                1.709e-04,
                1.585e-04,
                1.466e-04,
                1.350e-04,
                1.241e-04,
                1.140e-04,
                1.048e-04,
                9.646e-05,
            ]
        ),
        "w1450": np.array(
            [
                9.225e-05,
                1.012e-04,
                1.120e-04,
                1.244e-04,
                1.378e-04,
                1.521e-04,
                1.673e-04,
                1.844e-04,
                2.040e-04,
                2.261e-04,
                2.503e-04,
                2.782e-04,
                3.123e-04,
                3.553e-04,
                4.080e-04,
                4.693e-04,
                5.371e-04,
                6.116e-04,
                6.977e-04,
                8.036e-04,
                9.349e-04,
                1.088e-03,
                1.247e-03,
                1.397e-03,
                1.535e-03,
                1.683e-03,
                1.870e-03,
                2.102e-03,
                2.354e-03,
                2.595e-03,
                2.816e-03,
                3.038e-03,
                3.287e-03,
                3.576e-03,
                3.899e-03,
                4.263e-03,
                4.701e-03,
                5.274e-03,
                6.040e-03,
                7.029e-03,
                8.253e-03,
                9.796e-03,
                1.198e-02,
                1.556e-02,
                2.187e-02,
                3.272e-02,
                4.959e-02,
                7.204e-02,
                9.573e-02,
                1.127e-01,
                1.152e-01,
                1.017e-01,
                7.817e-02,
                5.350e-02,
                3.405e-02,
                2.147e-02,
                1.435e-02,
                1.055e-02,
                8.450e-03,
                7.145e-03,
                6.242e-03,
                5.591e-03,
                5.090e-03,
                4.627e-03,
                4.129e-03,
                3.595e-03,
                3.078e-03,
                2.631e-03,
                2.285e-03,
                2.048e-03,
                1.901e-03,
                1.800e-03,
                1.693e-03,
                1.550e-03,
                1.376e-03,
                1.199e-03,
                1.044e-03,
                9.165e-04,
                8.109e-04,
                7.215e-04,
                6.448e-04,
                5.785e-04,
                5.219e-04,
                4.744e-04,
                4.332e-04,
                3.938e-04,
                3.529e-04,
                3.121e-04,
                2.755e-04,
                2.458e-04,
                2.223e-04,
                2.033e-04,
                1.874e-04,
                1.738e-04,
                1.618e-04,
                1.509e-04,
                1.404e-04,
                1.298e-04,
                1.196e-04,
                1.100e-04,
                1.012e-04,
            ]
        ),
        "w1500": np.array(
            [
                9.472e-05,
                1.051e-04,
                1.164e-04,
                1.284e-04,
                1.409e-04,
                1.545e-04,
                1.701e-04,
                1.879e-04,
                2.075e-04,
                2.293e-04,
                2.550e-04,
                2.870e-04,
                3.277e-04,
                3.772e-04,
                4.336e-04,
                4.949e-04,
                5.618e-04,
                6.392e-04,
                7.344e-04,
                8.523e-04,
                9.897e-04,
                1.134e-03,
                1.269e-03,
                1.392e-03,
                1.519e-03,
                1.680e-03,
                1.887e-03,
                2.125e-03,
                2.359e-03,
                2.568e-03,
                2.760e-03,
                2.961e-03,
                3.194e-03,
                3.464e-03,
                3.772e-03,
                4.134e-03,
                4.591e-03,
                5.191e-03,
                5.970e-03,
                6.937e-03,
                8.105e-03,
                9.590e-03,
                1.177e-02,
                1.545e-02,
                2.197e-02,
                3.307e-02,
                5.019e-02,
                7.286e-02,
                9.671e-02,
                1.136e-01,
                1.159e-01,
                1.019e-01,
                7.792e-02,
                5.303e-02,
                3.357e-02,
                2.106e-02,
                1.399e-02,
                1.022e-02,
                8.157e-03,
                6.881e-03,
                5.999e-03,
                5.363e-03,
                4.888e-03,
                4.474e-03,
                4.043e-03,
                3.576e-03,
                3.103e-03,
                2.670e-03,
                2.308e-03,
                2.034e-03,
                1.850e-03,
                1.737e-03,
                1.654e-03,
                1.556e-03,
                1.418e-03,
                1.253e-03,
                1.091e-03,
                9.509e-04,
                8.366e-04,
                7.421e-04,
                6.621e-04,
                5.934e-04,
                5.341e-04,
                4.830e-04,
                4.402e-04,
                4.041e-04,
                3.701e-04,
                3.347e-04,
                2.979e-04,
                2.632e-04,
                2.338e-04,
                2.107e-04,
                1.921e-04,
                1.766e-04,
                1.635e-04,
                1.521e-04,
                1.420e-04,
                1.326e-04,
                1.232e-04,
                1.139e-04,
                1.050e-04,
            ]
        ),
        "w1550": np.array(
            [
                9.730e-05,
                1.073e-04,
                1.179e-04,
                1.294e-04,
                1.422e-04,
                1.569e-04,
                1.732e-04,
                1.909e-04,
                2.109e-04,
                2.346e-04,
                2.648e-04,
                3.031e-04,
                3.490e-04,
                4.006e-04,
                4.560e-04,
                5.166e-04,
                5.871e-04,
                6.737e-04,
                7.802e-04,
                9.037e-04,
                1.034e-03,
                1.156e-03,
                1.266e-03,
                1.376e-03,
                1.513e-03,
                1.695e-03,
                1.916e-03,
                2.143e-03,
                2.347e-03,
                2.522e-03,
                2.689e-03,
                2.874e-03,
                3.093e-03,
                3.350e-03,
                3.655e-03,
                4.031e-03,
                4.514e-03,
                5.136e-03,
                5.911e-03,
                6.839e-03,
                7.950e-03,
                9.398e-03,
                1.161e-02,
                1.544e-02,
                2.219e-02,
                3.352e-02,
                5.080e-02,
                7.358e-02,
                9.749e-02,
                1.144e-01,
                1.164e-01,
                1.021e-01,
                7.773e-02,
                5.265e-02,
                3.317e-02,
                2.071e-02,
                1.368e-02,
                9.938e-03,
                7.889e-03,
                6.636e-03,
                5.772e-03,
                5.147e-03,
                4.688e-03,
                4.308e-03,
                3.929e-03,
                3.520e-03,
                3.097e-03,
                2.693e-03,
                2.336e-03,
                2.043e-03,
                1.825e-03,
                1.682e-03,
                1.595e-03,
                1.524e-03,
                1.430e-03,
                1.297e-03,
                1.142e-03,
                9.922e-04,
                8.660e-04,
                7.631e-04,
                6.780e-04,
                6.061e-04,
                5.448e-04,
                4.919e-04,
                4.460e-04,
                4.074e-04,
                3.751e-04,
                3.458e-04,
                3.157e-04,
                2.833e-04,
                2.512e-04,
                2.229e-04,
                1.999e-04,
                1.817e-04,
                1.665e-04,
                1.535e-04,
                1.425e-04,
                1.329e-04,
                1.244e-04,
                1.162e-04,
                1.080e-04,
            ]
        ),
        "w1600": np.array(
            [
                9.890e-05,
                1.082e-04,
                1.185e-04,
                1.303e-04,
                1.438e-04,
                1.585e-04,
                1.747e-04,
                1.929e-04,
                2.152e-04,
                2.435e-04,
                2.793e-04,
                3.216e-04,
                3.684e-04,
                4.185e-04,
                4.733e-04,
                5.374e-04,
                6.158e-04,
                7.115e-04,
                8.223e-04,
                9.396e-04,
                1.051e-03,
                1.152e-03,
                1.249e-03,
                1.364e-03,
                1.521e-03,
                1.720e-03,
                1.938e-03,
                2.139e-03,
                2.309e-03,
                2.455e-03,
                2.602e-03,
                2.775e-03,
                2.984e-03,
                3.237e-03,
                3.551e-03,
                3.948e-03,
                4.454e-03,
                5.081e-03,
                5.834e-03,
                6.715e-03,
                7.772e-03,
                9.197e-03,
                1.146e-02,
                1.542e-02,
                2.235e-02,
                3.381e-02,
                5.117e-02,
                7.402e-02,
                9.807e-02,
                1.151e-01,
                1.171e-01,
                1.025e-01,
                7.780e-02,
                5.243e-02,
                3.286e-02,
                2.043e-02,
                1.344e-02,
                9.701e-03,
                7.655e-03,
                6.413e-03,
                5.562e-03,
                4.945e-03,
                4.496e-03,
                4.141e-03,
                3.805e-03,
                3.445e-03,
                3.067e-03,
                2.698e-03,
                2.359e-03,
                2.064e-03,
                1.825e-03,
                1.650e-03,
                1.539e-03,
                1.471e-03,
                1.409e-03,
                1.318e-03,
                1.190e-03,
                1.043e-03,
                9.061e-04,
                7.912e-04,
                6.978e-04,
                6.204e-04,
                5.554e-04,
                5.004e-04,
                4.532e-04,
                4.120e-04,
                3.769e-04,
                3.477e-04,
                3.221e-04,
                2.963e-04,
                2.683e-04,
                2.392e-04,
                2.123e-04,
                1.898e-04,
                1.718e-04,
                1.571e-04,
                1.444e-04,
                1.337e-04,
                1.246e-04,
                1.169e-04,
                1.098e-04,
            ]
        ),
        "w1650": np.array(
            [
                1.004e-04,
                1.098e-04,
                1.206e-04,
                1.325e-04,
                1.456e-04,
                1.597e-04,
                1.760e-04,
                1.960e-04,
                2.221e-04,
                2.551e-04,
                2.941e-04,
                3.370e-04,
                3.828e-04,
                4.331e-04,
                4.915e-04,
                5.623e-04,
                6.478e-04,
                7.463e-04,
                8.512e-04,
                9.532e-04,
                1.046e-03,
                1.132e-03,
                1.231e-03,
                1.364e-03,
                1.539e-03,
                1.742e-03,
                1.942e-03,
                2.113e-03,
                2.252e-03,
                2.376e-03,
                2.510e-03,
                2.672e-03,
                2.875e-03,
                3.133e-03,
                3.463e-03,
                3.880e-03,
                4.397e-03,
                5.016e-03,
                5.736e-03,
                6.568e-03,
                7.581e-03,
                8.998e-03,
                1.132e-02,
                1.541e-02,
                2.248e-02,
                3.402e-02,
                5.140e-02,
                7.430e-02,
                9.852e-02,
                1.158e-01,
                1.179e-01,
                1.031e-01,
                7.801e-02,
                5.233e-02,
                3.263e-02,
                2.021e-02,
                1.325e-02,
                9.513e-03,
                7.457e-03,
                6.216e-03,
                5.370e-03,
                4.756e-03,
                4.310e-03,
                3.974e-03,
                3.672e-03,
                3.354e-03,
                3.016e-03,
                2.681e-03,
                2.367e-03,
                2.084e-03,
                1.840e-03,
                1.644e-03,
                1.503e-03,
                1.415e-03,
                1.360e-03,
                1.303e-03,
                1.215e-03,
                1.093e-03,
                9.559e-04,
                8.295e-04,
                7.244e-04,
                6.389e-04,
                5.681e-04,
                5.090e-04,
                4.596e-04,
                4.173e-04,
                3.804e-04,
                3.486e-04,
                3.219e-04,
                2.991e-04,
                2.770e-04,
                2.529e-04,
                2.270e-04,
                2.020e-04,
                1.802e-04,
                1.627e-04,
                1.484e-04,
                1.365e-04,
                1.260e-04,
                1.172e-04,
                1.098e-04,
            ]
        ),
        "w1700": np.array(
            [
                1.027e-04,
                1.125e-04,
                1.234e-04,
                1.351e-04,
                1.478e-04,
                1.625e-04,
                1.809e-04,
                2.048e-04,
                2.349e-04,
                2.700e-04,
                3.086e-04,
                3.499e-04,
                3.959e-04,
                4.496e-04,
                5.146e-04,
                5.923e-04,
                6.808e-04,
                7.750e-04,
                8.673e-04,
                9.517e-04,
                1.029e-03,
                1.113e-03,
                1.224e-03,
                1.376e-03,
                1.561e-03,
                1.756e-03,
                1.930e-03,
                2.070e-03,
                2.183e-03,
                2.292e-03,
                2.417e-03,
                2.574e-03,
                2.778e-03,
                3.045e-03,
                3.389e-03,
                3.818e-03,
                4.335e-03,
                4.932e-03,
                5.611e-03,
                6.398e-03,
                7.392e-03,
                8.855e-03,
                1.133e-02,
                1.565e-02,
                2.297e-02,
                3.468e-02,
                5.206e-02,
                7.484e-02,
                9.888e-02,
                1.160e-01,
                1.180e-01,
                1.032e-01,
                7.796e-02,
                5.214e-02,
                3.241e-02,
                2.003e-02,
                1.311e-02,
                9.375e-03,
                7.297e-03,
                6.042e-03,
                5.194e-03,
                4.579e-03,
                4.136e-03,
                3.811e-03,
                3.533e-03,
                3.245e-03,
                2.938e-03,
                2.632e-03,
                2.347e-03,
                2.086e-03,
                1.852e-03,
                1.648e-03,
                1.486e-03,
                1.374e-03,
                1.305e-03,
                1.258e-03,
                1.202e-03,
                1.114e-03,
                9.963e-04,
                8.691e-04,
                7.543e-04,
                6.596e-04,
                5.824e-04,
                5.184e-04,
                4.651e-04,
                4.208e-04,
                3.830e-04,
                3.499e-04,
                3.210e-04,
                2.968e-04,
                2.763e-04,
                2.573e-04,
                2.366e-04,
                2.139e-04,
                1.907e-04,
                1.700e-04,
                1.530e-04,
                1.395e-04,
                1.281e-04,
                1.182e-04,
                1.097e-04,
            ]
        ),
        "w1750": np.array(
            [
                1.058e-04,
                1.157e-04,
                1.263e-04,
                1.378e-04,
                1.515e-04,
                1.688e-04,
                1.914e-04,
                2.193e-04,
                2.512e-04,
                2.856e-04,
                3.226e-04,
                3.641e-04,
                4.130e-04,
                4.722e-04,
                5.422e-04,
                6.215e-04,
                7.059e-04,
                7.895e-04,
                8.674e-04,
                9.391e-04,
                1.014e-03,
                1.109e-03,
                1.239e-03,
                1.405e-03,
                1.589e-03,
                1.761e-03,
                1.901e-03,
                2.010e-03,
                2.103e-03,
                2.200e-03,
                2.321e-03,
                2.481e-03,
                2.696e-03,
                2.979e-03,
                3.338e-03,
                3.773e-03,
                4.277e-03,
                4.846e-03,
                5.487e-03,
                6.244e-03,
                7.239e-03,
                8.772e-03,
                1.141e-02,
                1.597e-02,
                2.350e-02,
                3.530e-02,
                5.262e-02,
                7.518e-02,
                9.901e-02,
                1.160e-01,
                1.182e-01,
                1.034e-01,
                7.811e-02,
                5.212e-02,
                3.225e-02,
                1.985e-02,
                1.294e-02,
                9.210e-03,
                7.119e-03,
                5.857e-03,
                5.014e-03,
                4.405e-03,
                3.966e-03,
                3.651e-03,
                3.397e-03,
                3.142e-03,
                2.864e-03,
                2.582e-03,
                2.317e-03,
                2.076e-03,
                1.856e-03,
                1.656e-03,
                1.485e-03,
                1.351e-03,
                1.261e-03,
                1.206e-03,
                1.166e-03,
                1.110e-03,
                1.023e-03,
                9.104e-04,
                7.926e-04,
                6.880e-04,
                6.021e-04,
                5.319e-04,
                4.736e-04,
                4.252e-04,
                3.852e-04,
                3.513e-04,
                3.217e-04,
                2.956e-04,
                2.736e-04,
                2.553e-04,
                2.388e-04,
                2.214e-04,
                2.016e-04,
                1.807e-04,
                1.611e-04,
                1.446e-04,
                1.315e-04,
                1.208e-04,
                1.116e-04,
            ]
        ),
        "w1800": np.array(
            [
                1.084e-04,
                1.181e-04,
                1.288e-04,
                1.417e-04,
                1.583e-04,
                1.798e-04,
                2.057e-04,
                2.348e-04,
                2.657e-04,
                2.992e-04,
                3.372e-04,
                3.824e-04,
                4.369e-04,
                5.007e-04,
                5.719e-04,
                6.468e-04,
                7.212e-04,
                7.912e-04,
                8.561e-04,
                9.223e-04,
                1.004e-03,
                1.117e-03,
                1.265e-03,
                1.436e-03,
                1.604e-03,
                1.744e-03,
                1.849e-03,
                1.929e-03,
                2.004e-03,
                2.094e-03,
                2.217e-03,
                2.390e-03,
                2.627e-03,
                2.933e-03,
                3.306e-03,
                3.740e-03,
                4.224e-03,
                4.758e-03,
                5.365e-03,
                6.106e-03,
                7.128e-03,
                8.764e-03,
                1.159e-02,
                1.640e-02,
                2.412e-02,
                3.597e-02,
                5.313e-02,
                7.536e-02,
                9.886e-02,
                1.158e-01,
                1.181e-01,
                1.037e-01,
                7.842e-02,
                5.229e-02,
                3.223e-02,
                1.973e-02,
                1.280e-02,
                9.059e-03,
                6.950e-03,
                5.678e-03,
                4.838e-03,
                4.236e-03,
                3.800e-03,
                3.492e-03,
                3.256e-03,
                3.030e-03,
                2.782e-03,
                2.520e-03,
                2.272e-03,
                2.048e-03,
                1.845e-03,
                1.657e-03,
                1.487e-03,
                1.343e-03,
                1.234e-03,
                1.162e-03,
                1.117e-03,
                1.079e-03,
                1.022e-03,
                9.356e-04,
                8.281e-04,
                7.195e-04,
                6.246e-04,
                5.471e-04,
                4.837e-04,
                4.312e-04,
                3.879e-04,
                3.522e-04,
                3.221e-04,
                2.959e-04,
                2.726e-04,
                2.526e-04,
                2.361e-04,
                2.216e-04,
                2.069e-04,
                1.899e-04,
                1.712e-04,
                1.529e-04,
                1.370e-04,
                1.242e-04,
                1.140e-04,
            ]
        ),
    },
    "COS_G160M": {  # COS medium-res NUV grating
        "channels": np.array([1150, 1200, 1250, 1300, 1350, 1400, 1450]),
        "relwave": 12.23e-3
        * np.arange(-50, 51, 1, dtype=int),  # 12.23 mA/pixel for G160M
        "w1450": np.array(
            [
                8.099e-05,
                8.796e-05,
                9.576e-05,
                1.044e-04,
                1.140e-04,
                1.244e-04,
                1.351e-04,
                1.462e-04,
                1.590e-04,
                1.746e-04,
                1.932e-04,
                2.144e-04,
                2.381e-04,
                2.651e-04,
                2.975e-04,
                3.366e-04,
                3.817e-04,
                4.316e-04,
                4.876e-04,
                5.555e-04,
                6.436e-04,
                7.578e-04,
                8.971e-04,
                1.054e-03,
                1.223e-03,
                1.406e-03,
                1.610e-03,
                1.835e-03,
                2.066e-03,
                2.286e-03,
                2.485e-03,
                2.676e-03,
                2.888e-03,
                3.170e-03,
                3.564e-03,
                4.075e-03,
                4.653e-03,
                5.248e-03,
                5.871e-03,
                6.616e-03,
                7.610e-03,
                9.010e-03,
                1.105e-02,
                1.414e-02,
                1.909e-02,
                2.729e-02,
                4.038e-02,
                5.872e-02,
                7.952e-02,
                9.686e-02,
                1.047e-01,
                1.008e-01,
                8.742e-02,
                6.917e-02,
                5.052e-02,
                3.467e-02,
                2.314e-02,
                1.585e-02,
                1.162e-02,
                9.156e-03,
                7.559e-03,
                6.425e-03,
                5.615e-03,
                5.047e-03,
                4.615e-03,
                4.205e-03,
                3.763e-03,
                3.307e-03,
                2.888e-03,
                2.546e-03,
                2.289e-03,
                2.099e-03,
                1.946e-03,
                1.803e-03,
                1.655e-03,
                1.501e-03,
                1.345e-03,
                1.192e-03,
                1.048e-03,
                9.229e-04,
                8.195e-04,
                7.375e-04,
                6.712e-04,
                6.143e-04,
                5.634e-04,
                5.176e-04,
                4.756e-04,
                4.347e-04,
                3.933e-04,
                3.524e-04,
                3.146e-04,
                2.822e-04,
                2.550e-04,
                2.319e-04,
                2.115e-04,
                1.931e-04,
                1.762e-04,
                1.606e-04,
                1.468e-04,
                1.351e-04,
                1.250e-04,
            ]
        ),
        "w1500": np.array(
            [
                8.277e-05,
                8.956e-05,
                9.678e-05,
                1.047e-04,
                1.134e-04,
                1.229e-04,
                1.337e-04,
                1.466e-04,
                1.617e-04,
                1.791e-04,
                1.985e-04,
                2.198e-04,
                2.445e-04,
                2.740e-04,
                3.092e-04,
                3.498e-04,
                3.959e-04,
                4.490e-04,
                5.133e-04,
                5.935e-04,
                6.929e-04,
                8.132e-04,
                9.532e-04,
                1.109e-03,
                1.278e-03,
                1.460e-03,
                1.653e-03,
                1.856e-03,
                2.062e-03,
                2.259e-03,
                2.443e-03,
                2.626e-03,
                2.846e-03,
                3.155e-03,
                3.575e-03,
                4.069e-03,
                4.566e-03,
                5.050e-03,
                5.596e-03,
                6.336e-03,
                7.404e-03,
                8.923e-03,
                1.105e-02,
                1.413e-02,
                1.896e-02,
                2.702e-02,
                4.004e-02,
                5.853e-02,
                7.984e-02,
                9.804e-02,
                1.067e-01,
                1.030e-01,
                8.891e-02,
                6.947e-02,
                4.984e-02,
                3.358e-02,
                2.212e-02,
                1.507e-02,
                1.105e-02,
                8.720e-03,
                7.196e-03,
                6.092e-03,
                5.291e-03,
                4.747e-03,
                4.379e-03,
                4.068e-03,
                3.720e-03,
                3.320e-03,
                2.912e-03,
                2.551e-03,
                2.264e-03,
                2.050e-03,
                1.895e-03,
                1.777e-03,
                1.667e-03,
                1.549e-03,
                1.418e-03,
                1.276e-03,
                1.130e-03,
                9.876e-04,
                8.610e-04,
                7.577e-04,
                6.781e-04,
                6.161e-04,
                5.645e-04,
                5.188e-04,
                4.776e-04,
                4.409e-04,
                4.070e-04,
                3.730e-04,
                3.381e-04,
                3.039e-04,
                2.728e-04,
                2.461e-04,
                2.236e-04,
                2.040e-04,
                1.865e-04,
                1.706e-04,
                1.558e-04,
                1.424e-04,
                1.306e-04,
            ]
        ),
        "w1550": np.array(
            [
                8.329e-05,
                8.927e-05,
                9.598e-05,
                1.037e-04,
                1.126e-04,
                1.230e-04,
                1.351e-04,
                1.490e-04,
                1.648e-04,
                1.824e-04,
                2.022e-04,
                2.251e-04,
                2.518e-04,
                2.833e-04,
                3.198e-04,
                3.624e-04,
                4.127e-04,
                4.728e-04,
                5.448e-04,
                6.312e-04,
                7.350e-04,
                8.588e-04,
                1.001e-03,
                1.158e-03,
                1.322e-03,
                1.491e-03,
                1.667e-03,
                1.852e-03,
                2.040e-03,
                2.221e-03,
                2.393e-03,
                2.578e-03,
                2.820e-03,
                3.154e-03,
                3.566e-03,
                3.998e-03,
                4.406e-03,
                4.826e-03,
                5.369e-03,
                6.165e-03,
                7.321e-03,
                8.916e-03,
                1.107e-02,
                1.410e-02,
                1.888e-02,
                2.695e-02,
                4.005e-02,
                5.864e-02,
                8.012e-02,
                9.858e-02,
                1.075e-01,
                1.037e-01,
                8.935e-02,
                6.957e-02,
                4.975e-02,
                3.344e-02,
                2.197e-02,
                1.487e-02,
                1.082e-02,
                8.486e-03,
                6.982e-03,
                5.885e-03,
                5.067e-03,
                4.503e-03,
                4.145e-03,
                3.888e-03,
                3.623e-03,
                3.296e-03,
                2.928e-03,
                2.574e-03,
                2.272e-03,
                2.034e-03,
                1.857e-03,
                1.731e-03,
                1.636e-03,
                1.548e-03,
                1.450e-03,
                1.335e-03,
                1.205e-03,
                1.066e-03,
                9.273e-04,
                8.032e-04,
                7.022e-04,
                6.251e-04,
                5.662e-04,
                5.184e-04,
                4.766e-04,
                4.389e-04,
                4.057e-04,
                3.763e-04,
                3.485e-04,
                3.198e-04,
                2.899e-04,
                2.612e-04,
                2.353e-04,
                2.133e-04,
                1.945e-04,
                1.780e-04,
                1.631e-04,
                1.495e-04,
                1.370e-04,
            ]
        ),
        "w1600": np.array(
            [
                8.207e-05,
                8.817e-05,
                9.526e-05,
                1.036e-04,
                1.132e-04,
                1.242e-04,
                1.369e-04,
                1.512e-04,
                1.673e-04,
                1.855e-04,
                2.065e-04,
                2.308e-04,
                2.590e-04,
                2.920e-04,
                3.309e-04,
                3.771e-04,
                4.315e-04,
                4.953e-04,
                5.706e-04,
                6.612e-04,
                7.713e-04,
                9.010e-04,
                1.045e-03,
                1.194e-03,
                1.343e-03,
                1.497e-03,
                1.659e-03,
                1.832e-03,
                2.005e-03,
                2.169e-03,
                2.333e-03,
                2.528e-03,
                2.791e-03,
                3.131e-03,
                3.511e-03,
                3.879e-03,
                4.228e-03,
                4.626e-03,
                5.188e-03,
                6.025e-03,
                7.216e-03,
                8.808e-03,
                1.090e-02,
                1.384e-02,
                1.856e-02,
                2.667e-02,
                3.993e-02,
                5.880e-02,
                8.066e-02,
                9.950e-02,
                1.086e-01,
                1.047e-01,
                8.998e-02,
                6.980e-02,
                4.971e-02,
                3.328e-02,
                2.176e-02,
                1.463e-02,
                1.057e-02,
                8.245e-03,
                6.771e-03,
                5.697e-03,
                4.877e-03,
                4.294e-03,
                3.923e-03,
                3.688e-03,
                3.482e-03,
                3.227e-03,
                2.913e-03,
                2.582e-03,
                2.281e-03,
                2.031e-03,
                1.834e-03,
                1.687e-03,
                1.584e-03,
                1.507e-03,
                1.433e-03,
                1.346e-03,
                1.242e-03,
                1.120e-03,
                9.874e-04,
                8.555e-04,
                7.379e-04,
                6.429e-04,
                5.707e-04,
                5.157e-04,
                4.717e-04,
                4.336e-04,
                3.996e-04,
                3.695e-04,
                3.434e-04,
                3.195e-04,
                2.952e-04,
                2.694e-04,
                2.431e-04,
                2.190e-04,
                1.982e-04,
                1.808e-04,
                1.657e-04,
                1.522e-04,
                1.400e-04,
            ]
        ),
        "w1650": np.array(
            [
                8.084e-05,
                8.754e-05,
                9.533e-05,
                1.041e-04,
                1.143e-04,
                1.257e-04,
                1.386e-04,
                1.528e-04,
                1.691e-04,
                1.881e-04,
                2.105e-04,
                2.363e-04,
                2.662e-04,
                3.010e-04,
                3.419e-04,
                3.901e-04,
                4.466e-04,
                5.138e-04,
                5.951e-04,
                6.943e-04,
                8.114e-04,
                9.414e-04,
                1.077e-03,
                1.213e-03,
                1.351e-03,
                1.495e-03,
                1.648e-03,
                1.803e-03,
                1.951e-03,
                2.095e-03,
                2.254e-03,
                2.463e-03,
                2.746e-03,
                3.089e-03,
                3.440e-03,
                3.762e-03,
                4.078e-03,
                4.469e-03,
                5.037e-03,
                5.869e-03,
                7.025e-03,
                8.551e-03,
                1.055e-02,
                1.340e-02,
                1.807e-02,
                2.625e-02,
                3.975e-02,
                5.904e-02,
                8.140e-02,
                1.006e-01,
                1.099e-01,
                1.060e-01,
                9.100e-02,
                7.040e-02,
                4.986e-02,
                3.310e-02,
                2.143e-02,
                1.427e-02,
                1.022e-02,
                7.945e-03,
                6.525e-03,
                5.504e-03,
                4.713e-03,
                4.125e-03,
                3.736e-03,
                3.500e-03,
                3.327e-03,
                3.130e-03,
                2.868e-03,
                2.566e-03,
                2.272e-03,
                2.019e-03,
                1.813e-03,
                1.652e-03,
                1.534e-03,
                1.450e-03,
                1.383e-03,
                1.312e-03,
                1.227e-03,
                1.126e-03,
                1.011e-03,
                8.870e-04,
                7.660e-04,
                6.594e-04,
                5.736e-04,
                5.085e-04,
                4.594e-04,
                4.203e-04,
                3.868e-04,
                3.568e-04,
                3.299e-04,
                3.066e-04,
                2.854e-04,
                2.643e-04,
                2.419e-04,
                2.193e-04,
                1.981e-04,
                1.800e-04,
                1.645e-04,
                1.513e-04,
                1.392e-04,
            ]
        ),
        "w1700": np.array(
            [
                8.009e-05,
                8.719e-05,
                9.562e-05,
                1.053e-04,
                1.160e-04,
                1.275e-04,
                1.403e-04,
                1.553e-04,
                1.731e-04,
                1.939e-04,
                2.176e-04,
                2.440e-04,
                2.740e-04,
                3.093e-04,
                3.519e-04,
                4.038e-04,
                4.668e-04,
                5.435e-04,
                6.354e-04,
                7.406e-04,
                8.549e-04,
                9.749e-04,
                1.099e-03,
                1.227e-03,
                1.358e-03,
                1.490e-03,
                1.619e-03,
                1.743e-03,
                1.865e-03,
                1.998e-03,
                2.169e-03,
                2.407e-03,
                2.716e-03,
                3.057e-03,
                3.377e-03,
                3.661e-03,
                3.960e-03,
                4.352e-03,
                4.911e-03,
                5.700e-03,
                6.781e-03,
                8.220e-03,
                1.014e-02,
                1.295e-02,
                1.764e-02,
                2.597e-02,
                3.987e-02,
                5.976e-02,
                8.269e-02,
                1.023e-01,
                1.116e-01,
                1.075e-01,
                9.204e-02,
                7.085e-02,
                4.973e-02,
                3.258e-02,
                2.079e-02,
                1.368e-02,
                9.728e-03,
                7.521e-03,
                6.172e-03,
                5.234e-03,
                4.516e-03,
                3.961e-03,
                3.568e-03,
                3.321e-03,
                3.164e-03,
                3.010e-03,
                2.798e-03,
                2.529e-03,
                2.246e-03,
                1.992e-03,
                1.782e-03,
                1.612e-03,
                1.481e-03,
                1.385e-03,
                1.312e-03,
                1.244e-03,
                1.167e-03,
                1.079e-03,
                9.792e-04,
                8.697e-04,
                7.575e-04,
                6.519e-04,
                5.611e-04,
                4.889e-04,
                4.345e-04,
                3.937e-04,
                3.609e-04,
                3.325e-04,
                3.069e-04,
                2.848e-04,
                2.662e-04,
                2.495e-04,
                2.321e-04,
                2.130e-04,
                1.932e-04,
                1.747e-04,
                1.586e-04,
                1.448e-04,
                1.327e-04,
            ]
        ),
        "w1750": np.array(
            [
                7.992e-05,
                8.776e-05,
                9.662e-05,
                1.060e-04,
                1.159e-04,
                1.273e-04,
                1.413e-04,
                1.584e-04,
                1.784e-04,
                2.003e-04,
                2.233e-04,
                2.481e-04,
                2.770e-04,
                3.134e-04,
                3.601e-04,
                4.197e-04,
                4.930e-04,
                5.782e-04,
                6.709e-04,
                7.678e-04,
                8.709e-04,
                9.841e-04,
                1.107e-03,
                1.232e-03,
                1.348e-03,
                1.452e-03,
                1.546e-03,
                1.641e-03,
                1.748e-03,
                1.887e-03,
                2.084e-03,
                2.356e-03,
                2.684e-03,
                3.015e-03,
                3.311e-03,
                3.582e-03,
                3.883e-03,
                4.267e-03,
                4.773e-03,
                5.454e-03,
                6.403e-03,
                7.744e-03,
                9.647e-03,
                1.252e-02,
                1.737e-02,
                2.601e-02,
                4.037e-02,
                6.070e-02,
                8.375e-02,
                1.030e-01,
                1.120e-01,
                1.078e-01,
                9.262e-02,
                7.169e-02,
                5.060e-02,
                3.322e-02,
                2.106e-02,
                1.363e-02,
                9.455e-03,
                7.132e-03,
                5.754e-03,
                4.862e-03,
                4.232e-03,
                3.752e-03,
                3.385e-03,
                3.127e-03,
                2.960e-03,
                2.827e-03,
                2.661e-03,
                2.439e-03,
                2.187e-03,
                1.945e-03,
                1.735e-03,
                1.562e-03,
                1.423e-03,
                1.317e-03,
                1.236e-03,
                1.163e-03,
                1.088e-03,
                1.006e-03,
                9.153e-04,
                8.190e-04,
                7.207e-04,
                6.268e-04,
                5.426e-04,
                4.710e-04,
                4.134e-04,
                3.690e-04,
                3.353e-04,
                3.075e-04,
                2.825e-04,
                2.597e-04,
                2.401e-04,
                2.238e-04,
                2.088e-04,
                1.930e-04,
                1.756e-04,
                1.584e-04,
                1.429e-04,
                1.300e-04,
                1.193e-04,
            ]
        ),
    },
}
