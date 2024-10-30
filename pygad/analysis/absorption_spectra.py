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

## loading line data from .csv file [initially saved using lines dict defined in this file,
## .csv file can be manually updated with updated line data.]
## 240803: dict moved from beginning of file to end of file, commented out;
## 240812: commented out line dict moved to unused lines_LSF_data.txt file (only for visual purpose)
## can still access lines in the same way as before
## e.g., lines = pg.analysis.absorption_spectra.lines
# pd.options.mode.chained_assignment = None  # default='warn'
# df = pd.read_csv(environment.module_dir+"analysis/line_data.csv",index_col=0)
# df['A_ki'][df['A_ki'].isnull()] = 0.0
# lines = df.to_dict(orient='index')
# lines["Lyman_alpha"] = lines["H1215"]
# lines["Lyman_beta"] = lines["H1025"]
# lines["Lyman_gamma"] = lines["H972"]
datz = np.genfromtxt(
    environment.module_dir + "analysis/line_data.csv", 
    dtype=[("", "U20"), ("ion", "U10"), ("l", "U20"), ("f", "U20"), ("atomwt", "U20"), ("A_ki", "U20"), ("element", "U2")],
    delimiter=",", 
    skip_header=1)
for i, line in enumerate(datz):
    if line["A_ki"] == '':
        datz[i]["A_ki"] = "0.0 s**-1"
lines = {}
for line in datz:
    lines[line[0]] = {"ion": line[1], "l": line[2], "f": line[3], 
                      "atomwt": line[4], "A_ki": line[5], "element": line[6]}
    
## loading LSF data from .npy file [initially saved using LSF_data dict defined in this file,
## towards the end, ~3k lines, now moved to an unused LSF_data.txt file (only for visual purpose)]
## 240812: renamed LSF_data.txt to lines_LSF_data.txt
## can still access lines in the same way as before
## e.g., LSF_data = pg.analysis.absorption_spectra.LSF_data
lsfdata = np.load(environment.module_dir + "analysis/LSF_data.npy", allow_pickle=True)
LSF_data = lsfdata[()]

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
    return gamma / (np.pi * (x**2 + gamma**2))


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
    sigma0 = f * q_e**2 / (4.0 * epsilon0 * m_e * c)
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
        print((
            "find all necessary particles, beginning with those that have "
            + "the highest column density along the line of sight, that are "
            + "needed for getting %.1f%% of the total EW" % (100.0 * threshold)
        ))
        if isinstance(line, str):
            print(('  line "%s" at %s' % (line, los)))

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
        print(("in %s space EW = %s" % (EW_space, EW_full)))

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
        print((
            "%s of the %s N_intersecting particles needed "
            % (
                utils.nice_big_num_str(np.sum(contributing)),
                utils.nice_big_num_str(N_intersecting),
            )
            + "for a line with >= %.1f%% of the EW" % (100.0 * threshold)
        ))

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
    print (len(los))
    if len(los) == 2:
        print ("single LOS")
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
    else:
        print ("multiple LOS, number of LOS is ", len(los))
        return mock_absorption_spectra_multilos(
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
        los_dens_phys (UnitArr):The gas density for the velocity bins (in g cm^-3),
                                if return_los_phys=True.
        los_temp (UnitArr):     The (mass-weighted) particle temperatures
                                restricted to the velocity bins (in K).
        los_metal_frac(UnitArr):The metal mass fraction for the velocity bins, or
                                the metallicity (if no element is defined, or if 
                                element is H or He), if return_los_phys=True.
        los_vpec (UnitArr):     The LOS velocities of particles (in km/s) [formerly
                                defined as 'vel'], if return_los_phys=True.
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
    s0 = q_e**2 / (4.0 * epsilon0 * m_e * c)
    Xsec = f * s0 * l
    Xsec = Xsec.in_units_of(l_units**2 * v_units, subs=s)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print("create a mock absorption spectrum:")
        print(("  at", los))
        if isinstance(ion, str):
            print(("  for", ion, "at lambda =", l))
        else:
            print(("  at lambda =", l))
        print(("  with oscillator strength f =", f))
        print(("  => Xsec =", Xsec))
        print(("  and atomic weight", atomwt))
        print(("  => b(T=1e4K) =", b_0 * np.sqrt(1e4)))
        print(("  and a lifetime of 1/A_ki =", (1.0 / A_ki)))
        print(("  => Gamma =", Gamma))
        if v_turb is not None:
            v_perc = np.percentile(v_turb, [10, 90])
            print((
                "  and a turbulent motion per particle of v_turb ~(%.1f - %.1f) %s"
                % (v_perc[0], v_perc[-1], v_turb.units)
            ))
        print(('  using kernel "%s"' % kernel))

    v_edges = UnitArr(
        np.linspace(float(vel_extent[0]), float(vel_extent[1]), Nbins + 1),
        vel_extent.units,
    )

    # get ne number of ions per particle
    if isinstance(ion, str):
        ion = s.gas.get(ion)
    else:
        ion = UnitQty(ion, units=s["mass"].units, subs=s)
    #print (ion.units)
    #print ((ion.astype(np.float64) / atomwt).units)
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
            Npx = (1 + 2 * pad) * np.ones(3, dtype=np.int64)
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
                print(("  using an spatial extent of:", spatial_extent))
                print((
                    "  ... with %d bins of size %sx%s^2" % (N, col_width, spatial_res)
                ))

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
                print(("  using an spatial extent of:", spatial_extent))
                print(("  ... with %d bins of length %s" % (N, spatial_res)))

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

        n.convert_to(l_units**-2, subs=s)
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
    Xsec = float(Xsec.in_units_of(l_units**2 * v_units, subs=s))
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
            print(("  EW =", EW_l))
            print(("  v0 =", v_mean))
            print(("  l0 =", l_mean))
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

def mock_absorption_spectra_multilos(
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
        los_dens_phys (UnitArr):The gas density for the velocity bins (in g cm^-3),
                                if return_los_phys=True.
        los_temp (UnitArr):     The (mass-weighted) particle temperatures
                                restricted to the velocity bins (in K).
        los_metal_frac(UnitArr):The metal mass fraction for the velocity bins, or
                                the metallicity (if no element is defined, or if 
                                element is H or He), if return_los_phys=True.
        los_vpec (UnitArr):     The LOS velocities of particles (in km/s) [formerly
                                defined as 'vel'], if return_los_phys=True.
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
    # if ion == 'HI':
    #     velstat = True
    # else:
    #     velstat = False
    zaxis = (set([0, 1, 2]) - set([xaxis, yaxis])).pop()
    if set([xaxis, yaxis, zaxis]) != set([0, 1, 2]):
        raise ValueError("x- and y-axis must be in [0,1,2] and different!")
    los_arr = UnitQty(los, s["pos"].units, dtype=np.float64, subs=s)
    Nlos = len(los_arr)
    # print ("number of LOS is ", Nlos)
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
    s0 = q_e**2 / (4.0 * epsilon0 * m_e * c)
    Xsec = f * s0 * l
    Xsec = Xsec.in_units_of(l_units**2 * v_units, subs=s)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print("create a mock absorption spectrum:")
        print(("  at", len(los_arr)), "positions ")
        if isinstance(ion, str):
            print(("  for", ion, "at lambda =", l))
        else:
            print(("  at lambda =", l))
        print(("  with oscillator strength f =", f))
        print(("  => Xsec =", Xsec))
        print(("  and atomic weight", atomwt))
        print(("  => b(T=1e4K) =", b_0 * np.sqrt(1e4)))
        print(("  and a lifetime of 1/A_ki =", (1.0 / A_ki)))
        print(("  => Gamma =", Gamma))
        if v_turb is not None:
            v_perc = np.percentile(v_turb, [10, 90])
            print((
                "  and a turbulent motion per particle of v_turb ~(%.1f - %.1f) %s"
                % (v_perc[0], v_perc[-1], v_turb.units)
            ))
        print(('  using kernel "%s"' % kernel))

    v_edges = UnitArr(
        np.linspace(float(vel_extent[0]), float(vel_extent[1]), Nbins + 1),
        vel_extent.units,
    )

    # get ne number of ions per particle
    if isinstance(ion, str):
        ion = s.gas.get(ion)
    else:
        ion = UnitQty(ion, units=s["mass"].units, subs=s)
    #print (ion.units)
    #print ((ion.astype(np.float64) / atomwt).units)
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
            Npx = (1 + 2 * pad) * np.ones(3, dtype=np.int64)
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
                print(("  using an spatial extent of:", spatial_extent))
                print((
                    "  ... with %d bins of size %sx%s^2" % (N, col_width, spatial_res)
                ))

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
                print(("  using an spatial extent of:", spatial_extent))
                print(("  ... with %d bins of length %s" % (N, spatial_res)))

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

        n.convert_to(l_units**-2, subs=s)
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
    # if velstat:
    #     print ("some vel stats:")
    #     print ("Hflo -- min:", f'{np.min(H_flow):.3f} km/s', "max:", f'{np.max(H_flow):.3f} km/s')
    #     print ("vpec -- min:", f'{np.min(vpec_z):.3f} km/s', "max:", f'{np.max(vpec_z):.3f} km/s')
    #     print ("vsum -- min:", f'{np.min(vel):.3f} km/s', "max:", f'{np.max(vel):.3f} km/s')

    # zero_Hubble_flow_at.convert_to(los_pos.units, subs=s)
    # # H_flow = s.cosmology.H(s.redshift) * (los_pos - zero_Hubble_flow_at)
    # H_flow = UnitArr(0, "km/s")
    # H_flow.convert_to(vel.units, subs=s)
    # print("H_flow is set to ", H_flow)
    # vpec_z = vel  # DS: peculiar LOS velocities
    # vel = vel + H_flow

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

    los_arr = los_arr.in_units_of(l_units, subs=s).view(np.ndarray).astype(np.float64).copy()
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
    Xsec = float(Xsec.in_units_of(l_units**2 * v_units, subs=s))
    Gamma = float(Gamma.in_units_of(v_units))

    taus = np.empty((Nlos,Nbins), dtype=np.float64)
    los_dens = np.empty((Nlos,Nbins), dtype=np.float64)
    los_dens_phys = np.empty((Nlos,Nbins), dtype=np.float64)  # DS: gas density field
    los_temp = np.empty((Nlos,Nbins), dtype=np.float64)
    los_vpec = np.empty((Nlos,Nbins), dtype=np.float64)  # DS: LOS peculiar velocity field
    los_metal_frac = np.empty((Nlos,Nbins), dtype=np.float64)  # SA: LOS metallicity field
    restr_column_lims = restr_column_lims.view(np.ndarray).astype(np.float64)
    full_column_calc = False
    if full_column_calc:
        restr_column = np.empty((Nlos,N), dtype=np.float64)
    else:
        restr_column = np.empty((1), dtype=np.float64)
    C.cpygad.absorption_spectrum_multiple_los(
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
        C.c_void_p(los_arr.ctypes.data),
        C.c_void_p(vel_extent.ctypes.data),
        C.c_size_t(Nbins),
        C.c_size_t(Nlos),
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
            print(("  EW =", EW_l))
            print(("  v0 =", v_mean))
            print(("  l0 =", l_mean))
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
            print((
                "warning: continuum fitting failed after %d iterations: diff=%g > tol=%g"
                % (n_iter, diff, tol)
            ))
            break
        if len(l_unabs) < 0.1 * len(l):
            print((
                "warning: continuum fitting failed, too few pixels left: only %d of %d"
                % (len(l_unabs), len(l))
            ))
            break
        n_iter += 1
    contin = np.polyval(p, l)
    print((
        "Continuum fit done: Median (using %d/%d pixels, %d iterations) is %g"
        % (len(l_unabs), len(l), n_iter, med_contin)
    ))
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print(("contin=", contin))
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
    print((
        "Applied LSF at <lambda>=%1g: %s, channel %s" % (np.mean(l), grating, channel)
    ))  # ,np.mean(flux_conv)-np.mean(flux)))
    return flux_conv, noise_conv
