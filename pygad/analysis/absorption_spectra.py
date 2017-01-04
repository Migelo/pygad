"""
Produce mock absorption spectra for given line transition(s) and line-of-sight(s).

Doctests:
    >>> from ..environment import module_dir
    >>> from ..snapshot import Snap
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_320', physical=False)

    >>> vs = UnitArr([1,1e3,1e6], 'km/s')
    >>> print velocities_to_redshifts(vs, z0=0.123)
    [ 0.12300375  0.12674592  3.86892479]
    >>> for z0 in (0.0, 0.123, 1.0, 2.34):
    ...     zs = velocities_to_redshifts(vs, z0=z0)
    ...     vs_back = redshifts_to_velocities(zs, z0=z0)
    ...     if np.max(np.abs( (vs-vs_back)/vs-1 ))>1e-4:
    ...         print vs
    ...         print redshifts_to_velocities(zs, z0=0.123)
    
    >>> los_arr = UnitArr([[ 34700.,  35600.],
    ...                    [ 34550.,  35500.],
    ...                    [ 35000.,  35600.]], 'ckpc/h_0')
    >>> environment.verbose = environment.VERBOSE_QUIET
    >>> for los in los_arr:
    ...     print 'l.o.s.:', los
    ...     for line in ['H1215', 'OVI1031']:
    ...         print '  ', line
    ...         for method in ['particles', 'line', 'column']:
    ...             tau, dens, temp, v_edges, restr_column = mock_absorption_spectrum_of(
    ...                 s, los, line=line,
    ...                 vel_extent=UnitArr([2400.,3100.], 'km/s'),
    ...                 method=method,
    ...             )
    ...             N = dens.sum()
    ...             print '    N  = %.3e %s' % (N, N.units)
    ...             if method == 'particles':
    ...                 N_restr = np.sum(restr_column)
    ...                 N_restr.convert_to('cm**-2', subs=s)
    ...                 if np.abs((N_restr - N) / N) > 0.01:
    ...                     print '    N  = %.3e %s' % (N_restr, N_restr.units)
    ...             z_edges = velocities_to_redshifts(v_edges, z0=s.redshift)
    ...             l = UnitArr(lines[line]['l'])
    ...             l_edges = l * (1.0 + z_edges)
    ...             EW_l = EW(tau, l_edges)
    ...             print '    EW = %.3f %s' % (EW_l, EW_l.units)
    l.o.s.: [ 34700.  35600.] [ckpc h_0**-1]
       H1215
        N  = 2.301e+15 [cm**-2]
        EW = 1.401 [Angstrom]
        N  = 2.299e+15 [cm**-2]
        EW = 1.129 [Angstrom]
        N  = 2.304e+15 [cm**-2]
        EW = 1.111 [Angstrom]
       OVI1031
        N  = 7.455e+14 [cm**-2]
        EW = 0.673 [Angstrom]
        N  = 7.447e+14 [cm**-2]
        EW = 0.547 [Angstrom]
        N  = 7.462e+14 [cm**-2]
        EW = 0.539 [Angstrom]
    l.o.s.: [ 34550.  35500.] [ckpc h_0**-1]
       H1215
        N  = 5.303e+14 [cm**-2]
        EW = 0.628 [Angstrom]
        N  = 5.297e+14 [cm**-2]
        EW = 0.565 [Angstrom]
        N  = 5.309e+14 [cm**-2]
        EW = 0.559 [Angstrom]
       OVI1031
        N  = 2.252e+14 [cm**-2]
        EW = 0.281 [Angstrom]
        N  = 2.250e+14 [cm**-2]
        EW = 0.256 [Angstrom]
        N  = 2.254e+14 [cm**-2]
        EW = 0.253 [Angstrom]
    l.o.s.: [ 35000.  35600.] [ckpc h_0**-1]
       H1215
        N  = 4.382e+13 [cm**-2]
        EW = 0.289 [Angstrom]
        N  = 4.377e+13 [cm**-2]
        EW = 0.284 [Angstrom]
        N  = 4.386e+13 [cm**-2]
        EW = 0.284 [Angstrom]
       OVI1031
        N  = 1.210e+14 [cm**-2]
        EW = 0.186 [Angstrom]
        N  = 1.209e+14 [cm**-2]
        EW = 0.182 [Angstrom]
        N  = 1.211e+14 [cm**-2]
        EW = 0.182 [Angstrom]
    >>> environment.verbose = environment.VERBOSE_NORMAL
"""
__all__ = ['mock_absorption_spectrum_of', 'mock_absorption_spectrum',
           'EW', 'Voigt', 'Gaussian', 'Lorentzian', 'find_line_contributers',
           'velocities_to_redshifts', 'redshifts_to_velocities',
           ]

from ..units import Unit, UnitArr, UnitQty, UnitScalar
from ..physics import kB, m_H, c, q_e, m_e, epsilon0
from ..kernels import *
from .. import gadget
from .. import utils
from .. import C
from .. import environment
import numpy as np

lines = {
    'H1215':     {'ion':'HI',     'l':'1215.6701 Angstrom', 'f':0.4164,
                    'atomwt':m_H,       'gamma':'6.06076e-3 km/s'},
    'H1025':     {'ion':'HI',     'l':'1025.722 Angstrom',  'f':0.079121,
                    'atomwt':m_H},
    'H972':      {'ion':'HI',     'l':'972.5368 Angstrom',  'f':2.900e-02,
                    'atomwt':m_H},

    'HeII':      {'ion':'HeII',   'l': '303.918 Angstrom',  'f':0.4173, 
                    'atomwt':'3.971 u'},

    'CII1036':   {'ion':'CII',    'l':'1036.337 Angstrom',  'f':0.1270,
                    'atomwt':'12.011 u'},
    'CIII977':   {'ion':'CIII',   'l': '977.020 Angstrom',  'f':0.7620,
                    'atomwt':'12.011 u'},
    'CIV1548':   {'ion':'CIV',    'l':'1548.195 Angstrom',  'f':0.1908,
                    'atomwt':'12.011 u'},
    'CIV1550':   {'ion':'CIV',    'l':'1550.777 Angstrom',  'f':0.09520,
                    'atomwt':'12.011 u'},

    'OIV787':    {'ion':'OIV',    'l': '787.711 Angstrom',  'f':0.110,
                    'atomwt':'15.9994 u'},
    'OVI1031':   {'ion':'OVI',    'l':'1031.927 Angstrom',  'f':0.1329,
                    'atomwt':'15.9994 u'},

    'NeVIII770': {'ion':'NeVIII', 'l': '770.409 Angstrom',  'f':0.103,
                    'atomwt':'20.180 u'},

    'MgII2796':  {'ion':'MgII',   'l':'2796.352 Angstrom',  'f':0.6123,
                    'atomwt':'24.305 u'},

    'SiII1260':  {'ion':'SiII',  'l':'1260.522 Angstrom',  'f':1.180,
                    'atomwt':'28.086 u'},
    'SiIII1206': {'ion':'SiIII',  'l':'1206.500 Angstrom',  'f':1.669,
                    'atomwt':'28.086 u'},
    'SiIV1393':  {'ion':'SiIV',   'l':'1393.755 Angstrom',  'f':0.5140,
                    'atomwt':'28.086 u'},
    'SiIV1402':  {'ion':'SiIV',   'l':'1402.770 Angstrom',  'f':0.2553,
                    'atomwt':'28.086 u'},
}
lines['Lyman_alpha'] = lines['H1215']
lines['Lyman_beta']  = lines['H1025']
lines['Lyman_gamma'] = lines['H972']

def Gaussian(x, sigma):
    '''
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
    '''
    return np.exp(-0.5*(x/sigma)**2) / ( sigma * np.sqrt(2.*np.pi) )

def Lorentzian(x, gamma):
    '''
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
    '''
    return gamma / (np.pi * (x**2 + gamma**2))

def Voigt(x, sigma, gamma):
    '''
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
    '''
    from scipy.special import wofz
    z = (x + 1j*gamma) / (sigma * np.sqrt(2.))
    return np.real(wofz(z)) / ( sigma * np.sqrt(2.*np.pi) )

def find_line_contributers(s, los, line, vel_extent, threshold=0.95,
                           EW_space='wavelength', **kwargs):
    '''
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
    '''
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'find all necessary particles, beginning with those that have ' + \
              'the highest column density along the line of sight, that are ' + \
              'needed for getting %.1f%% of the total EW' % (100.*threshold)
        if isinstance(line, str):
            print '  line "%s" at %s' % (line, los)

    if isinstance(line,str):
        line = lines[line]
    taus, dens, temp, v_edges, restr_column = mock_absorption_spectrum_of(
            s.gas, los=los, line=line, vel_extent=vel_extent, **kwargs)
    N_intersecting = np.sum(restr_column>0)

    z_edges = velocities_to_redshifts(v_edges, z0=s.redshift)
    l_edges = UnitScalar(line['l']) * (1.0 + z_edges)
    if EW_space == 'wavelength':
        edges = l_edges
    elif EW_space == 'frequency':
        edges = -(c / l_edges).in_units_of('Hz')
    elif EW_space == 'redshift':
        edges = z_edges
    elif EW_space == 'velocity':
        edges = v_edges
    else:
        raise ValueError('Unknown `EW_space`: "%s"!' % EW_space)
    EW_full = EW(taus, edges)
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'in %s space EW = %s' % (EW_space, EW_full)

    # bisect by percentiles
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'finding the necessary particles...'
    low, mid, high = 0.,50.,100.
    Nlow, Nmid, Nhigh = map( lambda x: np.percentile(restr_column,x),
                             [low,mid,high] )
    verbosity = environment.verbose
    environment.verbose = environment.VERBOSE_QUIET
    while np.sum(restr_column>Nlow) > np.sum(restr_column>Nhigh) + 1:
        mid = (low + high) / 2.
        Nmid = np.percentile(restr_column, mid)
        taus, _, _, _, _ = mock_absorption_spectrum_of(
                s.gas[restr_column>Nmid], los=los, line=line,
                vel_extent=vel_extent, **kwargs)
        E = EW(taus,edges)
        if E < threshold*EW_full:
            high, Nhigh = mid, Nmid
        else:
            low, Nlow = mid, Nmid
    environment.verbose = verbosity
    contributing = np.array( (restr_column>Nmid), dtype=bool)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print '%s of the %s N_intersecting particles needed ' % (
                utils.nice_big_num_str(np.sum(contributing)),
                utils.nice_big_num_str(N_intersecting)) + \
              'for a line with >= %.1f%% of the EW' % (100.*threshold)

    return contributing

def mock_absorption_spectrum_of(s, los, line, vel_extent, **kwargs):
    '''
    Create a mock absorption spectrum for the given line of sight (l.o.s.) for the
    given line transition.

    This function basically just calls `mock_absorption_spectrum` for the given
    line:

        if isinstance(line,str):
            line = lines[line]
        return mock_absorption_spectrum(s, los, line['ion'],
                                        l=line['l'], f=line['f'],
                                        atomwt=line['atomwt'],
                                        vel_extent=vel_extent,
                                        **kwargs)
    '''
    try:
        if isinstance(line,str):
            line = lines[line]
        elif not isinstance(line,dict):
            raise ValueError('`line` needs to be a string or a dictionary, ' +
                             'not %s!' % type(line))
    except KeyError:
        raise KeyError("unkown line '%s' -- " % line +
                       "see `analysis.absorption_spectra.lines.keys()`")
    return mock_absorption_spectrum(s, los, line['ion'],
                                    l=line['l'], f=line['f'],
                                    atomwt=line['atomwt'],
                                    vel_extent=vel_extent,
                                    **kwargs)

def mock_absorption_spectrum(s, los, ion, l, f, atomwt,
                             vel_extent, Nbins=1000,
                             method='particles',
                             spatial_extent=None, spatial_res=None,
                             col_width=None, pad=7,
                             hsml='hsml', kernel=None,
                             restr_column_lims=None,
                             zero_Hubble_flow_at=0,
                             xaxis=0, yaxis=1):
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

    Returns:
        taus (np.ndarray):      The optical depths for the velocity bins.
        los_dens (UnitArr):     The column densities restricted to the velocity
                                bins (in cm^-2).
        los_temp (UnitArr):     The (mass-weighted) particle temperatures
                                restricted to the velocity bins (in K).
        v_edges (UnitArr):      The velocities at the bin edges.
        restr_column (np.ndarray):
                                The column densities of the particles/cells along
                                the line of sight that contributes to the given
                                window of interest defined by `restr_column_lims`.
    """
    # internally used units
    v_units = Unit('km/s')
    l_units = Unit('cm')

    zaxis = (set([0,1,2]) - set([xaxis,yaxis])).pop()
    if set([xaxis,yaxis,zaxis]) != set([0,1,2]):
        raise ValueError("x- and y-axis must be in [0,1,2] and different!")
    los = UnitQty(los, s['pos'].units, dtype=np.float64, subs=s)
    zero_Hubble_flow_at = UnitScalar(zero_Hubble_flow_at, s['pos'].units, subs=s)
    vel_extent = UnitQty(vel_extent, 'km/s', dtype=np.float64, subs=s)
    if restr_column_lims is None:
        restr_column_lims = vel_extent.copy()
    else:
        restr_column_lims = UnitQty(restr_column_lims, 'km/s', dtype=np.float64, subs=s)
    l = UnitScalar(l, 'Angstrom')
    f = float(f)
    atomwt = UnitScalar(atomwt, 'u')
    if kernel is None:
        kernel = gadget.general['kernel']

    b_0 = np.sqrt(2.0 * kB * UnitScalar('1 K') / atomwt)
    b_0.convert_to(v_units)
    s0 = q_e**2 / (4. * epsilon0 * m_e * c)
    Xsec = f * s0 * l
    Xsec = Xsec.in_units_of(l_units**2 * v_units, subs=s)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'create a mock absorption spectrum:'
        print '  at', los
        if isinstance(ion,str):
            print '  for', ion, 'at lambda =', l
        else:
            print '  at lambda =', l
        print '  with oscillator strength f =', f
        print '  => Xsec =', Xsec
        print '  and atomic weight', atomwt
        print '  => b(T=1e4K) =', b_0*np.sqrt(1e4)
        print '  using kernel "%s"' % kernel

    v_edges = UnitArr(np.linspace(vel_extent[0], vel_extent[1], Nbins+1),
                      vel_extent.units)

    # get ne number of ions per particle
    if isinstance(ion,str):
        ion = s.gas.get(ion)
    else:
        ion = UnitQty(ion, units=s['mass'].units, subs=s)
    # double precision needed in order not to overflow
    # 1 Msol / 1 u = 1.2e57, float max = 3.4e38, but double max = 1.8e308
    n = (ion.astype(np.float64) / atomwt).in_units_of(1, subs=s)
    #n = n.view(np.ndarray).astype(np.float64)

    if method != 'particles':
        # do SPH smoothing along the l.o.s.
        if spatial_extent is None:
            spatial_extent = [ np.min( s.gas['pos'][:,zaxis] ),
                               np.max( s.gas['pos'][:,zaxis] ) ]
            spatial_extent = UnitArr(spatial_extent, spatial_extent[-1].units)
            if 1.01 * s.boxsize > spatial_extent.ptp() > 0.8 * s.boxsize:
                # the box seems to be full with gas
                missing = s.boxsize - spatial_extent.ptp()
                spatial_extent[0] -= missing / 2.0
                spatial_extent[1] += missing / 2.0
            spatial_extent.convert_to(s['pos'].units, subs=s)
        else:
            spatial_extent = UnitQty( spatial_extent, s['pos'].units, subs=s )

        if spatial_res is None:
            spatial_res = UnitArr(np.percentile(s.gas['hsml'], .1),
                                  s.gas['hsml'].units)
        spatial_res = UnitScalar(spatial_res, s['pos'].units, subs=s)
        N = int(max( 1e3,
                     2.*(spatial_extent.ptp()/spatial_res).in_units_of(1,subs=s) ))

        if method == 'column':
            # do some padding in the 3D binning in order to use the the normation
            # process
            pad = int(pad)
            Npx = (1+2*pad)*np.ones(3, dtype=np.int)
            Npx[zaxis] = N
            # mask for getting the middle column of interest
            m = [pad] * 3
            m[zaxis] = slice(None)

            if col_width is None:
                col_width = spatial_res
            col_width = UnitScalar(col_width, s['pos'].units, subs=s)
            w = ((0.5+2.*pad) * col_width).in_units_of(los.units, subs=s)
            extent = UnitArr(np.empty((3,2), dtype=float), los.units)
            extent[xaxis] = [los[0]-w, los[0]+w]
            extent[yaxis] = [los[1]-w, los[1]+w]
            extent[zaxis] = spatial_extent

            binargs = {
                    'extent':   extent,
                    'Npx':      Npx,
                    'kernel':   kernel,
                    'dV':       'dV',
                    'hsml':     hsml,
                    'normed':   True,
            }

            # restrict to particles intersecting the l.o.s. column:
            sub = s.gas[ (s.gas['pos'][:,xaxis] - s.gas['hsml'] < los[0] + col_width) &
                         (s.gas['pos'][:,xaxis] + s.gas['hsml'] > los[0] - col_width) &
                         (s.gas['pos'][:,yaxis] - s.gas['hsml'] < los[1] + col_width) &
                         (s.gas['pos'][:,yaxis] + s.gas['hsml'] > los[1] - col_width) ]

            dV = sub['dV'].in_units_of(sub['pos'].units**3)

            if environment.verbose >= environment.VERBOSE_NORMAL:
                print '  using an spatial extent of:', spatial_extent
                print '  ... with %d bins of size %sx%s^2' % (N, col_width, spatial_res)

            from ..binning import SPH_to_3Dgrid
            def bin_func(s, qty, **args):
                Q, px = SPH_to_3Dgrid(sub, qty, **args)
                Q     = Q[m].reshape(N) * np.prod(px)
                return Q, px

            n_parts = n[sub._mask]
            n   , px    = bin_func(sub, n_parts/dV, **binargs)
            non0n       = (n!=0)
            vel , px    = bin_func(sub, n_parts*sub['vel'][:,zaxis]/dV,
                                   **binargs)
            vel[non0n]  = vel[non0n] / n[non0n]
            # average sqrt(T), since thats what the therm. broadening scales with
            temp, px    = bin_func(sub, n_parts*np.sqrt(sub['temp'])/dV,
                                   **binargs)
            temp[non0n] = temp[non0n] / n[non0n]
            temp      **= 2
            # we actually need the column densities, not the number of particles
            n          /= np.prod(px[[xaxis,yaxis]])

            # the z-coordinates for the Hubble flow
            los_pos     = UnitArr(np.linspace(spatial_extent[0],
                                              spatial_extent[1]-px[zaxis], N),
                                  spatial_extent.units)
        elif method == 'line':
            binargs = {
                    'los':      los,
                    'extent':   spatial_extent,
                    'Npx':      N,
                    'kernel':   kernel,
                    'dV':       'dV',
                    'hsml':     hsml,
                    'xaxis':    xaxis,
                    'yaxis':    yaxis,
            }

            # restrict to particles intersecting the l.o.s.:
            sub = s.gas[ (s.gas['pos'][:,xaxis] - s.gas['hsml'] < los[0]) &
                         (s.gas['pos'][:,xaxis] + s.gas['hsml'] > los[0]) &
                         (s.gas['pos'][:,yaxis] - s.gas['hsml'] < los[1]) &
                         (s.gas['pos'][:,yaxis] + s.gas['hsml'] > los[1]) ]

            dV = sub['dV'].in_units_of(sub['pos'].units**3)

            if environment.verbose >= environment.VERBOSE_NORMAL:
                print '  using an spatial extent of:', spatial_extent
                print '  ... with %d bins of length %s' % (N, spatial_res)

            from ..binning import SPH_3D_to_line
            bin_func = SPH_3D_to_line
            def bin_func(s, qty, **args):
                Q, px = SPH_3D_to_line(sub, qty, **args)
                Q    /= px
                Q.units = Q.units.gather()
                return Q, px

            n_parts = n[sub._mask]
            n   , px    = bin_func(sub, n_parts/dV, **binargs)
            # we actually need the column densities, not the number of particles
            n          *= px
            # for averaging, we want the integral over n_parts, not its density
            n_  , px    = bin_func(sub, n_parts, **binargs)
            non0n       = (n_!=0)
            vel , px    = bin_func(sub, n_parts*sub['vel'][:,zaxis],
                                   **binargs)
            vel[non0n]  = vel[non0n] / n_[non0n]
            # average sqrt(T), since thats what the therm. broadening scales with
            temp, px    = bin_func(sub, n_parts*np.sqrt(sub['temp']),
                                   **binargs)
            temp[non0n] = temp[non0n] / n_[non0n]
            temp      **= 2

            # the z-coordinates for the Hubble flow
            los_pos     = UnitArr(np.linspace(spatial_extent[0],
                                              spatial_extent[1]-px, N),
                                  spatial_extent.units)
        else:
            raise ValueError("Unkown method '%s'!" % method)

        n.convert_to(l_units**-2, subs=s)
        pos = None  # no use of positions in the C function
        hsml = None  # no use of smoothing lengths in the C function
        # inplace conversion possible (later conversion does not add to runtime!)
        vel.convert_to(v_units, subs=s)
        temp.convert_to('K', subs=s)
    else:
        pos = s.gas['pos'][:,(xaxis,yaxis)]
        vel = s.gas['vel'][:,zaxis]
        temp = s.gas['temp']
        if temp.base is not None:
            temp.copy()

        if isinstance(hsml, str):
            hsml = s.gas[hsml]
        elif isinstance(hsml, (Number, Unit)):
            hsml = UnitScalar(hsml,s['pos'].units) * np.ones(len(s.gas),dtype=np.float64)
        else:
            hsml = UnitQty(hsml, s['pos'].units, subs=s)
        if hsml.base is not None:
            hsml.copy()

        N = len(s.gas)

        # the z-coordinates for the Hubble flow
        los_pos = s.gas['pos'][:,zaxis]

    # add the Hubble flow
    zero_Hubble_flow_at.convert_to(los_pos.units, subs=s)
    H_flow = s.cosmology.H(s.redshift) * (los_pos - zero_Hubble_flow_at)
    H_flow.convert_to(vel.units, subs=s)
    vel = vel + H_flow

    if pos is not None:
        pos = pos.astype(np.float64).in_units_of(l_units,subs=s).view(np.ndarray).copy()
    vel  = vel.astype(np.float64).in_units_of(v_units,subs=s).view(np.ndarray).copy()
    temp = temp.in_units_of('K',subs=s).view(np.ndarray).astype(np.float64)
    if hsml is not None:
        hsml = hsml.in_units_of(l_units,subs=s).view(np.ndarray).astype(np.float64)

    los = los.in_units_of(l_units,subs=s) \
             .view(np.ndarray).astype(np.float64).copy()
    vel_extent = vel_extent.in_units_of(v_units,subs=s) \
                           .view(np.ndarray).astype(np.float64).copy()

    b_0 = float(b_0.in_units_of(v_units, subs=s))
    Xsec = float(Xsec.in_units_of(l_units**2 * v_units, subs=s))

    taus = np.empty(Nbins, dtype=np.float64)
    los_dens = np.empty(Nbins, dtype=np.float64)
    los_temp = np.empty(Nbins, dtype=np.float64)
    restr_column_lims = restr_column_lims.view(np.ndarray).astype(np.float64)
    restr_column = np.empty(N, dtype=np.float64)
    C.cpygad.absorption_spectrum(method == 'particles',
                                 C.c_size_t(N),
                                 C.c_void_p(pos.ctypes.data) if pos is not None else None,
                                 C.c_void_p(vel.ctypes.data),
                                 C.c_void_p(hsml.ctypes.data) if hsml is not None else None,
                                 C.c_void_p(n.ctypes.data),
                                 C.c_void_p(temp.ctypes.data),
                                 C.c_void_p(los.ctypes.data),
                                 C.c_void_p(vel_extent.ctypes.data),
                                 C.c_size_t(Nbins),
                                 C.c_double(b_0),
                                 C.c_double(Xsec),
                                 C.c_void_p(taus.ctypes.data),
                                 C.c_void_p(los_dens.ctypes.data),
                                 C.c_void_p(los_temp.ctypes.data),
                                 C.c_void_p(restr_column_lims.ctypes.data),
                                 C.c_void_p(restr_column.ctypes.data),
                                 C.create_string_buffer(kernel),
                                 C.c_double(s.boxsize.in_units_of(l_units))
    )

    los_dens = UnitArr(los_dens, 'cm**-2')
    los_temp = UnitArr(los_temp, 'K')
    restr_column = UnitArr(restr_column, 'cm**-2')

    if environment.verbose >= environment.VERBOSE_NORMAL:
        # if called with bad parameters sum(taus)==0 and, hence, no normation
        # possible:
        try:
            # calculate parameters
            z_edges = velocities_to_redshifts(v_edges, z0=s.redshift)
            l_edges = l * (1.0 + z_edges)
            EW_l = EW(taus, l_edges)
            extinct = np.exp(-np.asarray(taus))
            v_mean = UnitArr( np.average((v_edges[:-1]+v_edges[1:])/2.,
                                         weights=extinct), v_edges.units )
            l_mean = UnitArr( np.average((l_edges[:-1]+l_edges[1:])/2.,
                                         weights=extinct), l_edges.units )
            print 'created line with:'
            print '  EW =', EW_l
            print '  v0 =', v_mean
            print '  l0 =', l_mean
        except:
            pass

    return taus, los_dens, los_temp, v_edges, restr_column

def EW(taus, edges):
    '''
    Calculate the equivalent width of the given line / spectrum.
    
    Args:
        taus (array-like):  The optical depths in the bins.
        edges (UnitQty):    The edges of the bins. May (/should) have units.

    Returns:
        EW (float, UnitScalar):     The equivalent width in the given space (i.e.
                                    the units of the edges).
    '''
    if len(taus)+1 != len(edges):
        raise ValueError("The length of the edges does not match the length of " +
                         "the optical depths!")
    EW = np.sum( (1.0 - np.exp(-np.asarray(taus))) * (edges[1:]-edges[:-1]) )
    return EW

def velocities_to_redshifts(vs, z0=0.0):
    '''
    Convert velocities to redshifts, assuming an additional cosmological redshift.

    Note:
        The inverse is `redshifts_to_velocities`.

    Args:
        vs (UnitQty):       The velocities to convert to redshifts.
        z0 (float):         The cosmological redshift of the restframe in which
                            the velocities were measured.

    Returns:
        zs (np.ndarray):    The redshifts corresponding to the given velocities.
    '''
    vs = UnitQty(vs, 'km/s', dtype=float)
    zs = (vs / c).in_units_of(1).view(np.ndarray)
    if z0 != 0.0:   # avoid multiplication of 1.
        zs = (1.+zs)*(1.+z0) - 1.
    return zs

def redshifts_to_velocities(zs, z0=0.0):
    '''
    Convert redshifts to velocities, assuming an additional cosmological redshift.

    Note:
        This is the inverse to `velocities_to_redshifts`.

    Args:
        zs (UnitQty):       The redshifts to convert to velocities.
        z0 (float):         The cosmological redshift of the restframe in which
                            the velocities were measured.

    Returns:
        vs (np.ndarray):    The velocities corresponding to the given redshifts.
    '''
    zs = np.array(zs, dtype=float)
    if z0 != 0.0:   # avoid division by 1.
        zs = (zs+1.) / (1.+z0) - 1.
    vs = zs * c
    return zs

