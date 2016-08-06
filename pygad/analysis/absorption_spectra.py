"""
Produce mock absorption spectra for given line transition(s) and line-of-sight(s).
"""
__all__ = ['mock_absorption_spectrum_of', 'mock_absorption_spectrum',
           'EW', 'velocities_to_redshifts']

from ..units import Unit, UnitArr, UnitQty, UnitScalar
from ..physics import kB, m_H, c, q_e, m_e, epsilon0
from ..kernels import *
from .. import gadget
from .. import C
from .. import environment
import numpy as np

lines = {
    'H1215':     {'ion':'HI',     'l':'1215.6701 Angstrom', 'f':0.4164, 'atomwt':m_H},
    'HeII':      {'ion':'HeII',   'l': '303.918 Angstrom',  'f':0.4173, 'atomwt': '3.971 u'},
    'CIII977':   {'ion':'CIII',   'l': '977.020 Angstrom',  'f':0.7620, 'atomwt':'12.011 u'},
    'CIV1548':   {'ion':'CIV',    'l':'1548.195 Angstrom',  'f':0.1908, 'atomwt':'12.011 u'},
    'OIV787':    {'ion':'OIV',    'l': '787.711 Angstrom',  'f':0.110,  'atomwt':'15.9994 u'},
    'OVI1031':   {'ion':'OVI',    'l':'1031.927 Angstrom',  'f':0.1329, 'atomwt':'15.9994 u'},
    'NeVIII770': {'ion':'NeVIII', 'l': '770.409 Angstrom',  'f':0.103,  'atomwt':'20.180 u'},
    'MgII2796':  {'ion':'MgII',   'l':'2796.352 Angstrom',  'f':0.6123, 'atomwt':'24.305 u'},
    'SiIV1393':  {'ion':'SiIV',   'l':'1393.755 Angstrom',  'f':0.5280, 'atomwt':'28.086 u'},
}

def mock_absorption_spectrum_of(s, los, vel_extent, line,
                                spatial_bins=False, spatial_extent=None,
                                Nbins=1000, hsml='hsml', kernel=None,
                                zero_Hubble_flow_at=0,
                                xaxis=0, yaxis=1, **kwargs):
    '''
    Create a mock absorption spectrum for the given line of sight (l.o.s.) for the
    given line transition.

    This function basically just calls `mock_absorption_spectrum` for the given
    line.

    Args:
        s (Snap):               The snapshot to shoot the l.o.s. though.
        los (UnitQty):          The position of the l.o.s.. By default understood
                                as in units of s['pos'], if not explicitly
                                specified.
        vel_extent (UnitQty):   The limits of the spectrum in (rest frame)
                                velocity space. Units default to 'km/s'.
        line (str,dict):        Either a name of a line as defined in
                                `absorption_spectra.lines` or a custom dictionary
                                of the same kind, which values are than passed to
                                `mock_absorption_spectrum`.
        spatial_bins (bool, int):
                                If True, do not bin the particles directly to
                                velocity space, but first onto a spatial grid
                                along the l.o.s. and then to velocity space. If
                                this is an integer, it specifies the number of
                                these spatial bins.
        spatial_extent (UnitQty):
                                The extent in the spatial bins along the l.o.s..
        Nbins (int):            The number of bins for the spectrum.
        hsml (str, UnitQty, Unit):
                                The smoothing lengths to use. Can be a block name,
                                a block itself or a Unit that is taken as constant
                                volume for all particles.
        kernel (str):           The kernel to use for smoothing. (By default use
                                the kernel defined in `gadget.cfg`.)
        zero_Hubble_flow_at (UnitScalar):
                                The position along the l.o.s. where there is no
                                Hubble flow. If not units are given, they are
                                assume to be those of s['pos'].
        xaxis/yaxis (int):      The x- and y-axis for the l.o.s.. The implicitly
                                defined z-axis goes along the l.o.s.. The axis
                                must be chosen from [0,1,2].

    Returns:
        taus (np.ndarray):      The optical depths for the velocity bins.
        los_dens (np.ndarray):  The column densities restricted to the velocity bins.
        los_temp (np.ndarray):  The (mass-weighted) particle temperatures
                                restricted to the velocity bins.
        v_edges (UnitArr):      The velocities at the bin edges.
    '''
    if isinstance(line,str):
        line = lines[line]
    return mock_absorption_spectrum(s, los, vel_extent,
                                    line['ion'],
                                    l=line['l'], f=line['f'],
                                    atomwt=line['atomwt'],
                                    spatial_bins=spatial_bins,
                                    spatial_extent=spatial_extent,
                                    Nbins=Nbins, hsml=hsml, kernel=kernel,
                                    zero_Hubble_flow_at=zero_Hubble_flow_at,
                                    xaxis=xaxis, yaxis=yaxis, **kwargs)

def mock_absorption_spectrum(s, los, vel_extent, ion, l, f, atomwt,
                             spatial_bins=False, spatial_extent=None,
                             Nbins=1000, hsml='hsml', kernel=None,
                             zero_Hubble_flow_at=0,
                             xaxis=0, yaxis=1):
    """
    Create a mock absorption spectrum for the given line of sight (l.o.s.) for the
    given line transition.

    Credits to Neal Katz and Romeel Dave, who wrote a code taken as a basis for
    this one, first called specexbin and later specexsnap that did the same (with
    the spatial bins), and who helped me with the gist of this one.
    
    Args:
        s (Snap):               The snapshot to shoot the l.o.s. though.
        los (UnitQty):          The position of the l.o.s.. By default understood
                                as in units of s['pos'], if not explicitly
                                specified.
        vel_extent (UnitQty):   The limits of the spectrum in (rest frame)
                                velocity space. Units default to 'km/s'.
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
        spatial_bins (bool, int):
                                If True, do not bin the particles directly to
                                velocity space, but first onto a spatial grid
                                along the l.o.s. and then to velocity space. If
                                this is an integer, it specifies the number of
                                these spatial bins.
        spatial_extent (UnitQty):
                                The extent in the spatial bins along the l.o.s..
        Nbins (int):            The number of bins for the spectrum.
        hsml (str, UnitQty, Unit):
                                The smoothing lengths to use. Can be a block name,
                                a block itself or a Unit that is taken as constant
                                volume for all particles.
        kernel (str):           The kernel to use for smoothing. (By default use
                                the kernel defined in `gadget.cfg`.)
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
    """
    # internally used units
    v_units = Unit('km/s')
    l_units = Unit('cm')

    los = UnitQty(los, s['pos'].units, dtype=float)
    vel_extent = UnitQty(vel_extent, 'km/s', dtype=float)
    l = UnitScalar(l, 'Angstrom')
    f = float(f)
    atomwt = UnitScalar(atomwt, 'u')
    zaxis = (set([0,1,2]) - set([xaxis,yaxis])).pop()
    if set([xaxis,yaxis,zaxis]) != set([0,1,2]):
        raise ValueError("x- and y-axis must be in [0,1,2] and different!")
    if kernel is None:
        kernel = gadget.general['kernel']
    zero_Hubble_flow_at = UnitScalar(zero_Hubble_flow_at, s['pos'].units, subs=s)

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
        print '  and atomic weight', atomwt
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
    n = n.view(np.ndarray).astype(np.float64)

    if spatial_bins:
        # do SPH smoothing along the l.o.s.
        from ..binning import SPH_to_3Dgrid
        los = los.in_units_of(s['pos'].units, subs=s)
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
        if spatial_bins is True:
            N = int(max( 1e3,
                         (spatial_extent.ptp()/UnitArr('1 kpc')).in_units_of(1,subs=s) ))
        else:
            N = int( spatial_bins )
        Npx = np.ones(3, dtype=int)
        Npx[zaxis] = N
        w = (spatial_extent.ptp() / N / 2.0).in_units_of(los.units, subs=s)
        extent = UnitArr(np.empty((3,2), dtype=float), los.units)
        extent[xaxis] = [los[0]-w, los[0]+w]
        extent[yaxis] = [los[1]-w, los[1]+w]
        extent[zaxis] = spatial_extent
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print '  using an spatial extent of:', spatial_extent
            print '  ... with %d bins of size %s^3' % (N, 2.*w)
        # restrict to particles intersecting the l.o.s.:
        sub = s.gas[ (s.gas['pos'][:,xaxis] - s.gas['hsml'] < los[0]) &
                     (s.gas['pos'][:,xaxis] + s.gas['hsml'] > los[0]) &
                     (s.gas['pos'][:,yaxis] - s.gas['hsml'] < los[1]) &
                     (s.gas['pos'][:,yaxis] + s.gas['hsml'] > los[1]) ]
        dV = sub['dV'].in_units_of(sub['pos'].units**3)
        gridargs = {
                'extent': extent,
                'Npx': Npx,
                'kernel': kernel,
                'dV': dV,
                'hsml': hsml,
                'normed': False,
        }
        n_parts = n[sub._mask]
        n   , px    = SPH_to_3Dgrid(sub, n_parts/dV, **gridargs)
        n           = n.reshape(N) * np.prod(px)
        non0n       = (n!=0)
        vel , px    = SPH_to_3Dgrid(sub, n_parts*sub['vel'][:,zaxis]/dV, **gridargs)
        vel         = vel.reshape(N) * np.prod(px)
        vel[non0n]  = vel[non0n] / n[non0n]
        # average sqrt(T), since thats what the therm. broadening scales with
        temp, px    = SPH_to_3Dgrid(sub, n_parts*np.sqrt(sub['temp'])/dV, **gridargs)
        temp        = temp.reshape(N) * np.prod(px)
        temp[non0n] = temp[non0n] / n[non0n]
        temp      **= 2
        # `hsml` here is the pixel size, needed for calculating column densities
        hsml        = px[(xaxis,yaxis),]
        pos         = None  # no use of positions in the C function
        # the z-coordinates for the Hubble flow
        los_pos     = UnitArr(np.linspace(extent[zaxis][0],
                                  extent[zaxis][1]-px[zaxis], Npx[zaxis]),
                      extent.units)

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
    C.cpygad.absorption_spectrum(not bool(spatial_bins),
                                 C.c_size_t(N),
                                 C.c_void_p(pos.ctypes.data) if pos is not None else None,
                                 C.c_void_p(vel.ctypes.data),
                                 C.c_void_p(hsml.ctypes.data),
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
                                 C.create_string_buffer(kernel),
                                 C.c_double(s.boxsize.in_units_of(l_units))
    )

    los_dens = UnitArr(los_dens, 'cm**-2')
    los_temp = UnitArr(los_temp, 'K')

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

    return taus, los_dens, los_temp, v_edges

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

def velocities_to_redshifts(v_edges, z0=0.0):
    '''
    Convert velocities to redshifts, assuming an additional cosmological redshift.
    '''
    z_edges = (v_edges / c).in_units_of(1).view(np.ndarray)
    if z0 != 0.0:   # avoid multiplication of 1.
        z_edges = (1.+z_edges)*(1.+z0) - 1.
    return z_edges

