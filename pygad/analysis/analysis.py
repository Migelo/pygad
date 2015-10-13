'''
A collection of functions that calculate properties of snapshots (Snap) and
sub-snapshots (e.g. MaskedSnap).
'''
__all__ = ['radial_surface_density_profile', 'x_ray_luminosity']

from units import UnitArr, dist
import physics
import numpy as np
import sph

def radial_surface_density_profile(snap, quantity='mass', r_edges=None,
                                   N_sample=50, units=None, verbose=True):
    '''
    Create a radial surface density profile.

    Args:
        snap (Snap):        The snapshot to use.
        quantity (str, UnitArr):
                            The quantity to do the radial surface density profile
                            of.
        r_edges (array-like):
                            The edges of the radial bins. If no units are set,
                            they are assumed to be those of snap.pos. (default:
                            UnitArr(np.arange(51), units='kpc'))
        units (str, Unit):  The units of the surface density.
        verbose (bool):     Whether or not to print which sample is mapped at the
                            moment.

    Returns:
        Sigma (UnitArr):   The profile, i.e. the quantity summed for the
                            intervals defined by r_edges and then divided by the
                            area.
    '''
    if r_edges is None:
        r_edges = UnitArr(np.arange(51), units='kpc')
    elif not isinstance(r_edges, UnitArr):
        r_edges = UnitArr(r_edges)
    if r_edges._units is None:
        r_edges.units = snap.pos.units

    Sigma = profile(snap, quantity, r_edges=r_edges, mode='cylindrical',
                    average='none', N_sample=N_sample, verbose=verbose)
    for i in xrange(len(r_edges)-1):
        Sigma[i] /= np.pi*(r_edges[i+1]**2-r_edges[i]**2)
    Sigma.units /= r_edges.units**2
    if units is not None:
        Sigma.convert_to(units)

    return Sigma

def x_ray_luminosity(snap, lumtable='em.dat', tempbin=None, lx0bin=None,
                     dlxbin=None):
    '''
    Calculate X-ray luminosity of gas particles using a prepared emission table
    from XSPEC.
    
    Args:
        snap (Snap):            The snapshot to use.
        lumtable (str):         The filename of the XSPEC emission table to use
                                (default: 'em.dat' in current directory)
        tempbin, lx0bin, dlxbin (array-like):
                                Temperature, Lx0 and dLx bins: Can be passed
                                instead of lumtable file (e.g. to avoid reading
                                in the same file multiple times in a loop)
    '''
    Zref = 0.4          # metallicity (in solar units) used for XSPEC calculations
    red = 0.001         # redshift assumed for XSPEC luminosity table
    Da = UnitArr(4.3*1e3, units='kpc').in_units_of('cm')   #angular diameter
                        # distance corresponding to redshift 0.001 in cm (would be
                        # better if directly calculated from redshift)

    # Read in temperature bins and corresponding Lx0(T,Z=Zref) and (dLx/dZ)(T)
    # (both in 1e44 erg/s (per Zsol))
    if tempbin == None:
        tempbin, lx0bin, dlxbin = np.loadtxt(lumtable, usecols=(0,3,5),
                                             unpack=True)
        
    tlow = tempbin[0] - 0.5*(tempbin[1]-tempbin[0]) # lower temperature bin limit
    Z = snap.gas.mtls / physics.solar.Z()           # metallicity in solar units
    mp = physics.m_p.in_units_of('g')               # proton mass
    # emission measure if gas particles (n_e * n_H * V)
    em = snap.gas.ne * snap.gas.Z[:,6].in_units_of('g')**2 * \
            snap.gas.rho.in_units_of('g/cm**3') / \
            (snap.gas.mass.in_units_of('g')*mp**2)
    # rescaling factor for precomputed luminosities
    norm = UnitArr(1e-14,units='cm**5') * em / \
            (4 * np.pi * (Da*(1+red))**2)
    lx = np.zeros(snap.gas.rho.shape[0])            # array for X-ray luminosity
    temp = snap.gas.cste*1.3806e-16/1.6022e-9       # gas temperatures in keV    
    indices = np.zeros(snap.gas.rho.shape[0])       # array for fitting tempbin
                                                    # indices for gas particles
    dtemp = np.zeros(snap.gas.rho.shape[0])+1e30    # minimal differences of gas
                                    # particle temperature and binned temperatures
    
    # loop over tempbin array to find nearest tempbin for all gas particle
    # temperatures
    for i in xrange(0,tempbin.shape[0]):
        dtemp[np.where(np.abs(temp-tempbin[i]) < dtemp)] = \
                np.abs(temp[np.where(np.abs(temp-tempbin[i])<dtemp)] - tempbin[i])
        indices[np.where(np.abs(temp-tempbin[i]) == dtemp)] = i
    
    # calculate X-ray luminosities for all gas particles
    for i in xrange(0,tempbin.shape[0]):
        lx[np.where(indices == i)] = lx0bin[i] + \
                (Z[np.where(indices==i)] - Zref) * dlxbin[i]
    lx = lx * norm * 1e44           # luminosities of all gas particles [erg/s]
    lx[np.where(temp < tlow)] = 0   # particles below threshold temperature do not
                                    # contribute to Lx
    
    return lx    
    #return UnitArr(lx, units='erg/s')

