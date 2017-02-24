'''
TODO
'''
__all__ = ['Wiersma_CoolingTable']

import numpy as np
from ..units import UnitQty, UnitArr
from .. import environment
h5py = environment.secure_get_h5py()

FULL_ELEMENT_NAME = {
        'H':    'Hydrogen',
        'He':   'Helium',
        'Mg':   'Magnesium',
        'C':    'Carbon',
        'O':    'Oxygen',
        'Fe':   'Iron',
        'Si':   'Silicon',
        'N':    'Nitrogen',
        'Ne':   'Neon',
        'S':    'Sulphur',
        'Ca':   'Calcium',
}
SHORT_ELEMENT_NAME = { v:k for k,v in FULL_ELEMENT_NAME.iteritems() }

class Wiersma_CoolingTable(object):
    '''
    Handle HDF5 cooling tables from Wiersma et al. (2008) as taken from their
    website: http://www.strw.leidenuniv.nl/WSS08/

    Args:
        path (str):     The path to the HDF5 file to load the table from.
    '''

    def __init__(self, path):
        self._path = path
        self._read()

    def _read(self, verbose=None):
        '''
        Read the tables and additional information from the HDF5 file.

        Args:
            verbose (int):          The verbosity level. Defaults to
                                    `environment.verbose`.
        '''
        if verbose is None:
            verbose = environment.verbose
        if verbose >= environment.VERBOSE_NORMAL:
            print 'reading cooling tables from "%s"' % self._path
        with h5py.File(self._path, 'r') as f:
            self._note      = f.get('Header/Note').value[0]
            self._reference = f.get('Header/Reference').value[0]
            self._version   = f.get('Header/Version').value[0]
            if verbose >= environment.VERBOSE_TALKY:
                print '  note:       "%s"' % self._note
                print '  reference:  "%s"' % self._reference
                print '  version:    "%s"' % self._version

            self._redshift = f.get('Header/Redshift').value[0]

            self._species = f.get('Header/Species_names').value
            if verbose >= environment.VERBOSE_TALKY:
                print '  %2d species: %s' % (len(self._species),
                                            ', '.join(self._species))

            if verbose >= environment.VERBOSE_TALKY:
                print '  reading cooling tables for'
                print '    metal free gas & electron density...'
            tbls = {
                    'noZ': f.get('Metal_free/Net_Cooling').value,
                    'ne_nH': f.get('Metal_free/Electron_density_over_n_h').value,
            }
            f.get('Metal_free/Mean_particle_mass').value
            for el in self._species:
                if verbose >= environment.VERBOSE_TALKY:
                    print '    %s...' % el
                tbls[el] = f.get('%s/Net_Cooling' % el).value
            if verbose >= environment.VERBOSE_TALKY:
                print '  reading table bins'
            self._nH_bins = f.get('Metal_free/Hydrogen_density_bins').value
            self._T_bins = f.get('Metal_free/Temperature_bins').value
            self._fHe_bins = f.get('Metal_free/Helium_mass_fraction_bins').value
            if verbose >= environment.VERBOSE_TALKY:
                print '    nH [cm^-3]: ', self.nH_range
                print '    T [K]:      ', self.T_range
                print '    fHe:        ', self.fHe_range

            if verbose >= environment.VERBOSE_TALKY:
                print '  reading solar abundancies'
            self._solar_species = f.get('Header/Abundances/Abund_names').value
            self._solar_mass_frac = \
                    f.get('Header/Abundances/Solar_mass_fractions').value
            self._solar_ne_nH = f.get('Solar/Electron_density_over_n_h').value

        self._tbls = tbls

        def interp(x, y, z, kind='linear'):
            from scipy.interpolate import interp2d
            return interp2d(x, y, z.T, kind=kind, copy=True, bounds_error=True)
        self._metals_interp = {
                metal: interp(self._T_bins, self._nH_bins, self._tbls[metal])
                for metal in self._species
        }
        self._noZ_interp = [
                interp(self._T_bins, self._nH_bins, self._tbls['noZ'][ifHe])
                for ifHe in xrange(len(self._fHe_bins))
        ]
        self._ne_nH_interp = [
                interp(self._T_bins, self._nH_bins, self._tbls['ne_nH'][ifHe])
                for ifHe in xrange(len(self._fHe_bins))
        ]
        self._solar_ne_nH_interp = \
                interp(self._T_bins, self._nH_bins, self._solar_ne_nH)

    @property
    def note(self):
        return self._note

    @property
    def version(self):
        return self._version

    @property
    def reference(self):
        return self._reference

    @property
    def redshift(self):
        return self._redshift

    @property
    def species(self):
        return tuple(self._species)

    @property
    def nH_range(self):
        return tuple(self._nH_bins[(0,-1),])

    @property
    def T_range(self):
        return tuple(self._T_bins[(0,-1),])

    @property
    def fHe_range(self):
        return tuple(self._fHe_bins[(0,-1),])

    def _interpolate_fHe(self, interpolations, T, nH, fHe):
        '''
        For either `self._noZ_interp` or `self._ne_nH_interp`.
        '''
        assert interpolations is self._noZ_interp or \
               interpolations is self._ne_nH_interp
        from numbers import Number
        if isinstance(fHe, Number) or fHe.shape==tuple() or len(fHe)==1:
            fHe = fHe * np.ones(len(T), dtype=float)
        fHe = np.array( fHe ).astype(float)
        assert T.shape == nH.shape == fHe.shape

        ifHe = np.searchsorted(self._fHe_bins, fHe)-1
        if np.any(fHe<self._fHe_bins[ifHe]) or \
                np.any(ifHe>=len(self._fHe_bins)-1):
            raise IndexError('m_He/m_H (fHe) out of range')
        iTnH = zip(ifHe,T,nH)
        Qs_1 = np.array( [
                interpolations[i](t,n) for i,t,n in iTnH
        ] ).reshape(len(T))
        iTnH = zip(ifHe+1,T,nH)
        Qs_2 = np.array( [
                interpolations[i](t,n) for i,t,n in iTnH
        ] ).reshape(len(T))
        alpha = (fHe - self._fHe_bins[ifHe]) \
                / (self._fHe_bins[ifHe+1] - self._fHe_bins[ifHe])
        return (1.-alpha) * Qs_1 + alpha * Qs_2

    def get_cooling_for_species(self, species, T, nH, fHe=0.25):
        '''
        Get the cooling rate(s) of a given species / element in units of
        [erg cm^3 s^-1]: Lambda / n_H^2

        Note:
            These cooling rates do take the heating from the UV/X-ray background
            into account!

        Args:
            species (str):      The element to get the cooling rate(s) for. It can
                                either be one of the species listed in
                                `self.species` or 'noZ'/'metal-free'/'H+He'/'HHe'
                                for the cooling of Hydrogen and Helium only.
            T (UnitQty):        The temperature of the gas.
            nH (UnitQty):       The Hydrogen number density
            fHe (float):        The mass fraction of Helium. Only used if it is
                                asked for the cooling rate(s) for Hydrogen +
                                Helium.

        Returns:
            Lambda (UnitArr):   The cooling rate(s) for the given species in units
                                of [erg cm**3 s^-1].
        '''
        T  = UnitQty(T,'K').view(np.ndarray)
        nH = UnitQty(nH,'cm**-3').view(np.ndarray)

        if species in FULL_ELEMENT_NAME:
            species = FULL_ELEMENT_NAME[species]
        if species in SHORT_ELEMENT_NAME:
            if len(T) == len(nH) and len(T) > 1:
                Lambda = np.array( [
                    self._metals_interp[species](t,n) for t,n in zip(T,nH)
                ] ).reshape(len(T))
            else:
                Lambda = self._metals_interp[species](T,nH)
        elif species in ['noZ', 'metal-free', 'H+He', 'HHe']:
            # additional interpolation for HHe is needed
            Lambda = self._interpolate_fHe(self._noZ_interp, T, nH, fHe)
        else:
            raise ValueError("Unknown species '%s'" % species)

        return UnitArr(Lambda, 'erg cm**3 s**-1')

    def get_cooling(self, s, Compton_cooling=True, verbose=None):
        '''
        Calculate the cooling rates for all gas elements in a given
        (sub-)snapshot: Lambda / n_H^2 [erg cm^3 s^-1]

        Note:
            These cooling rates do take the heating from the UV/X-ray background
            into account!

        Args:
            s (Snap):               The snapshot for whichs gas to calculate the
                                    cooling rates.
            Compton_cooling (bool): Whether to also include the Compton cooling
                                    from the cosmic microwave background (CMB).
            verbose (int):          The verbosity level. Defaults to
                                    `environment.verbose`.

        Returns:
            Lambda (UnitArr):       The cooling rates (Lambda/n_H**2) for each gas
                                    particle in the (sub-)snapshot `s`.
        '''
        if verbose is None:
            verbose = environment.verbose
        if abs(s.redshift - self.redshift) > 0.01:
            raise RuntimeError("The snapshot's redshift (z=%0.2f) " % s.redshift +
                               "does not match the one of the tables " +
                               "(z=%0.2f)!" % self._redshift)
        g = s.gas if (len(s.gas)!=len(s) or len(s)==0) else s
        if verbose >= environment.VERBOSE_NORMAL:
            print 'calculate cooling rates for\n  %s' % g

        T = g['temp'].in_units_of('K',subs=s).view(np.ndarray)
        nH = (g['rho'] * g['H']/g['mass']).in_units_of('u/cm**3',subs=s).view(np.ndarray)
        fHe = ( g['He'] / (g['He'] + g['H']) ).in_units_of(1,subs=s).view(np.ndarray)

        T[T<self.T_range[0]] = self.T_range[0]
        T[T>self.T_range[1]] = self.T_range[1]
        nH[nH<self.nH_range[0]] = self.nH_range[0]
        nH[nH>self.nH_range[1]] = self.nH_range[1]
        fHe[fHe<self.fHe_range[0]] = self.fHe_range[0]
        fHe[fHe>self.fHe_range[1]] = self.fHe_range[1]

        if verbose >= environment.VERBOSE_NORMAL:
            print 'interpolate cooling rates for...'
        metal_cool = np.zeros(len(g), dtype=float)
        for species in self._species:
            try:
                if verbose >= environment.VERBOSE_NORMAL:
                    print '  %s...' % species
                el = SHORT_ELEMENT_NAME[species]
                solar_abund = \
                        self._solar_mass_frac[np.where(self._solar_species==species)][0]
                el_cool = self.get_cooling_for_species(el,T,nH,fHe) \
                    * (g[el]/g['mass']) / solar_abund
                metal_cool += el_cool
            except ValueError as e:
                print "%s: %s" % (type(e).__name__, e.message)
        if verbose >= environment.VERBOSE_NORMAL:
            print '  H+He...'
        hhe_cool = self.get_cooling_for_species('HHe',T,nH,fHe)

        if verbose >= environment.VERBOSE_NORMAL:
            print 'interpolate ne/nH and solar metallicity ne/nH...'
        hhe_ne = self._interpolate_fHe(self._ne_nH_interp, T, nH, fHe)
        solar_ne = np.array( [
                    self._solar_ne_nH_interp(t,n) for t,n in zip(T,nH)
        ] ).reshape(len(T))
        
        if verbose >= environment.VERBOSE_NORMAL:
            print 'combine to netto cooling rates...'
        net_cool = UnitArr( hhe_cool + (hhe_ne/solar_ne) * metal_cool,
                            'erg cm**3 s**-1')

        # add Compton cooling of the CMB
        if Compton_cooling: # and self._redshift > 0:
            if verbose >= environment.VERBOSE_NORMAL:
                print 'add the Compton cooling rates...'

            import quantities
            Stefan = quantities.constants['Stefan-Boltzmann constant']
            c = quantities.c
            m_e = quantities.m_e
            Thompson = quantities.constants['Thomson cross section']
            kB = quantities.kB

            T_CMB = UnitArr('2.728 K') * (1.0 + self._redshift)  # CMB temperature
            # note: nH was made unitless (for speed); [hhe_]ne is unitless anyway
            Compton_cool = -(16.0 * Stefan * Thompson * (T_CMB**4) / (m_e * c**2)) \
                                * kB * (T_CMB - T) * hhe_ne / UnitArr(nH,'cm**-3')
            Compton_cool.convert_to('erg cm**3 s**-1')
            #print 'Compton cooling [0%,25%,100%]-perc.:', \
            #        np.percentile(Compton_cool, [0,50,100] ), Compton_cool.units
            #print 'rel. Compton cooling [0%,25%,100%]-perc.:', \
            #        np.percentile((Compton_cool/net_cool).in_units_of(1), [0,50,100] )

            net_cool += Compton_cool

        return net_cool

