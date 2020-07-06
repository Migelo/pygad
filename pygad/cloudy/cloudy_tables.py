'''
Read and interpolate Cloudy tables for ionisation states as a function of density
and temperature for a given redshift.
'''
__all__ = ['config_ion_table', 'IonisationTable']

from .. import environment
from ..units import Unit, UnitQty
from .. import gadget
from .. import physics
import numpy as np
import copy
import re
import os

def config_ion_table(redshift):
    '''
    Load & interpolate an IonisationTable for the given redshift as defined in the
    config file `derived.cfg`.

    Args:
        redshift (float):   The redshift to load/interpolate the table(s) for.

    Returns:
        iontbl (IonisationTable):   The ionisation table class.
    '''
    from ..snapshot.derived import iontable
    if iontable['tabledir'] is None:
        raise RuntimeError('No Cloudy table directory defined in `derived.cfg`!')

    #for old style Oppenheimer tables:
    nH_vals = iontable['nH_vals']
    nH_vals = np.linspace(float(nH_vals[0]),
                          float(nH_vals[0] + nH_vals[1]*(nH_vals[2]-1)),
                          nH_vals[2])
    T_vals = iontable['T_vals']
    T_vals = np.linspace(float(T_vals[0]),
                         float(T_vals[0] + T_vals[1]*(T_vals[2]-1)),
                         T_vals[2])

    iontbl = IonisationTable(redshift,
                             tabledir=iontable['tabledir'],
                             table_pattern=iontable['pattern'],
                             flux_factor=iontable['flux_factor'],
                             style=iontable['style'],
                             nH=nH_vals,
                             T=T_vals,
                             ions=iontable.get('ions',[]))
    return iontbl

class IonisationTable(object):
    '''
    Read in a Cloudy table for fractions of ions and prepare it for interpolation.

    Note:
        For debugging purposes, there is a method to plot the tables read in:
        `show_table`.

    Args:
        redshift (float):       The redshift to create the table for. The two
                                tables inbetween which the redshift is are loaded
                                and linearly interpolated.
        tabledir (str):         The directory where all the tables are located.
        table_pattern (str):    The pattern of the table file names. It is their
                                filename with '<z>' where the redshift is encoded.
                                This code is 10*z rounded to the next integer.
        flux_factor (float):    Adjust the UVB by this factor (assume a optically
                                thin limit and scale down the densities during the
                                lookup by this factor).
                                (Note: `sigHI` used in the Rahmati fitting formula
                                needed for an approximation of self-shielding is
                                not adjusted!)
        style (str):            Either 'Oppenheimer old' or 'Oppenheimer new'.
        nH (array-like):        The Hydrogen number densities of the grid points
                                in units of log10(cm^-3).
        T (array-like):         The temperatures for the grid points in units of
                                log10(K).
        ions (iterable):        The names of the ions in the table.
    '''

    def __init__(self, redshift, tabledir,
                 table_pattern='lt<z>f100_i45', flux_factor=1.0,
                 style='Oppenheimer new', nH=None, T=None, ions=None):
        if style not in ['Oppenheimer old', 'Oppenheimer new']:
            raise ValueError('Unknown style "%s"!' % style)
        self._style = style

        if style == 'Oppenheimer old':
            nH = np.asarray(nH)
            T = np.asarray(T)
            if nH.ndim!=1 or T.ndim!=1:
                raise ValueError('`nH` and `T` need 1-dim., see docs!')
            self._nH_vals = nH.copy()
            self._T_vals = T.copy()
            self._ions = list(ions)
        else:
            self._nH_vals = None
            self._T_vals = None
            self._ions = None
        if table_pattern.count('<z>') != 1:
            raise ValueError('The table pattern has to have exactly one ' +
                             'occurrence of "<z>"!')

        self._redshift = float(redshift)
        self._tabledir = tabledir
        self._pattern = table_pattern
        self._table = self._get_table()
        self.flux_factor = flux_factor

    @property
    def redshift(self):
        return self._redshift
    @property
    def tabledir(self):
        return self._tabledir
    @property
    def table_pattern(self):
        return self._pattern
    @property
    def nH_vals(self):
        return self._nH_vals
    @property
    def T_vals(self):
        return self._T_vals
    @property
    def ions(self):
        return self._ions
    @property
    def data(self):
        return self._table
    @property
    def flux_factor(self):
        return self._flux_factor
    @flux_factor.setter
    def flux_factor(self, value):
        if float(value) <= 0.0:
            raise ValueError('The flux factor needs to be positive!')
        self._flux_factor = float(value)

    def interp_snap(self, ion, s, nH='nH', T='temp', selfshield=False,
                    warn_outofbounds=True):
        '''
        Convenience function: calling `interp_parts(ion, s.get(nH), s.get(T),
        selfshield, subs=s, warn_outofbounds=warn_outofbounds)`.
        '''
        if abs(s.redshift - self._redshift) > 0.01:
            import sys
            print('WARNING: the snapshot\'s redshift ' + \
                                 '(%.3f) does not match the ' % s.redshift + \
                                 'table\'s redshift (%.3f)!' % self._redshift, file=sys.stderr)
        return self.interp_parts(ion, s.get(nH), s.get(T), selfshield, subs=s,
                                 warn_outofbounds=warn_outofbounds)

    def interp_parts(self, ion, nH, T, selfshield=True, UVB=gadget.general['UVB'],
                     z=None, fbaryon=None, subs=None, warn_outofbounds=True):
        '''
        Given nH and temperatures for a list of particles, get the fractions for
        a given ion by a bilinear interpolation of the table for each particle.

        If values out of the table bounds are encountered, the corresponding
        values will be the nearest table values.

        Args:
            ion (int,str):      The index of name of the ion in question.
            nH (UnitQty):       The Hydrogen number densities of the particles (if
                                given unitless, interpreted as in cm^-3).
            T (UnitQty):        The temperatures of the particles (if given
                                unitless, interpreted as in K).
            selfshield (bool):  Whether to account for self-shielding by assuming
                                the attenuation of the UVB as experienced by HI
                                using the Rahmati+ (2013) prescription (formula
                                (14) of the paper). Note that this is just a rough
                                approximation!
            UVB (str):          The UVB background used for the self-shielding. If
                                no self-shielding is used, the argument is ignored.
            z (float):          The redshift used for the self-shielding. If None
                                and subs is a (sub-)snapshot, it is taken from the
                                snapshot. If no self-shielding is used, the
                                argument is ignored.
            fbaryon (float):    The cosmic baryon density used for the
                                self-shielding. If None and subs is a
                                (sub-)snapshot, it is taken from the snapshot's
                                cosmology. If no self-shielding is used, the
                                argument is ignored.
            subs (dict, Snap):  Used for substitutions that might be neccessary
                                when converting units (for more info see
                                `UnitArr.in_units_of`).
            warn_outofbounds (bool):
                                Whether to print out a warning if particles are
                                out of the bounds of the ionisation tables.

        Returns:
            f (np.ndarray):     The fractions of the ion for each particle as read
                                / interpolated from the table.
        '''
        if len(nH) != len(T):
            raise ValueError('Lengths of nH (%d) and T (%d) do not match!' % (
                                len(nH),len(T)))
        if isinstance(ion,str):
            ion = self._ions.index(str(ion))
        if ion<0 or self._table.shape[-1]<=ion:
            raise ValueError('Ion index (%d) out of bounds!' % ion)
        # units of table
        nH = UnitQty(nH, 'cm**-3', subs=subs)
        T  = UnitQty(T , 'K',      subs=subs)

        if selfshield:
            # with Rahmati+ (2013) description
            from ..snapshot.snapshot import Snapshot
            from ..cloudy import Rahmati_fGamma_HI
            if isinstance(subs,Snapshot):
                if z is None:
                    z = subs.redshift
                if fbaryon is None:
                    fbaryon = subs.cosmology.Omega_b / subs.cosmology.Omega_m
            else:
                if z is None:
                    raise ValueError('Redshift is needed for self-shielding!')
                if fbaryon is None:
                    # just some reasonable value
                    cosmo = physics.Planck2013()
                    fbaryon = cosmo.Omega_b / cosmo.Omega_m
                    import sys
                    print('WARNING: fbaryon was not given, using', fbaryon, file=sys.stderr)
            fG = Rahmati_fGamma_HI(z, nH, T, fbaryon, UVB=UVB,
                                   flux_factor=self.flux_factor, subs=subs)
            # limit the attenuation in very dense gas (assuming not all of the
            # particle mass actually is at these high densities)
            fG_limit = 1e-3
            fG[ fG < fG_limit ] = fG_limit
            # decreasing the radiation by a factor of fG corresponds to increasing
            # the density by the same factor fG:
            nH /= fG

        # assume the optical thin limit (as done for the self-shielding, too)
        if self._flux_factor != 1.0:
            nH /= self._flux_factor

        # look-up tables use log10-values
        nH = np.log10( nH )
        T  = np.log10( T  )

        # find lower indices
        N_nH = len(self._nH_vals)
        N_T  = len(self._T_vals)
        k_nH = -np.ones(len(nH), dtype=int)
        for k in range( N_nH ):
            k_nH[ self._nH_vals[k] < nH ] = k
        l_T = -np.ones(len(T), dtype=int)
        for l in range( N_T ):
            l_T[ self._T_vals[l] < T ] = l

        # do bilinear interpolation

        low_nH  = (k_nH<0)
        high_nH = (k_nH>=N_nH-1)
        low_T   = (l_T <0)
        high_T  = (l_T >=N_T -1)
        if warn_outofbounds:
            out_of_bounds = np.sum( low_nH | high_nH | low_T | high_T )
            if out_of_bounds:
                import sys
                from .. import utils
                print('WARNING: %s particles out of bounds!' % \
                        utils.nice_big_num_str(out_of_bounds), file=sys.stderr)
        # avoid invalid indices in the following (masking would slow down and be
        # not very readable):
        k_nH[ low_nH  ] = 0
        k_nH[ high_nH ] = N_nH-2
        l_T [ low_T   ] = 0
        l_T [ high_T  ] = N_T-2

        tbl = self._table[:,:,ion].ravel()
        f = np.empty(len(T), dtype=float)   # allocate memory

        f11 = tbl[  l_T    + (k_nH  ) * N_T]
        f21 = tbl[  l_T    + (k_nH+1) * N_T]
        f12 = tbl[ (l_T+1) + (k_nH  ) * N_T]
        f22 = tbl[ (l_T+1) + (k_nH+1) * N_T]
        # interpolate in nH
        nH1, nH2 = self._nH_vals[k_nH], self._nH_vals[k_nH+1]
        a = (nH - nH1) / (nH2 - nH1)
        f1 = (1.-a) * f11 + a * f21
        f2 = (1.-a) * f12 + a * f22
        f1[low_nH]  = f11[low_nH]
        f1[high_nH] = f21[high_nH]
        f2[low_nH]  = f12[low_nH]
        f2[high_nH] = f22[high_nH]
        # interpolate in T
        T1, T2 = self._T_vals[l_T], self._T_vals[l_T+1]
        a = (T - T1) / (T2 - T1)
        f = (1.-a) * f1 + a * f2
        f[low_T]  = f1[low_T]
        f[high_T] = f2[high_T]

        return f

    def _read_cloudy_table(self, filename):
        if self._style == 'Oppenheimer old':
            tbl = np.loadtxt(filename)
            if len(self._nH_vals) * len(self._T_vals) != tbl.shape[0]:
                raise ValueError("table size (%d) " % tbl.shape[0] +
                                 "does not match the grid given " +
                                 "(%dx%d)!" % (len(self._nH_vals),len(self._T_vals)))
            if len(self._ions) != tbl.shape[1]:
                raise ValueError("table column size (%d) " % tbl.shape[1] +
                                 "does not match the number of ions " +
                                 "(%d)!" % len(self._ions))
            tbl = tbl.reshape( (len(self._nH_vals), len(self._T_vals),
                                tbl.shape[1]) )
            redshift    = None
            nH_vals     = self._nH_vals
            T_vals      = self._T_vals
            short_ions  = self._ions
        elif self._style == 'Oppenheimer new':
            tbl = np.loadtxt(filename, skiprows=1)
            with open(filename, 'r') as f:
                head = f.readline().strip().strip('#')
            head = head.split()
            if len(head)-2!=tbl.shape[1] or head[-2]!='redshift=' or \
                    head[:2]!=['Hdens', 'Temp']:
                raise ValueError('Head of table "%s" of unexpected format!' % filename)

            redshift = float(head[-1])
            ions = head[2:-2]
            short_ions = []
            for ion in ions:
                i = [n for n,c in enumerate(ion) if c.isupper()][1]
                abb_el, state = ion[:i], ion[i:]
                # element name might be abbreviated, if ion name is to long otherwise
                el = None
                for full_el, short in physics.cooling.SHORT_ELEMENT_NAME.items():
                    if full_el[:len(abb_el)] == abb_el:
                        el = short
                        break
                if el is None:
                    raise RuntimeError('Short name of "%s[...]" not found!' % abb_el)
                short_ions.append(el+' '+state)

            try:    # numpy version >=1.9 required
                nH_vals, cnts = np.unique(tbl[:,0], return_counts=True)
                if np.any( cnts != cnts[0]) or np.any(tbl[:cnts[0],0]!=tbl[0,0]):
                    raise RuntimeError('Table "%s" of unexpected format!' % filename)
                T_vals,  cnts = np.unique(tbl[:,1], return_counts=True)
                if np.any(cnts!=cnts[0]) or cnts[0]!=len(nH_vals) or len(cnts)!=len(T_vals):
                    raise RuntimeError('Table "%s" of unexpected format!' % filename)
            except RuntimeError as rt:
                raise rt
            except:
                nH_vals = np.unique(tbl[:,0])
                T_vals  = np.unique(tbl[:,1])
            tbl = tbl[:,2:].reshape((len(nH_vals),len(T_vals),len(ions)))
        return redshift, nH_vals, T_vals, short_ions, tbl

    def _get_table(self):
        # find all the available tables
        pattern = re.compile( self._pattern.replace('<z>',r'(\d+)') + '$' )
        tables = {}
        for filename in os.listdir(self._tabledir):
            match = re.match(pattern, filename)
            if match:
                z = float(match.group(1)[0]+'.'+match.group(1)[1:])
                tables[z] = os.path.join(self._tabledir,filename)
        # check if any table was found
        if len(tables) == 0:
            raise RuntimeError('No Cloudy tables in the given directory ' +
                               '("%s")!' % self._tabledir)
        # get the two closest tables and interpolate between
        avail_z = np.array( sorted(tables.keys()) )
        i = np.argmin( np.abs( avail_z - self._redshift ) )
        z = avail_z[i]
        if (i==0 and self._redshift<=z) or \
                (i==len(avail_z)-1 and z<=self._redshift):
            # redshift at / over the the limits of the tables
            z1, z2 = avail_z[i], avail_z[i]
        elif z < self._redshift:
            z1, z2 = avail_z[i], avail_z[i+1]
        else:
            z1, z2 = avail_z[i-1], avail_z[i]
        if z1==z2 or z1==self._redshift or z2==self._redshift:
            z = z2 if z2 == self._redshift else z1
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print('load table:')
                print('  "%s" (z=%.3f)' % (tables[z], z))
            redshift, nH_vals, T_vals, ions, table = \
                    self._read_cloudy_table(tables[z])
            if self._style == 'Oppenheimer new':
                assert np.isclose(redshift, z)
                self._nH_vals = nH_vals
                self._T_vals = T_vals
                self._ions = ions
        else:
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print('load tables:')
                print('  "%s" (z=%.3f)' % (tables[z1], z1))
                print('  "%s" (z=%.3f)' % (tables[z2], z2))
            redshift1, nH_vals1, T_vals1, ions1, table1 = \
                    self._read_cloudy_table(tables[z1])
            redshift2, nH_vals2, T_vals2, ions2, table2 = \
                    self._read_cloudy_table(tables[z2])
            if self._style == 'Oppenheimer new':
                z1, z2 = redshift1, redshift2
                if not np.allclose( nH_vals1, nH_vals2 ):
                    raise RuntimeError('nH values in tables do not match!')
                if not np.allclose( T_vals1, T_vals2 ):
                    raise RuntimeError('T values in tables do not match!')
                if not np.all( ions1 == ions2 ):
                    raise RuntimeError('Ions in tables do not match!')
                self._nH_vals = nH_vals1
                self._T_vals  = T_vals1
                self._ions    = ions1
            a = (self._redshift - z1) / (z2 - z1)
            table = (1.-a) * table1 + a * table2

        return table

    def show_table(self, ion=None, cmap='Bright'):
        '''
        Plot the table of the given ion.

        Args:
            ion (str, int):     The ion to plot the table for. If None, all tables
                                are plotted (and a list is returned).
            cmap (str):         The colormap to use.

        Returns:
            fig (Figure):       The figure of the axis plotted on.
            ax (AxesSubplot):   The axis plotted on.
            im (AxesImage):     The image instance created.
            cbar (Colorbar):    The colorbar.
        '''
        import matplotlib as mpl
        from ..plotting import show_image
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        if ion is None:
            plts = []
            for ion in self._ions:
                plts.append( self.show_table(ion, cmap=cmap) )
            return plts
        ionname = 'ion'
        if isinstance(ion, str):
            ion = str(ion)
            ionname = ion
            ion = self._ions.index(ion)

        fontsize = 16

        # plot the table as an image
        fig, ax, im = show_image(self._table[:,:,ion],
                                 extent=[[self._nH_vals.min(),self._nH_vals.max()],
                                         [self._T_vals.min(),self._T_vals.max()]],
                                 cmap=cmap)
        ax.set_xlabel(r'$\log_{10}(n\ [%s])$' % Unit('cm**-3').latex(),
                      fontsize=fontsize)
        ax.set_ylabel(r'$\log_{10}(T\ [%s])$' % Unit('K').latex(),
                      fontsize=fontsize)

        # add a colorbar
        clim = [self._table[:,:,ion].min(), self._table[:,:,ion].max()]
        cbar = add_cbar(ax, r'$\log_{10}(f_\mathrm{%s})$' % ionname,
                        clim=clim, cmap=cmap, fontsize=fontsize,
                        nticks=7)


        return fig, ax, im, cbar

