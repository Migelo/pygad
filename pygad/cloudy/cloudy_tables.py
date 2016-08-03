'''
Read and interpolate Cloudy tables for ionisation states as a function of density
and temperature for a given redshift.
'''
__all__ = ['config_ion_table', 'IonisationTable']

from .. import environment
from ..units import Unit, UnitQty
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
    nH_vals = iontable['nH_vals']
    nH_vals = np.linspace(nH_vals[0],
                          nH_vals[0] + nH_vals[1]*(nH_vals[2]-1),
                          nH_vals[2])
    T_vals = iontable['T_vals']
    T_vals = np.linspace(T_vals[0],
                         T_vals[0] + T_vals[1]*(T_vals[2]-1),
                         T_vals[2])
    iontbl = IonisationTable(redshift,
                             nH=nH_vals,
                             T=T_vals,
                             ions=iontable['ions'],
                             tabledir=iontable['tabledir'],
                             table_pattern=iontable['pattern'])
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
        nH (array-like):        The Hydrogen number densities of the grid points
                                in units of log10(cm^-3).
        T (array-like):         The temperatures for the grid points in units of
                                log10(K).
        ions (iterable):        The names of the ions in the table.
        tabledir (str):         The directory where all the tables are located.
        table_pattern (str):    The pattern of the table file names. It is their
                                filename with '<z>' where the redshift is encoded.
                                This code is 10*z rounded to the next integer.
    '''

    def __init__(self, redshift, nH, T, ions, tabledir,
                 table_pattern='lt<z>f10'):
        nH = np.asarray(nH)
        T = np.asarray(T)
        if nH.ndim!=1 or T.ndim!=1:
            raise ValueError('`nH` and `T` need 1-dim., see docs!')
        if table_pattern.count('<z>') != 1:
            raise ValueError('The table pattern has to have exactly one ' +
                             'occurrence of "<z>"!')

        self._redshift = float(redshift)
        self._tabledir = tabledir
        self._pattern = table_pattern
        self._nH_vals = nH.copy()
        self._T_vals = T.copy()
        self._ions = list(ions)
        self._table = self._get_table()

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

    def interp_snap(self, ion, s, nH='H/mass*rho/m_H', T='temp'):
        '''
        Convenience function: calling `interp_parts(ion, s.get(nH), s.get(T), subs=s)`.
        '''
        if abs(s.redshift - self._redshift) > 0.01:
            import sys
            print >> sys.stderr, 'WARNING: the snapshot\'s redshift ' + \
                                 '(%.3f) does not match the ' % s.redshift + \
                                 'table\'s redshift (%.3f)!' % self._redshift
        return self.interp_parts(ion, s.get(nH), s.get(T), subs=s)

    def interp_parts(self, ion, nH, T, subs=None):
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
            subs (dict, snap):  Used for substitutions that might be neccessary
                                when converting units (for more info see
                                `UnitArr.in_units_of`).

        Returns:
            f (np.ndarray):     The fractions of the ion for each particle as read
                                / interpolated from the table.
        '''
        if len(nH) != len(T):
            raise ValueError('Lengths of nH (%d) and T (%d) do not match!' % (
                                len(nH),len(T)))
        if isinstance(ion,str):
            ion = self._ions.index(ion)
        if ion<0 or self._table.shape[-1]<=ion:
            raise ValueError('Ion index (%d) out of bounds!' % ion)
        # units of table
        nH = np.log10( UnitQty(nH, 'cm**-3', subs=subs) )
        T  = np.log10( UnitQty(T , 'K',      subs=subs) )

        # find lower indices
        N_nH = len(self._nH_vals)
        N_T  = len(self._T_vals)
        k_nH = -np.ones(len(nH), dtype=int)
        for k in xrange( N_nH ):
            k_nH[ self._nH_vals[k] < nH ] = k
        l_T = -np.ones(len(T), dtype=int)
        for l in xrange( N_T ):
            l_T[ self._T_vals[l] < T ] = l

        # do bilinear interpolation

        low_nH  = (k_nH<0)
        high_nH = (k_nH>=N_nH-1)
        low_T   = (l_T <0)
        high_T  = (l_T >=N_T -1)
        out_of_bounds = np.sum( low_nH | high_nH | low_T | high_T )
        if out_of_bounds:
            import sys
            print >> sys.stderr, 'WARNING: %d particles out of bounds!' % out_of_bounds
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
        table = np.loadtxt(filename)
        if len(self._nH_vals) * len(self._T_vals) != table.shape[0]:
            raise ValueError("table size (%d) " % table.shape[0] +
                             "does not match the grid given " +
                             "(%dx%d)!" % (len(self._nH_vals),len(self._T_vals)))
        if len(self._ions) != table.shape[1]:
            raise ValueError("table column size (%d) " % table.shape[1] +
                             "does not match the number of ions " +
                             "(%d)!" % len(self._ions))
        return table.reshape( (len(self._nH_vals), len(self._T_vals),
                               table.shape[1]) )

    def _get_table(self):
        # find all the available tables
        pattern = re.compile( self._pattern.replace('<z>',r'(\d+)') + '$' )
        tables = {}
        for filename in os.listdir(self._tabledir):
            match = re.match(pattern, filename)
            if match:
                z = 0.1 * int(match.group(1))
                tables[z] = os.path.join(self._tabledir,filename)
        # check if any table was found
        if len(tables) == 0:
            raise RuntimeError('No Cloudy tables in the given directory ' +
                               '("%s")!' % self._tabledir)
        # get the two closest tables and interpolate between
        avail_z = np.array( sorted(tables.iterkeys()) )
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
                print 'load table:'
                print '  "%s" (z=%.3f)' % (tables[z], z)
            table = self._read_cloudy_table(tables[z])
        else:
            if environment.verbose >= environment.VERBOSE_NORMAL:
                print 'load tables:'
                print '  "%s" (z=%.3f)' % (tables[z1], z1)
                print '  "%s" (z=%.3f)' % (tables[z2], z2)
            table1 = self._read_cloudy_table(tables[z1])
            table2 = self._read_cloudy_table(tables[z2])
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
            cbar (Colobar):     The colorbar.
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
        cax = inset_axes(ax, width="70%", height="3%", loc=1)
        norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                         orientation='horizontal')
        cbar.ax.tick_params(labelsize=8)
        for tl in cbar.ax.get_xticklabels():
            tl.set_fontsize(0.65*fontsize)
        cbar.set_label(r'$\log_{10}(f_\mathrm{%s})$' % ionname,
                       fontsize=fontsize, labelpad=12)

        return fig, ax, im, cbar

