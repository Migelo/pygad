'''
Defining the disc etc.

Rewrite to use np.in1d to get the correct particles.

Examples:
    >>> from ..snapshot import Snapshot
    >>> from ..transformation import *
    >>> from ..analysis import *
    >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_470', physical=False)
    >>> Translation(-shrinking_sphere(s.stars, [s.boxsize/2]*3,
    ...                               np.sqrt(3)*s.boxsize)).apply(s)
    load block pos... done.
    do a shrinking sphere...
      starting values:
        center = [ 36000.  36000.  36000.] [ckpc h_0**-1]
        R      = 1.247077e+05 [ckpc h_0**-1]
    load block mass... done.
    done.
    apply Translation to "pos" of "snap_M1196_4x_470"... done.
    >>> s['vel'] -= mass_weighted_mean(s[s['r']<'1 kpc'], 'vel')
    load block vel... done.
    derive block r... done.
    >>> orientate_at(s[s['r'] < '10 kpc'].baryons, 'L', total=True)
    derive block momentum... done.
    derive block angmom... done.
    apply Rotation to "vel" of "snap_M1196_4x_470"... done.
    apply Rotation to "pos" of "snap_M1196_4x_470"... done.
    >>> sub = s[BallMask('60 kpc',sph_overlap=True)]    # 60 kpc ~ 30% R200 (R200 ~ 211 kpc)
    derive block r... done.
    load block hsml... done.
    >>> sub
    <Snap "snap_M1196_4x_470":Ball(r=60.0 [kpc]); N=198,250; z=0.000>
    >>> if abs(sub['r'].max().in_units_of('Mpc',subs=s) - 9.905) > 0.01:
    ...     print(sub['r'].max().in_units_of('Mpc',subs=s))
    >>> no_gas_max_r = sub.SubSnap(list(set(range(6))-set(gadget.families['gas'])))['r'].max()
    >>> if abs( no_gas_max_r.in_units_of('kpc',subs=s)  -  60.0 ) > 0.01:
    ...     print(no_gas_max_r)
    >>> s.to_physical_units()
    >>> sub = s[BoxMask('120 kpc',sph_overlap=False)]
    >>> sub # doctest:+ELLIPSIS
    <Snap "snap_M1196_4x_470":Box([[-60,60],[-60,60],[-60,60]] [kpc],strict); N=218,98...; z=0.000>
    >>> if np.linalg.norm( np.abs(sub['pos']).max(axis=0) - [120/2]*3 ) > 0.1:
    ...     print(np.abs(sub['pos']).max(axis=0))

    >>> discmask = DiscMask(0.85,rmax='60.0 kpc')
    >>> disc = s[discmask]
    derive block momentum... done.
    derive block angmom... done.
    derive block Ekin... done.
    derive block jcirc... done.
    derive block jzjc... done.
    derive block rcyl... done.
    >>> disc
    <Snap "snap_M1196_4x_470":Disc(jzjc<0.85,rcyl<60.0 [kpc],z<5.0 [kpc]); N=36,419; z=0.000>
    >>> gal = s[s['r']<'60 kpc']
    >>> np.array(disc.parts,float) / np.array(gal.parts)
    array([0.78431754,  0.02731191,         nan,         nan,  0.26441513,
                   nan])
    >>> bulge = gal[discmask.inverted()]
    >>> bulge
    <Snap "snap_M1196_4x_470":masked:~Disc(jzjc<0.85,rcyl<60.0 [kpc],z<5.0 [kpc]); N=159,918; z=0.000>
    >>> float(len(disc)) / len(bulge)
    0.2277354644255181

    >>> IDs = bulge['ID'][:1000:3]
    load block ID... done.
    >>> sub = gal[IDMask(IDs)]
    >>> sub
    <Snap "snap_M1196_4x_470":masked:IDMask; N=334; z=0.000>
    >>> assert np.all( sub['ID'] == IDs )
    >>> assert np.all( bulge[:1000:3]['pos'] == sub['pos'] )
    >>> antisub = gal[~IDMask(IDs)]
    >>> assert not bool( set(antisub['ID']).intersection(sub['ID']) )
    >>> assert len(antisub) + len(sub) == len(gal)
    >>> assert set(sub['ID']) == set(gal[IDMask(set(IDs))]['ID'])

    >>> expr_mask = ExprMask("abs(vel[:,2]) < '100 km/s'")
    >>> sub = gal[expr_mask]
    >>> sub
    <Snap "snap_M1196_4x_470":masked:ExprMask("abs(vel[:,2]) < '100 km/s'"); N=145,157; z=0.000>
    >>> assert np.all(sub['ID']
    ...                 == gal['ID'][np.abs(gal['vel'][:,2]) < '100 km/s'])
    >>> assert len(sub) + len(gal[~expr_mask]) == len(gal)
    >>> sub = gal[ExprMask("(abs(vel[:,2]) < '100 km/s') & (r < '30 kpc')")]
    >>> assert np.all(sub['ID'] == gal['ID'][
    ...     (np.abs(gal['vel'][:,2]) < '100 km/s') & (gal['r'] < '30 kpc')])

    Combining masks with (bitwise) logical operators:
    >>> s[expr_mask & discmask]
    <Snap "snap_M1196_4x_470":CompoundMask(ExprMask("abs(vel[:,2]) < '100 km/s'") & Disc(jzjc<0.85,rcyl<60.0 [kpc],z<5.0 [kpc])); N=36,072; z=0.000>
    >>> s[expr_mask | discmask]
    <Snap "snap_M1196_4x_470":CompoundMask(ExprMask("abs(vel[:,2]) < '100 km/s'") | Disc(jzjc<0.85,rcyl<60.0 [kpc],z<5.0 [kpc])); N=1,901,758; z=0.000>
    >>> s[expr_mask ^ discmask]
    <Snap "snap_M1196_4x_470":CompoundMask(ExprMask("abs(vel[:,2]) < '100 km/s'") ^ Disc(jzjc<0.85,rcyl<60.0 [kpc],z<5.0 [kpc])); N=1,865,686; z=0.000>
    >>> s[expr_mask & discmask & BallMask('30 kpc')]
    <Snap "snap_M1196_4x_470":CompoundMask(CompoundMask(ExprMask("abs(vel[:,2]) < '100 km/s'") & Disc(jzjc<0.85,rcyl<60.0 [kpc],z<5.0 [kpc])) & Ball(r=30.0 [kpc],strict)); N=34,177; z=0.000>
    >>> np.round( np.percentile(
    ...     s[ BallMask('100 kpc') & ~BallMask('30 kpc') ]['r'], [0,100]) )
    array([ 30.,  100.])
'''
__all__ = ['SnapMask', 'BallMask', 'BoxMask', 'DiscMask', 'IDMask', 'ExprMask']

import numpy as np
from ..units import *
from .. import gadget
#from .snapshot import SubSnap, SubSnap is now method of Snapshot
import operator


class SnapMask(object):
    '''The base class for more complicated masks.'''

    def __init__(self, inverse=False):
        self._inverse = bool(inverse)

    def is_inverse(self):
        return self._inverse

    def inverted(self):
        '''Return the inverse of this mask.'''
        raise NotImplementedError()

    def __invert__(self):
        return self.inverted()

    def __str__(self):
        s = '~' if self._inverse else ''
        s += self.__class__.__name__
        return s

    def _get_mask_for(self, s):
        '''
        The actual implementation for getting the mask for this class, not
        accounting for the _inverse flag.
        '''
        raise NotImplementedError('This is just the interface!')

    def get_mask_for(self, s):
        '''
        Actually get the mask for a given snapshot.

        Args:
            s (Snap):   The (sub-)snapshot to mask.

        Returns:
            mask (np.ndarray[bool]):    The mask to apply.
        '''
        mask = self._get_mask_for(s)
        return ~mask if self._inverse else mask

    def __and__(self, other):
        return CompoundMask(self, operator.and_, other)

    def __or__(self, other):
        return CompoundMask(self, operator.or_, other)

    def __xor__(self, other):
        return CompoundMask(self, operator.xor, other)


class BallMask(SnapMask):
    '''
    A mask for all particles within a given radius.

    Args:
        R (UnitScalar):         The maximum radius.
        center (UnitQty):       The center of the ball. Default: origin.
        sph_overlap (bool):     If True, also include gas particles, that actually
                                lie outside of the ball of radius R, but they are
                                smoothed into it.
        periodic_snap (bool):   Whether to consider the snapshot to be masked
                                periodic with its boxside or not.
    '''

    def __init__(self, R, center=None, sph_overlap=False, periodic_snap=True):
        super(BallMask, self).__init__()
        self.R = R
        self.center = center
        self.sph_overlap = sph_overlap
        self.periodic_snap = periodic_snap

    def inverted(self):
        inv = BallMask(self._R, self._center, sph_overlap=self.sph_overlap)
        inv._inverse = not self._inverse
        return inv

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        self._R = UnitScalar(value)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        if value is not None:
            self._center = UnitQty(value).copy()
        else:
            self._center = UnitQty([0] * 3)

    def __str__(self):
        s = '~' if self._inverse else ''
        s += 'Ball('
        if not np.all(self._center == 0):
            s += 'center=%s,' % self._center
        s += 'r=%s' % self._R
        if not self.sph_overlap:
            s += ',strict'
        return s + ')'

    def _get_mask_for(self, s):
        from ..utils import periodic_distance_to, dist
        R = self._R.in_units_of(s['r'].units, subs=s)
        center = self._center.in_units_of(s['r'].units, subs=s)

        if self.periodic_snap:
            r = periodic_distance_to(s['pos'], center, s.boxsize)
        else:
            r = dist(s['pos'], center) if not np.all(center == 0) else s['r']
        mask = r < R

        if self.sph_overlap and 'gas' in s:
            for pt in gadget.families['gas']:
                sub = s.SubSnap([pt])
                if self.periodic_snap:
                    r = periodic_distance_to(sub['pos'], center, s.boxsize)
                else:
                    r = dist(sub['pos'], center) if not np.all(center == 0) \
                        else sub['r']
                mask[sum(s.parts[:pt]):sum(s.parts[:pt + 1])] |= \
                    r - sub['hsml'] < R

        return mask.view(np.ndarray)


class BoxMask(SnapMask):
    '''
    A mask for all particles within a square box.

    The box is always a cube and aligned with the axes.

    Args:
        extent (Unit, UnitArr): The size of the box. Can either be a .
        center (UnitQty):       The center of the box, if extent is just a scalar.
                                Otherwise it will be ignored. Default: origin.
        sph_overlap (bool):     If True, also include gas particles, that actually
                                lie outside of the box, but they are smoothed
                                into it.
    '''

    def __init__(self, extent, center=None, sph_overlap=False):
        super(BoxMask, self).__init__()
        # might be needed for calculation of self.center in setting self.extent:
        self._extent = UnitArr([[-1, 1]] * 3)
        self.extent = extent
        if center is not None:
            self.center += center
        self.sph_overlap = sph_overlap

    def inverted(self):
        inv = BoxMask(self._extent, sph_overlap=self.sph_overlap)
        inv._inverse = not self._inverse
        return inv

    @property
    def extent(self):
        return self._extent

    @extent.setter
    def extent(self, value):
        extent = UnitQty(value, dtype=np.float64).copy()
        if extent.shape in [(), (3,)]:
            L = extent
            center = self.center
            extent = UnitArr(np.empty((3, 2), dtype=np.float64), extent.units)
            extent[:, 0] = center - L / 2.0
            extent[:, 1] = center + L / 2.0
        if extent.shape != (3, 2):
            raise ValueError('Extent has to ba a scalar or an array of shape ' +
                             '(3,) or (3,2), but got shape %s!' % (extent.shape,))
        self._extent = extent

    @property
    def center(self):
        return (self._extent[:, 1] + self._extent[:, 0]) / 2.0

    @center.setter
    def center(self, value):
        center_new = UnitQty(value, self._extent.units).copy()
        if center_new.shape != (3,):
            raise ValueError('Center has to have shape (3,)!')
        diff = center_new - (self._extent[:, 1] + self._extent[:, 0]) / 2.0
        self._extent[:, 0] += diff 
        self._extent[:, 1] += diff
    def __str__(self):
        s = '~' if self._inverse else ''
        s += 'Box(['
        for i in range(3):
            s += '[%.4g,%.4g]%s' % (tuple(self._extent[i]) +
                                    ('' if i == 2 else ',',))
        s += '] %s' % self._extent.units
        if not self.sph_overlap:
            s += ',strict'
        return s + ')'

    def _get_mask_for(self, s):
        ext = self._extent.in_units_of(s['pos'].units, subs=s)

        mask = (ext[0, 0] <= s['pos'][:, 0]) & (s['pos'][:, 0] <= ext[0, 1]) & \
               (ext[1, 0] <= s['pos'][:, 1]) & (s['pos'][:, 1] <= ext[1, 1]) & \
               (ext[2, 0] <= s['pos'][:, 2]) & (s['pos'][:, 2] <= ext[2, 1])

        if self.sph_overlap and 'gas' in s:
            for pt in gadget.families['gas']:
                sub = s.SubSnap([pt])
                mask[sum(s.parts[:pt]):sum(s.parts[:pt + 1])] |= \
                    (ext[0, 0] <= sub['pos'][:, 0] + sub['hsml']) & \
                    (sub['pos'][:, 0] - sub['hsml'] <= ext[0, 1]) & \
                    (ext[1, 0] <= sub['pos'][:, 1] + sub['hsml']) & \
                    (sub['pos'][:, 1] - sub['hsml'] <= ext[1, 1]) & \
                    (ext[2, 0] <= sub['pos'][:, 2] + sub['hsml']) & \
                    (sub['pos'][:, 2] - sub['hsml'] <= ext[2, 1])

        return mask.view(np.ndarray)


class DiscMask(SnapMask):
    '''
    A mask for all particles that belong to the disc.

    Here, the disc is specified by minimum jzjc (see definition of derived
    blocks!). Additionally one can require a maximum cylindrical distance form the
    center and/or a maximum modulus of the z-coordinate.

    Note:
        This definition is only sensible, if the (sub-)snapshot is already
        centered on the galaxy and orientated such that the total angular momentum
        (of the disc) points along positve z.

    Args:
        jzjc_min (float):   The minimum ratio of the z-component of the angular
                            momentum and the hypothetical circular velocity.
        rmax (UnitScalar):  An additional requirement on the cylindrical(!)
                            radius. If None, this requirement is ignored.
        zmax (UnitScalar):  An additional requirement on the z-coordinate. If
                            None, this requirement is ignored.
    '''

    def __init__(self, jzjc_min=0.85, rmax='50 kpc', zmax='5 kpc'):
        super(DiscMask, self).__init__()
        self.jzjc_min = jzjc_min
        self.rmax = rmax
        self.zmax = zmax

    def inverted(self):
        inv = DiscMask(self.jzjc_min, self._rmax, zmax=self._zmax)
        inv._inverse = not self._inverse
        return inv

    @property
    def rmax(self):
        return self._rmax

    @rmax.setter
    def rmax(self, value):
        if value is None:
            self._rmax = None
        else:
            self._rmax = UnitScalar(value)

    @property
    def zmax(self):
        return self._zmax

    @zmax.setter
    def zmax(self, value):
        if value is None:
            self._zmax = None
        else:
            self._zmax = UnitScalar(value)

    def __str__(self):
        s = '~' if self._inverse else ''
        s += 'Disc(jzjc<%.2f' % self.jzjc_min
        if self._rmax is not None:
            s += ',rcyl<%s' % self._rmax
        if self._zmax is not None:
            s += ',z<%s' % self._zmax
        return s + ')'

    def _get_mask_for(self, s):
        mask = s['jzjc'] > np.sign(np.mean(s['jzjc'])) * self.jzjc_min
        if self._rmax is not None:
            rmax = self._rmax.in_units_of(s['rcyl'].units, subs=s)
            mask &= s['rcyl'] < rmax
        if self._zmax is not None:
            zmax = self._zmax.in_units_of(s['pos'].units, subs=s)
            mask &= np.abs(s['pos'][:, 2]) < zmax
        return mask


class IDMask(SnapMask):
    '''
    Mask a snapshot to a list of given IDs.

    The given IDs do not all have to be present in the snapshot to be masked; the
    masked one, though does contain all those and only those partilces which IDs
    are given and present.

    Args:
        IDs (array-like):   IDs from previous snapshot to mask the current one.
    '''

    def __init__(self, IDs):
        super(IDMask, self).__init__()
        if isinstance(IDs, set):
            IDlist = list(IDs)
            if len(IDs) == 0:
                self._IDs = np.array(IDlist, dtype=np.uint64)
            else:
                self._IDs = np.array(IDlist)
        else:
            self._IDs = np.array(IDs, copy=True)
        return

    def inverted(self):
        inv = IDMask(self._IDs)
        inv._inverse = not self._inverse
        return inv

    @property
    def IDs(self):
        return set(self._IDs)

    def _get_mask_for(self, s):
        IDs = self._IDs
        if IDs.dtype != s['ID'].dtype:
            raise RuntimeError("You cannot mask with IDs of different type "
                               "(%s) than those of the snapshot (%s)!" % (
                                   IDs.dtype, s['ID'].dtype))
        return np.in1d(s['ID'], IDs)


class ExprMask(SnapMask):
    '''
    Mask a snapshot by a given expression.

    The expression defines a simple snapshot mask and shall be used in similar way
    as `Snap.get()`.

    Args:
        expr (str):     A python expression as it can be used in Snap.get()
                        returning a boolean array.
    '''

    def __init__(self, expr):
        super(ExprMask, self).__init__()
        self.expr = expr

    @property
    def expr(self):
        return self._expr

    @expr.setter
    def expr(self, expr):
        if not isinstance(expr, str):
            raise ValueError('`expr` must be a string')
        self._expr = expr.strip()

    def inverted(self):
        inv = ExprMask(self._expr)
        inv._inverse = not self._inverse
        return inv

    def __str__(self):
        s = '~' if self._inverse else ''
        s += 'ExprMask("%s")' % self._expr
        return s

    def _get_mask_for(self, s):
        mask = s.get(self._expr)
        return mask.view(np.ndarray)


class CompoundMask(SnapMask):
    '''
    A snapshot mask built from combining two others. Normally this class is not
    isinstantiated by the constructur, but by an operation of two others.

    Args:
        m1 (SnapMask):  The first operand.
        op (operator.and_, operand.or_, operand.xor):
                        Operation to combine the two masks.
        m2 (SnapMask):  The second operand.
    '''

    def __init__(self, m1, op, m2):
        super(CompoundMask, self).__init__()
        for m in (m1, m2):
            if not isinstance(m, SnapMask):
                raise ValueError('"%s" is not a SnapMask!' % m)
        if op not in (operator.and_, operator.or_, operator.xor):
            raise ValueError('Operator "%s" is not supported' % op)

        self._masks = [m1, m2]
        self._op = op

    @property
    def operator(self):
        return self._op

    def inverted(self):
        inv = CompoundMask(self._masks[0], self._op, self._masks[1])
        inv._inverse = not self._inverse
        return inv

    def __str__(self):
        s = '~' if self._inverse else ''
        s += 'CompoundMask(%s %s %s)' % (
            self._masks[0],
            {operator.and_: '&',
             operator.or_: '|',
             operator.xor: '^',
             }.get(self._op, str(self._op)),
            self._masks[1],
        )
        return s

    def _get_mask_for(self, s):
        m1, m2 = [m.get_mask_for(s) for m in self._masks]
        return self._op(m1, m2)

