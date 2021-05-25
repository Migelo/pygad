'''
A octree implementation that stores indices rather than positions itself. This
approach saves memory and makes it possible to also refere to other data attached
to the points as long as it is appropriately stored in an array, too (like in
snapshots).

Trees, subtrees, and nodes are all represented by the same OctNode class.

Note:
    This is a pure Python implementation and, hence, not very fast! However, an
    octree gives some more spatial information than the bare cloud of points.

Example:
    Start with a regular grid of points:
    >>> pos = np.linspace(0,1, 32)
    >>> pos = np.vstack(map(np.ravel, np.meshgrid(pos, pos, pos))).T
    >>> OctNode.SPLIT_NUM = 12
    >>> tree = Octree(pos)
    >>> count_nodes(tree)
    4681
    >>> count_leaves(tree)
    4096
    >>> max_depth(tree)
    4
    >>> tree.get_child([7,0,5,3])
    OctNode(center=[0.281,0.4062,0.3436], side=0.06256, depth=4, leaf)
    >>> tree.get_child([7,0,5,3]).child
    [12554, 12555, 12586, 12587, 13578, 13579, 13610, 13611]
    >>> find_node(tree, pos[12586]) is tree.get_child([7,0,5,3])
    True
    >>> for r in pos:
    ...     n = find_node(tree, r)
    ...     if max(abs(r - n.center)) > n.side:
    ...         print(r, n, n.side)
    >>> for i in range(20):
    ...     r = np.random.rand(3)
    ...     np_idcs = np.where(cdist(pos, [[.3,.3,.3]]).ravel() < 0.1)[0]
    ...     ot_idcs = find_ngbs(tree, [.3,.3,.3], 0.1, pos)
    ...     if not set(np_idcs) == set(ot_idcs):
    ...         print('numpy:')
    ...         print(pos[np_idcs])
    ...         print( 'octree:')
    ...         print(pos[ot_idcs])

    A more realistic sample of positions (just every 100th for speed reasons):
    >>> from ..snapshot.snapshot import Snapshot
    >>> from ..environment import module_dir
    >>> snap = Snapshot(module_dir+'snaps/snap_M1196_4x_470', physical=False,
    ...             load_double_prec=True)
    >>> snap = snap[::100]
    >>> OctNode.MAX_DEPTH = 11  # making the following tree build way faster
    >>> tree = Octree(snap)
    load block pos... done.
    >>> count_particles(tree), count_nodes(tree), count_leaves(tree)
    (20791, 5515, 4736)
    >>> apply_on(tree, delete_where_too_many_heavy_particles(snap,
    ...                                     min_side='50 ckpc / h_0'))
    (0, 18643)
    >>> sub = snap[all_indices(tree)]
    >>> sub.parts
    [8573, 9334, 0, 0, 736, 0]
    >>> apply_on(tree, populate_center_of_mass(snap['pos'], snap['mass']))  # doctest:+ELLIPSIS
    load block mass... done.
    (array([34497.34533395, 35431.564173  , 33097.62806547]), 2.926226872834377)
    >>> from .. import analysis
    >>> com = analysis.center_of_mass(sub)
    >>> d = np.sqrt(np.sum((tree.com - com)**2))
    >>> if not d < 0.5:
    ...     print('com      =', com)
    ...     print('tree.com =', tree.com)
    >>> del_quantity(tree, 'com')
    >>> del_quantity(tree, 'mass')
'''
__all__ = ['OctNode', 'Octree', 'count_nodes', 'count_leaves', 'count_particles',
           'max_depth', 'find_node', 'find_ngbs', 'all_indices', 'apply_on',
           'del_quantity', 'populate_center_of_mass',
           'delete_where_too_many_heavy_particles']

import numpy as np
from scipy.spatial.distance import cdist
from ..units import *
from .. import environment
import sys


def find_octant(pos, ref):
    '''Index corresponding to an octant of 'pos' with respect to 'ref'.'''
    res = 0
    if pos[0] < ref[0]: res += 1
    if pos[1] < ref[1]: res += 2
    if pos[2] < ref[2]: res += 4
    return res


class OctNode(object):
    '''
    Class for an octree node.

    You should not create it directly, but via the factory function Octree. The
    data stored per leaf is not the positions of the individual particle, but
    rather a reference to an array holding all positions (and possible more)
    together with the indices of the particles of that leaf. This saves memory and
    enables one to also refere to other data of the particle stored somewhere
    else via the indices.

    Note:
        This class does not use data encapsulation for convenience.
        Be responsible!

    Static variables:
        MAX_DEPTH (int):        The maximum depth to use for the octree. It is
                                neccessary to avoid stack overflows in cases where
                                two particles are very close to each other.
        SPLIT_NUM (int):  The number of particles at which to split a leaf
                                node. Hence, the maximum number of particles per
                                (leaf) node is SPLIT_NUM-1.
    '''

    MAX_DEPTH = 15
    SPLIT_NUM = 7

    def __init__(self, center, side, depth):
        self.center = center  # the center of this octree node cube
        self.side = side  # the (full) side length of this cube
        self.depth = depth  # the depth level of this node (root is 0)
        self.is_leaf = True  # whether this node has no children
        self.child = []  # either the child nodes or the indices of the
        # positions if this is a leaf node

    def __repr__(self):
        re = 'OctNode('
        re += 'center=[%.4g,%.4g,%.4g]' % tuple(coord for coord in self.center)
        re += ', side=%.4g' % self.side
        re += ', depth=%d' % self.depth
        if self.is_leaf:
            re += ', leaf'
        else:
            re += ', non-leaf'
        re += ')'
        return re

    def particles(self):
        '''Count the number of particles in the (sub-)tree of this node.'''
        if self.is_leaf:
            return len(self.child)
        cnt = 0
        for child in self.child:
            if child is not None:
                cnt += child.particles()
        return cnt

    def get_child(self, path):
        '''
        Get a child by following the path of octants.

        Args:
            path (list):    The child octant number path to follow. Example:
                            path=[3,0,7] would give the child of octant 7 of the
                            the child of octant 0 of the child of octant 3 of this
                            node.

        Returns:
            child (OctNode, ...):   The corresponding child.
        '''
        if len(path) == 1:
            return self.child[path[0]]
        else:
            return self.child[path[0]].get_child(path[1:])

    def insert(self, idx, pos_data):
        '''
        Insert a data point into this node.

        Note:
            Using a UnitArr as pos_data can be slow, since it is indexed in only
            the first dimension, leaving the units. consider passing a np.ndarray
            view.

        Args:
            idx (int):              The index of the position in the position
                                    vector pos_data.
            pos_data (array-like):  All positions. It has to have shape (N,3) ---
                                    an array of 3-dim. position vectors. However,
                                    it is not neccessary that all positions of
                                    this array are inserted to the tree.
        '''
        if self.is_leaf:
            self.child.append(idx)
            if len(self.child) > OctNode.SPLIT_NUM \
                    and self.depth < OctNode.MAX_DEPTH:
                self.is_leaf = False
                idcs = self.child
                self.child = [None] * 8
                for idx in idcs:
                    self.insert(idx, pos_data)
        else:
            octant = find_octant(pos_data[idx], self.center)
            if self.child[octant] is None:
                cntr = self.center
                off = self.side / 4.0
                if octant == 0:
                    cntr = (cntr[0] + off, cntr[1] + off, cntr[2] + off)
                elif octant == 1:
                    cntr = (cntr[0] - off, cntr[1] + off, cntr[2] + off)
                elif octant == 2:
                    cntr = (cntr[0] + off, cntr[1] - off, cntr[2] + off)
                elif octant == 3:
                    cntr = (cntr[0] - off, cntr[1] - off, cntr[2] + off)
                elif octant == 4:
                    cntr = (cntr[0] + off, cntr[1] + off, cntr[2] - off)
                elif octant == 5:
                    cntr = (cntr[0] - off, cntr[1] + off, cntr[2] - off)
                elif octant == 6:
                    cntr = (cntr[0] + off, cntr[1] - off, cntr[2] - off)
                elif octant == 7:
                    cntr = (cntr[0] - off, cntr[1] - off, cntr[2] - off)
                cntr = np.array(cntr)
                self.child[octant] = OctNode(cntr, self.side / 2.0, self.depth + 1)
            self.child[octant].insert(idx, pos_data)


def Octree(pos_data, idx=None, center=None, world_size=None):
    '''
    Build an octree from position data.

    Args:
        pos_data (UnitArr, np.ndarray):     A (N,3)-array of positions.
        idx (array-like):                   The indices of the particles/positions
                                            in pos_data to add to the tree. If
                                            None, all particles are added.
        center (UnitArr, array-like):       The center of the octree. If omitted,
                                            the middle of all position is taken.
        world_size (UnitArr, Unit, float):  The size (full side length) of the
                                            root node of the octree. If not
                                            specified, the maximum distance in any
                                            coordinate of pos_data from the center
                                            is taken.

    Returns:
        root (OctNode):     The root node of the octree built.
    '''
    from ..snapshot.snapshot import Snapshot
    if isinstance(pos_data, Snapshot):
        s = pos_data
        bz = float(s.boxsize.in_units_of(s['pos'].units, copy=False))
        return Octree(s['pos'], center=[bz / 2.] * 3, world_size=bz)

    if environment.verbose >= environment.VERBOSE_TALKY:
        print('build a Octree with %s positions' % (
            utils.nice_big_num_str(len(s))))
        sys.stdout.flush()

    if center is None:
        upper = np.max(pos_data, axis=0)
        lower = np.min(pos_data, axis=0)
        center = (upper + lower) / 2.0

    if world_size is None:
        try:
            world_size = 1.001 * pos_data.ptp(axis=0).max()
        except:
            world_size = 1.001 * np.array(pos_data).ptp(axis=0).max()

    if isinstance(pos_data, UnitArr):
        units = pos_data.units
        if hasattr(center, 'in_units_of'):
            center = center.in_units_of(units, copy=False)
        if hasattr(world_size, 'in_units_of'):
            world_size = world_size.in_units_of(units, copy=False)
        pos_data = pos_data.view(np.ndarray)
    pos_data = np.array(pos_data)

    root = OctNode(np.array(list(center)), float(world_size), 0)

    if idx is None:
        idx = range(len(pos_data))
    for i in idx:
        root.insert(i, pos_data)

    if environment.verbose >= environment.VERBOSE_TALKY:
        print('done.')
        sys.stdout.flush()

    return root


def count_nodes(node):
    '''Count the total number of nodes within the given tree/node.'''
    cnt = 1  # there is at least this node
    if not node.is_leaf:
        for child in node.child:
            if child is not None:
                cnt += count_nodes(child)
    return cnt


def count_leaves(node):
    '''Count the number of leaf nodes within the given tree/node.'''
    if node.is_leaf:
        return 1
    cnt = 0
    for child in node.child:
        if child is not None:
            cnt += count_leaves(child)
    return cnt


def count_particles(node):
    '''Count the number of particles within the given tree/node.'''
    if node.is_leaf:
        return len(node.child)
    cnt = 0
    for child in node.child:
        if child is not None:
            cnt += count_particles(child)
    return cnt


def max_depth(node):
    '''Find the maximum depth within the given tree/node.'''
    if node.is_leaf:
        return node.depth
    d_max = -np.inf
    for child in node.child:
        if child is not None:
            d_max = max(d_max, max_depth(child))
    return d_max


def find_node(tree, pos):
    '''
    Find the 'smallest' node that would contain the position.

    Args:
        tree (OctNode):         The root node of the tree to search in.
        pos (array-like):       The reference position to find the node for.

    Returns:
        node (OctNode):         The smallest node (i.e. the one with the highest
                                depth) that contains the position. It is not
                                neccessarily a leaf node; it might be a node that
                                has children, but not the one that would contain
                                the position of interest.
    '''
    if tree.is_leaf:
        return tree
    o = find_octant(pos, tree.center)
    if tree.child[o] is None:
        return tree
    return find_node(tree.child[o], pos)


def _find_ngbs(node, pos, r, pos_data):
    '''Helper function for speed. See find_ngbs!'''
    if node.is_leaf:
        d = cdist(pos_data[node.child], [pos]).ravel()
        return list(np.array(node.child)[d <= r])
    # half the size of the cube of a child node (plus some safety margin)
    child_half_side = 1.01 * node.side / 4.0
    ngbs = []
    for child in node.child:
        if child is None:
            continue
        # if the distance to the edge of the child node is too large, also skip
        if all(abs(child.center - pos) > r + child_half_side):
            continue
        ngbs += _find_ngbs(child, pos, r, pos_data)
    return ngbs


def find_ngbs(node, pos, r, pos_data):
    '''
    Find the all neighbours of the given position that are closer than r.

    Note:
        Since numpy uses C in the backend (e.g. for loops), it is still orders of
        magnidutes faster to just use scipy.spatial.distance.cdist together with
        np.where rather than this function (even if the tree is already built)!

    Args:
        node (OctNode):             The root node of the tree to search in.
        pos (UnitArr, array-like):  The reference position to find neighbours for.
        r (UnitArr, Unit, float):   The maximum distance to the neighbours.
        pos_data (UnitArr, array-like):
                                    The position data of the octree.
    '''
    if isinstance(pos_data, UnitArr):
        units = pos_data.units
        if hasattr(pos, 'in_units_of'):
            pos = pos.in_units_of(units, copy=False)
        if hasattr(r, 'in_units_of'):
            r = float(r.in_units_of(units, copy=False))
        pos_data = pos_data.view(np.ndarray)
    return _find_ngbs(node, pos, r, pos_data)


def _all_indices(node):
    '''Helper function for speed (no copy). See all_indices!'''
    if node.is_leaf:
        return node.child
    else:
        idx = []
        for child in node.child:
            if child is not None:
                idx += _all_indices(child)
        return idx


def all_indices(node):
    '''Gather all position indices within the tree.'''
    if node.is_leaf:
        return np.array(node.child, int)
    else:
        return np.array(_all_indices(node), int)


def apply_on(node, action):
    '''
    Apply the defined action on each node recursively from bottom.

    Args:
        node (OctNode):     The tree on which nodes the action is applied on.
        action (callable):  The action that is applied with the following call
                            signature: action(node, children), where children is a
                            list of the results of all the children (if it does
                            not exist, it is None).
    '''
    children = None
    if not node.is_leaf:
        children = [None] * 8
        for o, child in enumerate(node.child):
            if child is not None:
                children[o] = apply_on(child, action)
    return action(node, children)


def del_quantity(node, name):
    '''Delete a quantity with the given name at each node.'''
    if not node.is_leaf:
        for child in node.child:
            if child is not None:
                del_quantity(child, name)
    delattr(node, name)


class populate_center_of_mass(object):
    '''
    A callable that can be passed to 'apply_on' in order to populate the tree with
    center of mass (attribute 'com') and total mass (attribute 'mass')
    information.

    Args:
        pos (np.array):     The position data of the tree.
        mass (np.array):    The mass data of the tree.

    Call signature:
        fill_center_of_mass(node,children) -> (com,mass)

        where children is an array of the results of the children of this node.
        For children which do not exist the entry is None.
    '''

    def __init__(self, pos, mass):
        # a view as a np.ndarray makes the slicing (indexing in the first
        # dimension) faster
        self.pos = pos.view(np.ndarray)
        self.mass = mass.view(np.ndarray)

    def __call__(self, node, children):
        if node.is_leaf:
            r = self.pos[node.child]
            m = self.mass[node.child]
        else:
            r, m = np.zeros((8, 3)), np.zeros(8)
            for o, child in enumerate(children):
                if child is not None:
                    r[o] = child[0]
                    m[o] = child[1]
        setattr(node, 'mass', m.sum())
        setattr(node, 'com', (m * r.T).sum(axis=1) / node.mass)
        return node.com, node.mass


class delete_where_too_many_heavy_particles(object):
    '''
    A callable that can be passed to 'apply_on' in order to delete all tree leaves
    where the fraction of low resolution-particles is too high.

    Args:
        snap (Snap):        The snapshot with the position data of the tree.
        min_side (UnitScalar):
                            The minimum side length of a node to be allowed to be
                            deleted. If this would be 0.0, propably only leaf
                            nodes containing low resolution particles would be
                            deleted, since OctNode.SPLIT_NUM is typically of order
                            10 and, hence, a single particle brings the fraction
                            up to 10%.
        max_frac (float):   The maximum fraction of los resolution-particles
                            allowed.
        lowres_pts (list):  The particle types to consider low resolution
                            particles.
                            Default: [2,3]

    Call signature:
        delete_where_too_many_heavy_particles(node,children) -> (N_low_res,N_tot)

        where children is an array of the results of the children of this node.
        For children which do not exist the entry is None.
    '''

    def __init__(self, snap, min_side, max_frac=1e-3, lowres_pts=None):
        self.max_frac = max_frac
        self.min_side = float(UnitScalar(min_side, snap.boxsize.units, subs=snap))

        self.lowres_pts = [2, 3] if lowres_pts is None else list(lowres_pts)
        self.N_cumsum = np.cumsum(snap.parts)

        def is_low_res(i):
            for pt in self.lowres_pts:
                if self.N_cumsum[pt - 1] <= i < self.N_cumsum[pt]:
                    return True
            return False

        self.is_low_res_idx = is_low_res

    def __call__(self, node, children):
        if node.is_leaf:
            N_low_res = sum(self.is_low_res_idx(i) for i in node.child)
            N_tot = len(node.child)
            return N_low_res, N_tot
        else:
            N_low_res, N_tot = 0, 0
            for o, child in enumerate(children):
                if child is not None:
                    if node.child[o].side > self.min_side and (child[1] == 0 \
                                                               or float(child[0]) / child[1] > self.max_frac):
                        node.child[o] = None
                    else:
                        N_low_res += child[0]
                        N_tot += child[1]
            return N_low_res, N_tot

