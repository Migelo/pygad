'''
A fast octree class (implemented in C, here is only the interface)

TODO:
    * handle UnitArr's! (save units of stored max_H?!)
    * find a way *not* to restrict the numbers of neighbours returned in
      `find_ngbs_within` and `find_ngbs_SPH`
    * implement 2D tree (quadtree)? -> e.g. speed-up for maps

Doctests:
    Generate N random points in a box of [0,L]**3
    Keep N small, since there is a comparison with brute force done!
    >>> N, L, N_ngbs = int(3e4), 1.3, 100
    >>> pos = L * np.random.random((N,3))

    >>> ngbs = N_ngbs * (0.9 + 0.2*np.random.random(N))
    >>> H = L * ( ngbs/N / (4./3*np.pi) )**(1/3.)
    >>> tree = cOctree(pos, H)

    some very basic testing of the interfacing
    >>> x_min, x_max = pos.min(axis=0), pos.max(axis=0)
    >>> center, side_2 = (x_max+x_min)/2.0, max(x_max-x_min)/2.0
    >>> assert np.linalg.norm(tree.center-center)/L<1e-6
    >>> assert abs(tree.side_2-side_2)/L<1e-6
    >>> assert tree.tot_num_part == N
    >>> assert tree.count_nodes(False) <= N
    >>> assert tree.is_in_node(center)
    >>> assert tree.max_H == np.max(H)
    >>> assert not tree.is_in_node(center+[side_2*2.0]*3)
    >>> assert not tree.is_in_node(center-[side_2*2.0]*3)

    >>> assert tree.root is tree
    >>> particles, nodes = 0, 0
    >>> for o in range(8):
    ...     child = tree.get_child(o)
    ...     assert tree.get_octant(child.center) == o
    ...     assert child.parent is tree
    ...     assert child.root is tree
    ...     for o in range(8):
    ...         grandchild = child.get_child(o)
    ...         assert grandchild.parent is child
    ...         assert grandchild.root is tree
    ...     assert tree.is_in_node(child.center)
    ...     particles += child.tot_num_part
    ...     nodes += child.count_nodes()
    >>> assert particles == tree.tot_num_part
    >>> assert nodes + 1 == tree.count_nodes()

    brute force neighbours
    >>> r = center + np.array([-0.3,0.1,0.2]) * side_2
    >>> h = np.mean(H)
    >>> ngbs = []
    >>> for i in range(N):
    ...     d = np.linalg.norm(pos[i] - r)
    ...     if d < h:
    ...         ngbs.append(i)
    >>> assert 0 < len(ngbs) < 3*N_ngbs
    >>> tree_ngbs = tree.find_ngbs_within(r, h, pos, np.inf, max_ngbs=3*N_ngbs)
    >>> assert set(tree_ngbs) == set(ngbs)
    >>> ngbs = []
    >>> for i in range(N):
    ...     d = np.linalg.norm(pos[i] - r)
    ...     if d < H[i]:
    ...         ngbs.append(i)
    >>> assert 0 < len(ngbs) < 3*N_ngbs
    >>> tree_ngbs = tree.find_ngbs_SPH(r, H, pos, np.inf, max_ngbs=3*N_ngbs)
    >>> assert set(tree_ngbs) == set(ngbs)

    Test periodic neighbour finding
    >>> N_p_ngbs = 10
    >>> h = (float(N_p_ngbs)/N_ngbs)**(1/3.) * h
    >>> tree_ngbs = tree.find_ngbs_within([0]*3, h, pos, L, max_ngbs=3*N_ngbs)
    >>> from scipy.spatial.distance import cdist
    >>> normal_dists = cdist(pos[tree_ngbs], [[0,0,0]])
    >>> periodic_dists = np.min( cdist(pos[tree_ngbs], [[0,0,0],
    ...                                                 [0,0,L],
    ...                                                 [0,L,0],
    ...                                                 [0,L,L],
    ...                                                 [L,0,0],
    ...                                                 [L,0,L],
    ...                                                 [L,L,0],
    ...                                                 [L,L,L]]),
    ...                          axis=1)
    >>> assert np.all(periodic_dists<h)
    >>> assert np.any(normal_dists>h)

    More neighbour finding
    >>> pos = np.array([[0.1,0.3,0.2], [0.9,0.3,0.2], [0.8,0.5,0.1],
    ...                 [0.1,0.6,0.8], [0.2,0.2,0.3], [0.6,0.6,0.7]])
    >>> tree = cOctree(pos)
    >>> idx = np.arange(len(pos))

    TODO: the conditioning does not work (properly) in the bitbucket Pipeline
    >>> tree.find_ngbs_within(pos[0], 0.2, pos) #, cond=idx!=0)
    array([0, 4], dtype=uint64)
    >>> tree.find_ngbs_within(pos[0], 0.2, pos, cond=None)
    array([0, 4], dtype=uint64)
    >>> tree.find_next_ngb([0.5]*3, pos)
    5
    >>> tree.find_next_ngb([0.5]*3, pos, cond=idx%2==0)
    4
    >>> tree.find_next_ngb([0.5]*3, pos)#, cond=np.zeros(len(pos)))
    5
'''
__all__ = ['cOctree']

from ..C import *
import sys
import numpy as np
import warnings
import weakref
from .. import environment
from .. import utils

cpygad.new_octree_from_pos.restype = c_void_p
cpygad.new_octree_from_pos.argtypes = [c_size_t, c_void_p]
cpygad.free_octree.argtypes = [c_void_p]
cpygad.get_octree_center.argtypes = [c_void_p, c_void_p]
cpygad.get_octree_side_2.restype = c_double
cpygad.get_octree_side_2.argtypes = [c_void_p]
cpygad.get_octree_is_leaf.restype = c_int
cpygad.get_octree_is_leaf.argtypes = [c_void_p]
cpygad.get_octree_num_children.restype = c_uint
cpygad.get_octree_num_children.argtypes = [c_void_p]
cpygad.get_octree_tot_part.restype = c_size_t
cpygad.get_octree_tot_part.argtypes = [c_void_p]
cpygad.get_octree_max_H.restype = c_double
cpygad.get_octree_max_H.argtypes = [c_void_p]
cpygad.get_octree_max_H.restype = c_double
cpygad.get_octree_max_H.argtypes = [c_void_p]
cpygad.get_octree_max_depth.restype = c_int
cpygad.get_octree_max_depth.argtypes = [c_void_p]
cpygad.get_octree_node_count.restype = c_size_t
cpygad.get_octree_node_count.argtypes = [c_void_p, c_int]
cpygad.get_octree_in_region.restypes = c_int
cpygad.get_octree_in_region.argtypes = [c_void_p, c_void_p]
cpygad.fill_octree.argtypes = [c_void_p, c_size_t, c_void_p]
cpygad.update_octree_max_H.argtypes = [c_void_p, c_void_p]
cpygad.update_octree_const_max_H.argtypes = [c_void_p, c_double]
cpygad.get_octree_child.restype = c_void_p
cpygad.get_octree_child.argtypes = [c_void_p, c_int]
cpygad.get_octree_octant.restype = c_uint
cpygad.get_octree_octant.argtypes = [c_void_p, c_void_p]
cpygad.get_octree_ngbs_within.argtypes = [c_void_p,
                                          c_void_p, c_double,
                                          c_size_t, c_void_p, POINTER(c_size_t),
                                          c_void_p, c_double,
                                          c_void_p]
cpygad.get_octree_ngbs_SPH.argtypes = [c_void_p,
                                       c_void_p, c_void_p,
                                       c_size_t, c_void_p, POINTER(c_size_t),
                                       c_void_p, c_double, c_double]
cpygad.get_octree_next_ngb.argtypes = [c_void_p, c_void_p, c_void_p, c_double,
                                       c_void_p]


class _MAX_TREE_LEVEL_class(type):
    _MAX_TREE_LEVEL = int(c_int.in_dll(cpygad, 'MAX_TREE_LEVEL').value)

    def _get_MAX_TREE_LEVEL(self):
        return self._MAX_TREE_LEVEL

    def _set_MAX_TREE_LEVEL(self, value):
        raise AttributeError("can't set attribute")

    MAX_TREE_LEVEL = property(_get_MAX_TREE_LEVEL, _set_MAX_TREE_LEVEL)


class cOctree(object, metaclass=_MAX_TREE_LEVEL_class):
    '''
    A octree implementation with the backend in written in C.

    Actually this is a wrapper to the C++ template class Tree<3>. Internally only
    indices are stored so that any property of a particle can be referenced.

    Args:
        center (array-like):    TODO
        side_2 (float):         TODO
    '''

    @property
    def MAX_TREE_LEVEL(self):
        return cOctree.MAX_TREE_LEVEL

    def __init__(self, pos, H=None):
        if environment.verbose >= environment.VERBOSE_TALKY:
            print('build a cOctree with %s positions' % (
                utils.nice_big_num_str(len(s))))
            sys.stdout.flush()
        pos = np.asarray(pos, dtype=np.float64)
        if pos.shape[1:] != (3,):
            raise ValueError('Positions have to have shape (N,3)!')
        if pos.base is not None:
            pos = pos.copy()

        self.__parent = None
        self.__node_ptr = cpygad.new_octree_from_pos(len(pos), pos.ctypes.data)
        # If this object does not have any references anymore, and hence will get
        # garbage collected, the weakref will not reference any object anymore
        # and, hence, call its callback function, which in turn has the
        # responsibility to free the underlying C memory. We need to make sure the
        # closure does not hold references to `self`. Otherwise the weakref would
        # always point to a living object, and thus would never call its callback
        # function.
        # Note: child node cOctrees do not get instantiated with a call of
        # `__init__`.
        self.__weakref_for_freeing_C_memory = weakref.ref(
            self,
            lambda wr, ptr=self.__node_ptr: cpygad.free_octree(ptr),
        )

        if H is not None:
            self.update_max_H(H)

        if environment.verbose >= environment.VERBOSE_TALKY:
            print('done.')
            sys.stdout.flush()

    @property
    def parent(self):
        '''The parent node of this node (None for root).'''
        return self.__parent

    @property
    def root(self):
        '''The root of this octree node (self for root node).'''
        root = self
        while root.__parent:
            root = root.__parent
        return root

    @property
    def center(self):
        center = np.empty((3,), dtype=np.float64)
        cpygad.get_octree_center(self.__node_ptr, center.ctypes.data)
        return center

    @property
    def side_2(self):
        '''Half the side length of the tree box.'''
        return float(cpygad.get_octree_side_2(self.__node_ptr))

    @property
    def full_side(self):
        '''The total side length of the tree box.'''
        return 2.0 * self.side_2

    @property
    def is_leaf(self):
        '''Whether this is a leaf node.'''
        return bool(cpygad.get_octree_is_leaf(self.__node_ptr))

    @property
    def num_children(self):
        '''Number of child nodes.'''
        return int(cpygad.get_octree_num_children(self.__node_ptr))

    @property
    def tot_num_part(self):
        '''Get the total number of particles in the tree.'''
        return int(cpygad.get_octree_tot_part(self.__node_ptr))

    @property
    def max_H(self):
        '''The maximum smoothing length (as support radius) in the tree.'''
        return float(cpygad.get_octree_max_H(self.__node_ptr))

    @property
    def max_depth(self):
        '''The maximum depth of this particular tree. (Only node would be 0.)'''
        return int(cpygad.get_octree_max_depth(self.__node_ptr))

    def get_child(self, o):
        '''Return the child of octant o.'''
        if 0 <= o < self.num_children:
            # Do not call __init__ and hence do not set the weakref, which would
            # free the memory of this child, which is part of the parent's tree!
            child = cOctree.__new__(cOctree)
            child.__parent = self
            child.__node_ptr = cpygad.get_octree_child(self.__node_ptr, o)
            return child
        else:
            return None

    def get_octant(self, r):
        '''Get the octant of position `r`.'''
        r = np.asarray(r, dtype=np.float64).copy()
        if r.shape != (3,):
            raise ValueError('Position has to have shape (3,)!')
        return int(cpygad.get_octree_octant(self.__node_ptr, r.ctypes.data))

    def count_nodes(self, count_non_leaves=True):
        '''
        Count all nodes in the tree.

        Args:
            count_non_leaves (bool):    Whether to count all nodes (that is also
                                        include nodes that are no leaves). If set
                                        to False, only count leaf nodes.

        Returns:
            nodes (int):                The number of nodes.
        '''
        nol = int(bool(count_non_leaves))
        return int(cpygad.get_octree_node_count(self.__node_ptr, nol))

    def is_in_node(self, r):
        '''Check whether position r lies within this node.'''
        r = np.asarray(r, dtype=np.float64).copy()
        if r.shape != (3,):
            raise ValueError('Position has to have shape (3,)!')
        return bool(cpygad.get_octree_in_region(self.__node_ptr, r.ctypes.data))

    def update_max_H(self, H=None):
        '''
        Fill the tree with particles (without deleting former particles from
        tree).

        Args:
            H (array-like, float):  The smoothing lengthes of the particles. Has
                                    to have shape (N,) or can be a float (all
                                    smoothing lengthes then are the same).
        '''
        from numbers import Number
        if isinstance(H, Number):
            H = float(H)
            cpygad.update_octree_const_max_H(self.__node_ptr, H)
        else:
            H = np.asarray(H, dtype=np.float64)
            if H.shape != (self.tot_num_part,):
                raise ValueError('Smoothing lengthes have to have shape (N,)!')
            if H.base is not None:
                H = H.copy()
            cpygad.update_octree_max_H(self.__node_ptr, H.ctypes.data)

    def find_ngbs_within(self, r, H, pos, periodic=np.inf, cond=None, max_ngbs=100):
        '''
        Find all particles in tree within distance `H` from position `r`.

        Args:
            r (array-like):     Reference position.
            H (float):          Maximum distance to search for neighbours in.
            pos (array-like):   The positions corresponding to the indices of the
                                tree.
            periodic (float):   Assume the particles to sit in a periodic cube
                                with this side length.
            cond (array-like):  Boolean array of a condition to be fulfilled such
                                that a particle is registered as a neighbour. In
                                other words, each particle i for which cond[i] is
                                False is excluded from the neighbours.
            max_ngbs (int):     Only return this number of neighbours at maximum.
                                (Not necessarily the closest ones!)

        Returns:
            ngbs (np.ndarray):  List if the indices of the neighbours (in random
                                order). At maximum, though, `max_ngbs` of them.
        '''
        r = np.asarray(r, dtype=np.float64).copy()
        H = float(H)
        pos = np.asarray(pos, dtype=np.float64)
        if pos.shape[1:] != (3,):
            raise ValueError('Positions have to have shape (N,3)!')
        if pos.base is not None:
            pos = pos.copy()
        max_ngbs = int(max_ngbs)
        periodic = float(periodic)

        ngbs = np.empty(max_ngbs, dtype=np.uintp)
        N_ngbs = c_size_t()
        if cond is not None:
            cond = np.asarray(cond, dtype=np.int32)
            if cond.shape != (len(pos),):
                raise ValueError('Unmatching shape of `cond`: %s!' % (cond.shape,))
            if cond.base is not None:
                cond = cond.copy()
            cond = cond.ctypes.data
        cpygad.get_octree_ngbs_within(self.__node_ptr,
                                      r.ctypes.data, H,
                                      max_ngbs, ngbs.ctypes.data, byref(N_ngbs),
                                      pos.ctypes.data, periodic,
                                      cond,
                                      )
        #ngbs.resize(N_ngbs.value)
        erg = np.resize(ngbs, N_ngbs.value)
        return erg

    def find_ngbs_SPH(self, r, H, pos, periodic=np.inf, max_ngbs=100):
        '''
        Find all particles in tree within distance `H` from position `r`.

        Args:
            r (array-like):     Reference position.
            H (array-like):     The smoothing lengthes (support radius)
                                corresponding to the indices of the tree.
            pos (array-like):   The positions corresponding to the indices of the
                                tree.
            periodic (float):   Assume the particles to sit in a periodic cube
                                with this side length.
            max_ngbs (int):     Only return this number of neighbours at maximum.
                                (Not necessarily the closest ones!)

        Returns:
            ngbs (np.ndarray):  List if the indices of the neighbours (in random
                                order). At maximum, though, `max_ngbs` of them.
        '''
        r = np.asarray(r, dtype=np.float64).copy()
        pos = np.asarray(pos, dtype=np.float64)
        if pos.shape[1:] != (3,):
            raise ValueError('Positions have to have shape (N,3)!')
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (len(pos),):
            raise ValueError('Positions have to have shape (N,)!')
        if pos.base is not None:
            pos = pos.copy()
        if H.base is not None:
            H = H.copy()
        max_ngbs = int(max_ngbs)
        periodic = float(periodic)

        ngbs = np.empty(max_ngbs, dtype=np.uintp)
        N_ngbs = c_size_t()
        cpygad.get_octree_ngbs_SPH(self.__node_ptr,
                                   r.ctypes.data, H.ctypes.data,
                                   max_ngbs, ngbs.ctypes.data, byref(N_ngbs),
                                   pos.ctypes.data,
                                   periodic,
                                   0.0,
                                   )
        #ngbs.resize(N_ngbs.value)
        erg = np.resize(ngbs, N_ngbs.value)
        return erg

    def find_next_ngb(self, r, pos, periodic=np.inf, cond=None):
        '''
        Find all particles in tree within distance `H` from position `r`.

        Args:
            r (array-like):     Reference position.
            pos (array-like):   The positions corresponding to the indices of the
                                tree.
            periodic (float):   Assume the particles to sit in a periodic cube
                                with this side length.
            cond (array-like):  Boolean array of a condition to be fulfilled such
                                that a particle is registered as a neighbour. In
                                other words, each particle i for which cond[i] is
                                False is excluded from the neighbours.

        Returns:
            ngb (int):          Index of the next neighbour to position r that
                                fulfills the condition. None if there is no such
                                next neighbour.
        '''
        r = np.asarray(r, dtype=np.float64).copy()
        pos = np.asarray(pos, dtype=np.float64)
        if pos.shape[1:] != (3,):
            raise ValueError('Positions have to have shape (N,3)!')
        if pos.base is not None:
            pos = pos.copy()
        periodic = float(periodic)

        if cond is not None:
            cond = np.asarray(cond, dtype=np.int32)
            if cond.shape != (len(pos),):
                raise ValueError('Unmatching shape of `cond`: %s!' % (cond.shape,))
            if cond.base is not None:
                cond = cond.copy()
            cond = cond.ctypes.data
        ngb = cpygad.get_octree_next_ngb(self.__node_ptr,
                                         r.ctypes.data,
                                         pos.ctypes.data,
                                         periodic,
                                         cond,
                                         )
        if ngb == -1:
            ngb = None

        return ngb

