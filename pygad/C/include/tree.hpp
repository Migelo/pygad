#pragma once
#include "general.hpp"

#include <vector>

extern const int MAX_TREE_LEVEL;
extern const double TREE_NODE_OPEN_TOL;

template<int d>
class Tree {
    public:
        static_assert(0<d, "Dimension has to be positive!");
        static const int dim=d;
        // number of children / maximum number of elements
        static const int NC = pow_int<d>(2);

        Tree();
        Tree(const double center_[d], double side_2_);

        ~Tree();

        const double *center() const {return _center;}
        double center(int i) const {return _center[i];}
        double side_2() const {return _side_2;}
        bool is_leaf() const {return _leaf;}
        size_t tot_part() const {return _tot_part;}
        unsigned num_children() const {return _num_child;}
        const Tree<d> *child(int i) const {
            assert(0<=i and i<(int)_num_child);
            return _child.node[i];
        }
        Tree<d> *child(int i) {
            // avoid code dublication and call const-version of this
            return const_cast<Tree<d> *>(const_cast<const Tree<d> *>(this)->child(i));
        }
        double max_H() const {return _max_H;}

        bool is_in_region(const double pos[d]) const;
        unsigned get_oct(const double pos[d]) const;
        void get_oct_center(unsigned oct, double center[d]) const;

        void add_point(const double *pos, size_t idx, int depth=0);
        void fill_max_H(const double *H);
        void fill_max_H(double H);

        size_t count_nodes(bool count_non_leaves=true) const;
        size_t count_particles() const;
        int get_max_depth() const;

        template<typename F>
        std::vector<size_t> ngbs_within_if(const double r[d], double H,
                                           const double *pos,
                                           const double periodic,
                                           F cond) const;
        std::vector<size_t> ngbs_within(const double r[d], double H,
                                        const double *pos,
                                        const double periodic) const {
            return ngbs_within_if(r, H, pos, periodic, [](size_t i){return true;});
        }
        std::vector<size_t> ngbs_SPH(const double r[d], const double *H,
                                     const double *pos,
                                     const double periodic,
                                     const double tol) const;
        template<typename F>
        std::pair<size_t,double> next_ngb_with(const double r[d],
                                               const double *pos,
                                               const double periodic,
                                               F cond) const;

    private:
        double _center[d];
        double _side_2;
        bool _leaf;
        size_t _tot_part;
        unsigned _num_child;
        double _max_H;
        union {
            size_t idx[NC];
            Tree<d> *node[NC];
        } _child;
};

extern "C" void *new_octree_uninitialized();
extern "C" void *new_octree(const double center_[3], double side_2_);
extern "C" void *new_octree_from_pos(size_t N, const double *const pos);
extern "C" void free_octree(void *const octree);
extern "C" void fill_octree(void *const octree, size_t N, const double *const pos);
extern "C" void update_octree_max_H(void *const octree, const double *const H);
extern "C" void update_octree_const_max_H(void *const octree, double H);
extern "C" void get_octree_center(const void *const octree, double center[3]);
extern "C" double get_octree_side_2(const void *const octree);
extern "C" int get_octree_is_leaf(const void *const octree);
extern "C" unsigned get_octree_num_children(const void *const octree);
extern "C" size_t get_octree_tot_part(const void *const octree);
extern "C" double get_octree_max_H(const void *const octree);
extern "C" size_t get_octree_max_depth(const void *const octree);
extern "C" size_t get_octree_node_count(const void *const octree, int count_non_leaves);
extern "C" int get_octree_in_region(const void *const octree, const double r[3]);
extern "C" void *get_octree_child(void *const octree, int i);
extern "C" unsigned get_octree_octant(void *const octree, const double r[3]);
extern "C" void get_octree_ngbs_within(void *const octree,
                                       const double r[3], double H,
                                       size_t max_ngbs, size_t *ngbs, size_t *N_ngbs,
                                       const double *const pos,
                                       const double periodic,
                                       const int32_t *cond);
extern "C" void get_octree_ngbs_SPH(void *const octree,
                                    const double r[3], const double *const H,
                                    size_t max_ngbs, size_t *ngbs, size_t *N_ngbs,
                                    const double *const pos,
                                    const double periodic,
                                    const double tol);
extern "C" size_t get_octree_next_ngb(void *const octree,
                                      const double r[3],
                                      const double *const pos,
                                      const double periodic,
                                      const int32_t *cond);


template<int d>
Tree<d>::Tree()
    : _center(), _side_2(), _leaf(true), _tot_part(0), _num_child(0), _max_H(), _child()
{
}

template<int d>
Tree<d>::Tree(const double center_[d], double side_2_)
    : _center(), _side_2(side_2_), _leaf(true), _tot_part(0), _num_child(0),
      _max_H(), _child()
{
    for (int i=0; i<d; i++)
        _center[i] = center_[i];
}

template<int d>
Tree<d>::~Tree() {
    if (not _leaf) {
        for (int i=0; i<NC; i++)
            delete _child.node[i];
    }
}

template<int d>
bool Tree<d>::is_in_region(const double pos[d]) const {
    double max_d = std::abs(pos[0]-_center[0]);
    for (int i=1; i<d; i++)
        max_d = std::max<double>(max_d, std::abs(pos[i]-_center[i]));
    return max_d <= _side_2;
}

/*
 * for each dimension i:
 *      boolean: pos[i] > center[i]
 * encode in binary: lowest bit is lowest dimension
 */
template<int d>
unsigned Tree<d>::get_oct(const double pos[d]) const {
    unsigned oct = 0u;
    for (int i=0; i<d; i++) {
        oct += (pos[i] > _center[i]) << i;
    }
    return oct;
}

template<int d>
void Tree<d>::get_oct_center(unsigned oct, double center[d]) const {
    for (int i=0; i<d; i++)
        center[i] = _center[i];
    double off = _side_2 / 2.0;
    for (int i=0; i<d; i++) {
        if ((oct >> i) & 1u)
            center[i] += off;
        else
            center[i] -= off;
    }
}

template<int d>
void Tree<d>::add_point(const double *pos, size_t idx, int depth) {
    if (_leaf) {
        if (_num_child < NC) {
            _child.idx[_num_child++] = idx;
        } else {
            size_t idcs[NC];
            std::memcpy(idcs, _child.idx, sizeof(idcs));
            for (int i=0; i<NC; i++)
                _child.node[i] = NULL;
            _num_child = 0;
            _leaf = false;
            _tot_part = 0;
            for (const auto j : idcs) {
                add_point(pos, j, depth);
            }
            add_point(pos, idx, depth);
            _tot_part--;    // would otherwise be counted twice!
        }
    } else {
        unsigned oct;
        if (depth < MAX_TREE_LEVEL) {
            oct = get_oct(&pos[d*idx]);
        } else {
            oct = 0;    // in case all the child nodes are still NULL
            size_t min = -1;
            for (int i=0; i<NC; i++) {
                Tree<d> *node = _child.node[i];
                if (not node) {
                    min = 0;
                    oct = i;
                } else if ((node->_tot_part)<min) {
                    min = node->_tot_part;
                    oct = i;
                }
            }
        }
        Tree<d> *child = _child.node[oct];
        if (not child) {
            double oct_center[d];
            get_oct_center(oct, oct_center);
            child = _child.node[oct] = new Tree<d>(oct_center, _side_2/2.0);
            if (not child) {
                fprintf(stderr, "not enough memmory!!!\n");
                exit(-1);
            }
            _num_child++;
        }
        child->add_point(pos, idx, depth+1);
    }
    _tot_part++;
    return;
}

template<int d>
void Tree<d>::fill_max_H(const double *H) {
    _max_H = 0.0;
    if (_leaf) {
        for (unsigned i=0; i<_num_child; i++)
            _max_H = std::max(_max_H, H[_child.idx[i]]);
    } else {
        for (unsigned i=0; i<NC; i++) {
            Tree<d> *node = _child.node[i];
            if (node) {
                node->fill_max_H(H);
                _max_H = std::max(_max_H, node->_max_H);
            }
        }
    }
}

template<int d>
void Tree<d>::fill_max_H(double H) {
    _max_H = H;
    if (not _leaf) {
        for (unsigned i=0; i<NC; i++) {
            Tree<d> *node = _child.node[i];
            if (node)
                node->fill_max_H(H);
        }
    }
}

template<int d>
size_t Tree<d>::count_nodes(bool count_non_leaves) const {
    if (_leaf)
        return 1;
    size_t nodes = 0;
    for (unsigned i=0; i<NC; i++) {
        const auto node = _child.node[i];
        if (node)
            nodes += node->count_nodes(count_non_leaves);
    }
    if (count_non_leaves)
        nodes++;    // counting this (non-leaf!) node itself
    return nodes;
}

template<int d>
size_t Tree<d>::count_particles() const {
    if (_leaf)
        return _num_child;
    size_t parts = 0;
    for (unsigned i=0; i<NC; i++) {
        const auto node = _child.node[i];
        if (node)
            parts += node->count_particles();
    }
    return parts;
}

template<int d>
int Tree<d>::get_max_depth() const {
    if (_leaf)
        return 0;
    int max_depth = 0;
    for (unsigned i=0; i<NC; i++) {
        const auto node = _child.node[i];
        if (node)
            max_depth = std::max(max_depth, node->get_max_depth()+1);
    }
    return max_depth;
}

template<int d>
template<typename F>
std::vector<size_t> Tree<d>::ngbs_within_if(const double r[d], double H,
                                            const double *pos,
                                            const double periodic,
                                            F cond) const {
    std::vector<size_t> ngb_idx;
    if (_leaf) {
        for (unsigned i=0; i<_num_child; i++) {
            size_t idx = _child.idx[i];
            if (dist2_periodic<d>(r,&pos[d*idx],periodic) < H*H and cond(idx)) {
                ngb_idx.push_back(idx);
            }
        }
    } else {
        double side_2_H = _side_2/2.0 + H;  // the same for all children
        for (const auto node : _child.node) {
            if (node) {
                double max_d = dist_max_periodic<d>(r, node->_center, periodic);
                if (TREE_NODE_OPEN_TOL*max_d < side_2_H) { // open node and add neighbors therein
                    std::vector<size_t> node_ngbs = node->ngbs_within_if(r, H, pos, periodic, cond);
                    ngb_idx.insert(ngb_idx.end(), node_ngbs.begin(), node_ngbs.end());
                }
            }
        }
    }
    return ngb_idx;
}

template<int d>
std::vector<size_t> Tree<d>::ngbs_SPH(const double r[d], const double *H,
                                      const double *pos,
                                      const double periodic,
                                      const double tol) const {
    std::vector<size_t> ngb_idx;
    if (_leaf) {
        for (unsigned i=0; i<_num_child; i++) {
            size_t idx = _child.idx[i];
            double Hi = H[idx] + tol;
            if (dist2_periodic<d>(r,&pos[d*idx],periodic) < Hi*Hi)
                ngb_idx.push_back(idx);
        }
    } else {
        for (const auto node : _child.node) {
            if (node) {
                double max_d = dist_max_periodic<d>(r, node->_center, periodic);
                if (TREE_NODE_OPEN_TOL*max_d < node->_side_2 + node->_max_H + tol) {
                    std::vector<size_t> node_ngbs = node->ngbs_SPH(r, H, pos, periodic, tol);
                    ngb_idx.insert(ngb_idx.end(), node_ngbs.begin(), node_ngbs.end());
                }
            }
        }
    }
    return ngb_idx;
}

template<int d>
template<typename F>
std::pair<size_t,double> Tree<d>::next_ngb_with(const double r[d],
                                                const double *pos,
                                                const double periodic,
                                                F cond) const {
    std::pair<size_t,double> ngb = {-1, periodic};
    if (_leaf) {
        ngb.second = std::pow(ngb.second,2);
        for (unsigned i=0; i<_num_child; i++) {
            size_t idx = _child.idx[i];
            double d2 = dist2_periodic<d>(r,&pos[d*idx],periodic);
            if (d2 < ngb.second and cond(idx)) {
                ngb.first = idx;
                ngb.second = d2;
            }
        }
        ngb.second = std::sqrt(ngb.second);
    } else {
        for (const auto node : _child.node) {
            if (node) {
                double max_d = dist_max_periodic<d>(r, node->_center, periodic);
                if (max_d - node->_side_2 < ngb.second) {
                    std::pair<size_t,double> ngb_oct;
                    ngb_oct = node->next_ngb_with(r, pos, periodic, cond);
                    if (ngb_oct.second < ngb.second)
                        ngb = ngb_oct;
                }
            }
        }
    }
    return ngb;
}

