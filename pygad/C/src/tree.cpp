#include "tree.hpp"

// 2**10 =  1e3 side length ratio
// 2**15 = 33e3 side length ratio
// 2**20 =  1e6 side length ratio
// 2**25 = 34e6 side length ratio
// 2**30 =  1e9 side length ratio
const int MAX_TREE_LEVEL = 25;
const double TREE_NODE_OPEN_TOL = 1.0001;

template class Tree<2>;
template class Tree<3>;

extern "C" void *new_octree_uninitialized() {
    return new Tree<3>();
}
extern "C" void *new_octree(const double center_[3], double side_2_) {
    return new Tree<3>(center_, side_2_);
}
extern "C" void *new_octree_from_pos(size_t N, const double *const pos) {
    // find extent of positions
    double min[3], max[3];
    for (int k=0; k<3; k++) {
        min[k] = pos[k];
        max[k] = pos[k];
    }
    for (size_t j=1; j<N; j++) {
        for (size_t k=0; k<3; k++) {
            if (min[k] > pos[3*j+k])
                min[k] = pos[3*j+k];
            if (max[k] < pos[3*j+k])
                max[k] = pos[3*j+k];
        }
    }

    // calculate center and side length
    double center[3];
    double side_2 = 0.0;
    for (int k=0; k<3; k++) {
        center[k] = (min[k]+max[k]) / 2.0;
        side_2 = fmax(side_2, (max[k]-min[k])/2.0);
    }

    // build tree
    Tree<3> *tree = new Tree<3>(center, side_2);
    assert(tree);
    for (size_t j=0; j<N; j++)
        tree->add_point(pos, j);

    return tree;
}
extern "C" void fill_octree(void *const octree, size_t N, const double *const pos) {
    Tree<3> *tree = (Tree<3> *)octree;
    for (size_t i=0; i<N; i++)
        tree->add_point(pos, i);
}
extern "C" void update_octree_max_H(void *const octree, const double *const H) {
    Tree<3> *tree = (Tree<3> *)octree;
    tree->fill_max_H(H);
}
extern "C" void update_octree_const_max_H(void *const octree, double H) {
    Tree<3> *tree = (Tree<3> *)octree;
    tree->fill_max_H(H);
}
extern "C" void free_octree(void *const octree) {
    delete (Tree<3> *const)octree;
}
extern "C" void get_octree_center(const void *const octree, double center[3]) {
    const Tree<3> *tree = (const Tree<3> *)octree;
    for (int i=0; i<3; i++)
        center[i] = tree->center(i);
}
extern "C" double get_octree_side_2(const void *const octree) {
    return ((const Tree<3> *)octree)->side_2();
}
extern "C" int get_octree_is_leaf(const void *const octree) {
    return ((const Tree<3> *)octree)->is_leaf();
}
extern "C" unsigned get_octree_num_children(const void *const octree) {
    return ((const Tree<3> *)octree)->num_children();
}
extern "C" size_t get_octree_tot_part(const void *const octree) {
    return ((const Tree<3> *)octree)->tot_part();
}
extern "C" double get_octree_max_H(const void *const octree) {
    return ((const Tree<3> *)octree)->max_H();
}
extern "C" size_t get_octree_max_depth(const void *const octree) {
    return ((const Tree<3> *)octree)->get_max_depth();
}
extern "C" size_t get_octree_node_count(const void *const octree, int count_non_leaves) {
    return ((const Tree<3> *)octree)->count_nodes(count_non_leaves);
}
extern "C" int get_octree_in_region(const void *const octree, const double r[3]) {
    return ((const Tree<3> *)octree)->is_in_region(r);
}
extern "C" void *get_octree_child(void *const octree, int i) {
    Tree<3> *const tree = (Tree<3> *)octree;
    return tree->child(i);
}
extern "C" unsigned get_octree_octant(void *const octree, const double r[3]) {
    Tree<3> *const tree = (Tree<3> *)octree;
    return tree->get_oct(r);
}
extern "C" void get_octree_ngbs_within(void *const octree,
                                       const double r[3], double H,
                                       size_t max_ngbs, size_t *ngbs, size_t *N_ngbs,
                                       const double *const pos,
                                       const double periodic,
                                       const int32_t *cond) {
    const Tree<3> *tree = (const Tree<3> *)octree;
    std::vector<size_t> v_ngbs;
    if (cond) {
        v_ngbs = tree->ngbs_within_if(r, H, pos, periodic,
                                      [&cond](size_t i){return cond[i];});
    } else {
        v_ngbs = tree->ngbs_within(r, H, pos, periodic);
    }
    *N_ngbs = std::min(v_ngbs.size(), max_ngbs);
    for (size_t i=0; i<*N_ngbs; i++) {
        ngbs[i] = v_ngbs[i];
    }
}
extern "C" void get_octree_ngbs_SPH(void *const octree,
                                    const double r[3], const double *const H,
                                    size_t max_ngbs, size_t *ngbs, size_t *N_ngbs,
                                    const double *const pos,
                                    const double periodic,
                                    const double tol) {
    const Tree<3> *tree = (const Tree<3> *)octree;
    std::vector<size_t> v_ngbs = tree->ngbs_SPH(r, H, pos, periodic, tol);
    *N_ngbs = std::min(v_ngbs.size(), max_ngbs);
    for (size_t i=0; i<*N_ngbs; i++) {
        ngbs[i] = v_ngbs[i];
    }
}
extern "C" size_t get_octree_next_ngb(void *const octree,
                                      const double r[3],
                                      const double *const pos,
                                      const double periodic,
                                      const int32_t *cond) {
    const Tree<3> *tree = (const Tree<3> *)octree;
    std::pair<size_t,double> ngb;
    if (cond) {
        ngb = tree->next_ngb_with(r, pos, periodic,
                                  [&cond](size_t i){return cond[i];});
    } else {
        ngb = tree->next_ngb_with(r, pos, periodic, [](size_t i){return true;});
    }
    return ngb.first;
}

