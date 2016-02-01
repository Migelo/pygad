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
extern "C" void fill_octree(void *const octree, size_t N, const double *const pos) {
    Tree<3> *tree = (Tree<3> *)octree;
    for (size_t i=0; i<N; i++)
        tree->add_point(pos, i);
}
extern "C" void update_octree_max_H(void *const octree, size_t N, const double *const H) {
    Tree<3> *tree = (Tree<3> *)octree;
    if (H)
        tree->fill_max_H(H);
    else
        tree->fill_max_H_zero();
}
extern "C" void get_octree_ngbs_within(void *const octree,
                                       const double r[3], double H,
                                       size_t max_ngbs, size_t *ngbs, size_t *N_ngbs,
                                       const double *const pos,
                                       const double periodic) {
    const Tree<3> *tree = (const Tree<3> *)octree;
    std::vector<size_t> v_ngbs = tree->ngbs_within(r, H, pos, periodic);
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


