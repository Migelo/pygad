#include "binning.hpp"
#include "kernels.hpp"
#include "tree.hpp"

extern "C"
void eval_sph_at(size_t M,
                 double *r,
                 double *vals,
                 size_t N,
                 double *pos,
                 double *hsml,
                 double *dV,
                 double *qty,
                 const char *kernel_,
                 void *octree) {
    double periodic = INFINITY;

    //printf("initialze kernel...\n");
    Kernel<3> kernel(kernel_);

    Tree<3> *tree = NULL;
    if (octree == NULL) {
        //printf("initizalize tree...\n");
        tree = (Tree<3> *)new_octree_from_pos(N, pos);
        assert(tree);
        tree->fill_max_H(hsml);
    } else {
        tree = (Tree<3> *)octree;
    }

    //printf("calculate SPH property from %zu particles at %zu positions...\n", N, M);
#pragma omp parallel for shared(r, vals, pos, hsml, dV, qty, tree)
    for (size_t i=0; i<M; i++) {
        double *ri = r+(3*i);

        std::vector<size_t> ngbs = tree->ngbs_SPH(ri, hsml, pos, periodic, 0.0);

        vals[i] = 0.0;
        for (const size_t j : ngbs) {
            double dj = dist_periodic<3>(ri, pos+(3*j), periodic);
            double hj = hsml[j];
            double dVj_Wj = dV[j] * kernel.value_ql1(dj/hj, hj);

            vals[i] += dVj_Wj * qty[j];
        }
    }

    if (octree == NULL) {
        //printf("delete tree...\n");
        delete tree;
    }
}

