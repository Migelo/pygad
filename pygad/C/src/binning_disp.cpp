#include "binning_disp.hpp"

template <typename T>
struct mean_with_weights {
    T operator()(const std::vector<T> &X, const std::vector<T> &w) const { return mean<T>(X,w); }
};

template <typename T>
struct median_with_weights {
    T operator()(const std::vector<T> &X, const std::vector<T> &w) const { return median<T>(X,w); }
};

template <typename T>
struct stddev_with_weights {
    T operator()(const std::vector<T> &X, const std::vector<T> &w) const { return stddev<T>(X,w); }
};

void bin_sph_proj_mean(size_t N,
                       double *pos,
                       double *hsml,
                       double *dV,
                       double *qty,
                       double *av,
                       double *extent,
                       size_t Npx[2],
                       double *grid,
                       const char *kernel_,
                       double periodic) {
    bin_sph_proj_by_particle(N, pos, hsml, dV, qty, av,
                             extent, Npx, grid, kernel_, periodic,
                             mean_with_weights<double>(), 0.0);
}

void bin_sph_proj_median(size_t N,
                         double *pos,
                         double *hsml,
                         double *dV,
                         double *qty,
                         double *av,
                         double *extent,
                         size_t Npx[2],
                         double *grid,
                         const char *kernel_,
                         double periodic) {
    bin_sph_proj_by_particle(N, pos, hsml, dV, qty, av,
                             extent, Npx, grid, kernel_, periodic,
                             median_with_weights<double>(), 0.0);
}

void bin_sph_proj_stddev(size_t N,
                         double *pos,
                         double *hsml,
                         double *dV,
                         double *qty,
                         double *av,
                         double *extent,
                         size_t Npx[2],
                         double *grid,
                         const char *kernel_,
                         double periodic) {
    bin_sph_proj_by_particle(N, pos, hsml, dV, qty, av,
                             extent, Npx, grid, kernel_, periodic,
                             stddev_with_weights<double>(), 0.0);
}

