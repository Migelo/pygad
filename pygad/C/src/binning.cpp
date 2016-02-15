#include "binning.hpp"

double H_lim_out_of_grid = 0.5;

void sph_bin_3D(size_t N,
                double *pos,
                double *hsml,
                double *dV,
                double *qty,
                double *extent,
                size_t Npx[3],
                double *grid,
                const char *kernel_,
                double periodic) {
    bin_sph<3>(N, pos, hsml, dV, qty, extent, Npx, grid, kernel_, periodic);
}

void sph_3D_bin_2D(size_t N,
                   double *pos,
                   double *hsml,
                   double *dV,
                   double *qty,
                   double *extent,
                   size_t Npx[2],
                   double *grid,
                   const char *kernel_,
                   double periodic) {
    bin_sph<2,true>(N, pos, hsml, dV, qty, extent, Npx, grid, kernel_, periodic);
}

