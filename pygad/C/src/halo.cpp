#include "halo.hpp"

void virial_info(size_t N, const double *mass, const double *r,
                 double rho_threshold, size_t N_min, double *info) {
    // sort r by index
    size_t *sorted_idcs = (size_t *)malloc(N*sizeof(size_t));
    argsort(N, r, sorted_idcs);
    // go over indices, add up mass and calculate density for each particle
    // once rho_threshold is reached, break and give radius
    double M = 0.0;
    double rho;
    size_t i = 0, ii;
    for (ii=0; ii<N; ii++) {
        i = sorted_idcs[ii];
        M += mass[i];
        rho = M / (4./3.*M_PI * r[i]*r[i]*r[i]);
        if (rho < rho_threshold)
            break;
    }
    free(sorted_idcs);
    if (ii<N_min or ii==N) {
        info[0] = 0.0;
        info[1] = 0.0;
    } else {
        info[0] = r[i];
        info[1] = M;
    }
}


