#pragma once
#include "general.hpp"

template<bool periodic>
void shrinking_sphere(double *center,
                      size_t N, const double *pos, const double *mass,
                      const double center0[3], double R0,
                      double shrink_factor, size_t stop_N,
                      double boxsize);

extern "C"
void shrinking_sphere_periodic(
                      double *center,
                      size_t N, const double *pos, const double *mass,
                      const double center0[3], double R0,
                      double shrink_factor, size_t stop_N,
                      double boxsize) {
    shrinking_sphere<true>(center, N, pos, mass, center0, R0, shrink_factor, stop_N, boxsize);
}
extern "C"
void shrinking_sphere_nonperiodic(
                      double *center,
                      size_t N, const double *pos, const double *mass,
                      const double center0[3], double R0,
                      double shrink_factor, size_t stop_N) {
    shrinking_sphere<false>(center, N, pos, mass, center0, R0, shrink_factor, stop_N, 0.0);
}

extern "C"
void virial_info(size_t N, const double *mass, const double *r,
                 double rho_threshold, size_t N_min, double *info);


template<bool periodic>
void shrinking_sphere(double *center,
                      size_t N, const double *pos, const double *mass,
                      const double *center0, double R0,
                      double shrink_factor, size_t stop_N,
                      double boxsize) {
    double R2 = R0*R0;
    shrink_factor = shrink_factor*shrink_factor;
    for (int k=0; k<DIM; k++)
        center[k] = center0[k];
    size_t Nidcs = N;
    size_t *idcs = (size_t *)malloc(Nidcs*sizeof(size_t));
    size_t *next_idcs = (size_t *)malloc(Nidcs*sizeof(size_t));
    for (size_t i=0; i<N; i++)
        idcs[i] = i;
    size_t N_left = stop_N+1;
    while (N_left > stop_N) {
        double com[DIM];
        for (int k=0; k<DIM; k++)
            com[k] = 0.0;
        double M = 0.0;
        N_left = 0;
        for (size_t ii=0; ii<Nidcs; ii++) {
            size_t i = idcs[ii];
            const double *r = pos + (DIM*i);
            double d2 = periodic ? dist2_periodic<DIM>(r, center, boxsize)
                                 : dist2<DIM>(r, center);
            if (d2 < R2) {
                for (int k=0; k<DIM; k++)
                    com[k] += mass[i] * r[k];
                M += mass[i];
                next_idcs[N_left] = idcs[ii];
                N_left++;
            }
        }
        if (N_left == 0)
            break;
        Nidcs = N_left;
        size_t *tmp = idcs;
        idcs = next_idcs;
        next_idcs = tmp;

        for (int k=0; k<DIM; k++)
            center[k] = com[k] / M;
        R2 *= shrink_factor;
    }
    free(idcs);
    free(next_idcs);
}

