#include "kernels.h"

double cubic(double u) {
    if (u < 0.5) {
        return 2.5464790894703255 * (1.+6.*(u-1.)*u*u);
    } else {
        double one_u = 1. - u;
        return 5.09295817894 * one_u*one_u*one_u;
    }
}

void cubic_vec(const double *u, size_t N, double *w) {
    for (size_t i=0; i<N; i++)
        w[i] = cubic(u[i]);
}

