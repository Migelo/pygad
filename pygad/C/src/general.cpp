#include "general.h"

double dist2_periodic(const double *x1, const double *x2, double P) {
    double d2 = 0.0;
    for (int i=0; i<DIM; i++) {
        double di = fmin(fabs(fmod(x1[i] - x2[i], P)),
                         fabs(fmod(x2[i] - x1[i], P)));
        d2 += di*di;
    }
    return d2;
}

double dist_periodic(const double *x1, const double *x2, double P) {
    return sqrt(dist2_periodic(x1,x2,P));
}

double dist2(const double *x1, const double *x2) {
    double d2 = 0.0;
    for (int i=0; i<DIM; i++) {
        double di = x1[i] - x2[i];
        d2 += di*di;
    }
    return d2;
}

double dist(const double *x1, const double *x2) {
    return sqrt(dist2(x1,x2));
}

double norm(const double *x) {
    double norm = 0.0;
    for (int i=0; i<DIM; i++)
        norm += x[i]*x[i];
    return sqrt(norm);
}

