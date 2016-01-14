#include <stddef.h>
#include <math.h>

#define DIM     3

extern "C" double dist2_periodic(const double *x1, const double *x2, double P);
extern "C" double dist_periodic(const double *x1, const double *x2, double P);
extern "C" double dist2(const double *x1, const double *x2);
extern "C" double dist(const double *x1, const double *x2);
extern "C" double norm(const double *x);

