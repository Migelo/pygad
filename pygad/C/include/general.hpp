#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cassert>

#pragma once

#define DIM     3

template <typename T>
constexpr T pow_int(T b, int p) {
    if (p == 0)
        return T(1);
    else if (p < 0)
        return T(1) / pow_int(b,-p);
    return b * pow_int(b,p-1);
};

void argsort(size_t N, const double *x, size_t *idcs);

inline double dist_periodic_1D(double x1, double x2, double P) {
    double d = fabs(x1-x2);
    return fmin(d, fabs(P-d));
}

template <int d>
double dist2_periodic(const double x1[d], const double x2[d], double P) {
    double d2 = 0.0;
    for (int i=0; i<d; i++) {
        double di = dist_periodic_1D(x1[i], x2[i], P);
        d2 += di*di;
    }
    return d2;
}

template <int d>
double dist_periodic(const double x1[d], const double x2[d], double P) {
    return sqrt(dist2_periodic<d>(x1,x2,P));
}


template <int d>
double dist2(const double x1[d], const double x2[d]) {
    double d2 = 0.0;
    for (int i=0; i<d; i++) {
        double di = x1[i] - x2[i];
        d2 += di*di;
    }
    return d2;
}

template <int d>
double dist(const double x1[d], const double x2[d]) {
    return sqrt(dist2<d>(x1,x2));
}

template <int d>
double dist_max_periodic(const double x1[d], const double x2[d], double P) {
    double dist = 0.0;
    for (int i=0; i<d; i++) {
        double di = dist_periodic_1D(x1[i], x2[i], P);
        dist = std::max(di,dist);
    }
    return dist;
}

template <int d>
double dist_max(const double x1[d], const double x2[d]) {
    double dist = 0.0;
    for (int i=0; i<d; i++) {
        double di = fabs(x1[i] - x2[i]);
        dist = std::max(di,dist);
    }
    return dist;
}


template <int d>
double norm(const double x[d]) {
    double norm = 0.0;
    for (int i=0; i<d; i++)
        norm += x[i]*x[i];
    return sqrt(norm);
}

