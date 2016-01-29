#include "general.hpp"

#include <algorithm>

void argsort(size_t N, const double *x, size_t *idcs) {
    for (size_t i=0; i<N; i++)
        idcs[i] = i;
    std::sort(idcs, idcs+N,
              [&](size_t a, size_t b) {return x[a]<x[b];} );
}

