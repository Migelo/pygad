#pragma once
#include "general.hpp"
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
                 void *octree=NULL);

