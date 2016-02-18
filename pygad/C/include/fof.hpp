#pragma once
#include "general.hpp"
#include "tree.hpp"

extern "C"
void find_fof_groups(size_t N,
                     double *pos,
                     double *mass,
                     double l,
                     size_t min_parts,
                     int sort,
                     size_t *FoF,
                     double periodic,
                     void *octree=NULL);

