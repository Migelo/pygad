#pragma once
#include "general.hpp"
#include "tree.hpp"

extern "C"
void find_fof_groups(size_t N,
                     double *pos,
                     double *vel,
                     double *mass,
                     double l,
                     double dvmax,
                     size_t min_parts,
                     int sort,
                     size_t *FoF,
                     double periodic,
                     void *octree=NULL);

