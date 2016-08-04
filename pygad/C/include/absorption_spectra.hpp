#pragma once
#include "general.hpp"
#include "kernels.hpp"
#include "looploop.hpp"

extern "C"
void absorption_spectrum(bool particles,
                         size_t N,
                         double *pos,
                         double *vel,
                         double *hsml,
                         double *n,
                         double *temp,
                         double *los_pos,
                         double *vel_extent,
                         size_t Nbins,
                         double b_0,
                         double Xsec,
                         double *taus,
                         double *los_dens,
                         double *los_temp,
                         const char *kernel_,
                         double periodic);

