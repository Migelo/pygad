#pragma once
#include "general.hpp"
#include "kernels.hpp"
#include "looploop.hpp"

extern "C"
void absorption_spectrum(bool particles,
                         size_t N,
                         double *pos,
                         double *vel,
			 double *vpec_z, // DS LOS peculiar velocity
                         double *hsml,
                         double *n,
                         double *temp,
			 double *rho, // DS: density
			 double *metal_frac, // SA: metal mass fraction
                         double *los_pos,
                         double *vel_extent,
                         size_t Nbins,
                         double b_0,
                         double *v_turb,
                         double Xsec,
                         double Gamma,
                         double *taus,
                         double *los_dens,
			 double *los_dens_phys, // DS: density field
			 double *los_metal_frac, // SA: LOS metal mass fraction
                         double *los_temp,
			 double *los_vpec, // DS LOS peculiar velocity field
                         double *v_lims,
                         double *column,
                         const char *kernel_,
                         double periodic);

