#include "absorption_spectra.hpp"

static inline bool in_lims(double v, double *v_lims) {
    return ( v_lims[0] <= v ) and ( v <= v_lims[1] );
}

template <bool particles>
void _absorption_spectrum(size_t N,
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
                          double *v_lims,
                          double *kernel_int,
                          const char *kernel_,
                          double periodic) {
    double dv = (vel_extent[1] - vel_extent[0]) / Nbins;
    Kernel<3> &kernel = kernels.at(kernel_);
    if ( particles ) {
        kernel.require_table_size(2048,0);
    } else {
        kernel.require_table_size(0,1024);
    }

    std::memset(taus, 0, Nbins*sizeof(double));
    std::memset(los_dens, 0, Nbins*sizeof(double));
    std::memset(los_temp, 0, Nbins*sizeof(double));

#pragma omp parallel for default(shared) schedule(dynamic,10)
    for (size_t j=0; j<N; j++) {
        kernel_int[j] = 0.0;    // for proper values when skipping
        double Nj = n[j];

        if ( particles ) {
            double *rj = pos+(2*j);
            double hj = hsml[j];
            // calculate the projected distance of the particle to the l.o.s.
            double dj = dist_periodic<2>(los_pos, rj, periodic);
            // skip particles that are too far away
            if ( dj > hj )
                continue;
            double Wj = kernel.proj_value(dj/hj, hj);
            Nj *= Wj;
            kernel_int[j] = Wj;
        }

        // column density of the particles / cells along the line of sight
        double vj = vel[j];
        double Tj = temp[j];

        // get the (middle) velocity bin index
        double vi = (vj - vel_extent[0]) / dv;

        // thermal broadening tb_b(v) = 1/(b*sqrt(pi)) * exp( -(v/b)^2 )
        double b = b_0 * std::sqrt(Tj);
        constexpr double int_width = 8.0;   // go out to this times the rms
        if ( vj+int_width*b < vel_extent[0] or vj-int_width*b > vel_extent[1] ) {
            kernel_int[j] = 0.0;
            continue;   // out of bounds -- don't bin into bin #0 or #Nbins-1
        }
        size_t vi_min = std::max<double>(0.0,     std::floor((vj-int_width*b - vel_extent[0]) / dv));
        size_t vi_max = std::min<double>(Nbins-1, std::ceil( (vj+int_width*b - vel_extent[0]) / dv));

        if ( vi_min == vi_max ) {
            assert( std::abs(vi_min - vi) < 0.501 );
#pragma omp atomic
            taus[vi_min] += Nj;
#pragma omp atomic
            los_dens[vi_min] += Nj;
#pragma omp atomic
            los_temp[vi_min] += Tj * Nj;

            if( not in_lims(vj,v_lims) )
                kernel_int[j] = 0.0;
        } else {
            double contrib_lim = 0.0;
            //double contrib_total = 0.0;
            // antiderivative of tb_b: int_0_v dv' tb_b(v') = 1/2 * erf(v/b)
            for ( size_t i=vi_min; i<=vi_max; i++ ) {
                double Dtb;
                double Dv = (i-vi) * dv;
                if ( b < 10.*dv ) {
                    // standard deviation gets comparable with the bin size, do
                    // proper integrals of the line-profile over the bins
                    double v0 = (i-vi-0.5) * dv;
                    double v1 = (i-vi+0.5) * dv;
                    Dtb = 0.5 * (std::erf(v1/b) - std::erf(v0/b));
                } else {
                    // approximate the line as constant over the bin
                    Dtb = std::exp(-std::pow(Dv/b,2.0)) / (b * std::sqrt(M_PI)) * dv;
                }
                double DtbNj = Dtb * Nj;
                // TODO: addtional loop for the Lorentz profile
#pragma omp atomic
                taus[i] += DtbNj;
#pragma omp atomic
                los_dens[i] += DtbNj;
#pragma omp atomic
                los_temp[i] += Tj * DtbNj;

                //contrib_total += Dtb;
                if ( in_lims(vj+Dv,v_lims) )
                    contrib_lim += Dtb;
            }

            kernel_int[j] *= contrib_lim; // / contrib_total;
        }
    }

    for (size_t i=0; i<Nbins; i++) {
        taus[i] *= Xsec / dv;

        if ( los_dens[i] != 0.0 ) {
            los_temp[i] /= los_dens[i];
        }
    }
}

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
                         double *v_lims,
                         double *kernel_int,
                         const char *kernel_,
                         double periodic) {
    if ( particles ) {
        return _absorption_spectrum<true>(N, pos, vel, hsml, n, temp,
                                          los_pos, vel_extent, Nbins,
                                          b_0, Xsec,
                                          taus, los_dens, los_temp,
                                          v_lims, kernel_int,
                                          kernel_, periodic);
    } else {
        return _absorption_spectrum<false>(N, pos, vel, hsml, n, temp,
                                           los_pos, vel_extent, Nbins,
                                           b_0, Xsec,
                                           taus, los_dens, los_temp,
                                           v_lims, kernel_int,
                                           kernel_, periodic);
    }
}

