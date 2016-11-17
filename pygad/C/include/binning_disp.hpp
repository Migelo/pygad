#pragma once
#include "general.hpp"
#include "kernels.hpp"
#include "binning.hpp"
#include "vector_stats.hpp"

extern "C"
void bin_sph_proj_mean(size_t N,
                       double *pos,
                       double *hsml,
                       double *dV,
                       double *qty,
                       double *av,
                       double *extent,
                       size_t Npx[2],
                       double *grid,
                       const char *kernel_,
                       double periodic);

extern "C"
void bin_sph_proj_median(size_t N,
                         double *pos,
                         double *hsml,
                         double *dV,
                         double *qty,
                         double *av,
                         double *extent,
                         size_t Npx[2],
                         double *grid,
                         const char *kernel_,
                         double periodic);

extern "C"
void bin_sph_proj_stddev(size_t N,
                         double *pos,
                         double *hsml,
                         double *dV,
                         double *qty,
                         double *av,
                         double *extent,
                         size_t Npx[2],
                         double *grid,
                         const char *kernel_,
                         double periodic);

template <typename F>
void bin_sph_proj_by_particle(size_t N,
                              double *pos,
                              double *hsml,
                              double *dV,
                              double *qty,
                              double *av,
                              double *extent,
                              size_t Npx[2],
                              double *grid,
                              const char *kernel_,
                              double periodic,
                              F reduction_func,
                              double empty_val) {
    constexpr int d=2;
    Kernel<d+1> &kernel = kernels.at(kernel_);
    kernel.require_table_size(2048,0);

    //printf("initizalize grid...\n");
    size_t Ngrid = Npx[0];
    for (int k=1; k<d; k++)
        Ngrid *= Npx[k];
    // TODO: not neccessary anymore, but check!
    //memset(grid, 0, Ngrid*sizeof(double));
    //assert(grid[Ngrid-1]==0.0);

    // allocate memory for the normation S
    double *S = (double *)std::malloc(N*sizeof(double));
    assert( S != nullptr );

    //auto t_start = std::chrono::high_resolution_clock::now();

    // prepare the binning, initialize
    double res[d];
    for (int k=0; k<d; k++)
        res[k] = (extent[2*k+1]-extent[2*k]) / Npx[k];
    double res_min = res[0];
    for (int k=1; k<d; k++)
        res_min = std::min(res_min, res[k]);
    double dV_px = res[0];
    for (int k=1; k<d; k++)
        dV_px *= res[k];
    // calculate all the normations
#pragma omp parallel for default(shared) schedule(dynamic,10)
    for (size_t j=0; j<N; j++) {
        double *rj = pos+(d*j);
        double hj = hsml[j];

        // Mind the reversed indexing of i_min, i_max, and i due to performace
        // at accessing array elements in reversed loop order in
        // nested_loops<d>::do_loops(...)!
        size_t i_min[d], i_max[d];
        for (int k=0; k<d; k++) {
            i_min[(d-1)-k] = std::max<double>( (rj[k]-extent[2*k]-hj) / res[k],       0.0);
            i_min[(d-1)-k] = std::min(i_min[(d-1)-k], Npx[k]);
            i_max[(d-1)-k] = std::min<size_t>( (rj[k]-extent[2*k]+hj) / res[k] + 1.0, Npx[k]);
        }

        double S_j;
        // no correction for particles that extent out of the grid and the
        // integral is not over the entire kernel
        if (hj > H_lim_out_of_grid*res_min and extents_out_of_grid<d>(i_min, i_max, Npx)) {
            S_j = 1.0;
        } else {
            S_j = 0.0;
            size_t i[d];
            double grid_r[d];
            nested_loops<d>::do_loops(i, i_min, i_max,
                [&](unsigned n, size_t *i, size_t *i_min, size_t *i_max){
                    const unsigned k = (d-1)-n;
                    grid_r[k] = extent[2*k] + (i[n]+0.5)*res[k];
                },
                [&](unsigned n, size_t *i, size_t *i_min, size_t *i_max){
                    const unsigned k = (d-1)-n;
                    grid_r[k] = extent[2*k] + (i[n]+0.5)*res[k];
                    double dj = dist_periodic<d>(grid_r, rj, periodic);
                    S_j += dV_px * kernel.proj_value(dj/hj, hj);
            });
        }

        S[j] = S_j;
    }

    //printf("bin %zu particles (projected) on %dD grid...\n", N, d);
    // Mind the reversed indexing of i_min, i_max, and i due to performace
    // at accessing array elements in reversed loop order in
    // nested_loops<d>::do_loops(...)!
    assert( d == 2 );
#pragma omp parallel for default(shared) schedule(dynamic,10)
    for (size_t i1=0; i1<Npx[0]; i1++) {
        size_t i[d];
        i[1] = i1;
        double grid_r[d];
        grid_r[0] = extent[2*0] + (i[1]+0.5)*res[0];
        for (i[0]=0; i[0]<Npx[1]; i[0]++) {
            grid_r[1] = extent[2*1] + (i[0]+0.5)*res[1];
            size_t I = construct_linear_idx<d>(i, Npx);
            std::vector<double> Qs;
            std::vector<double> weights;

            for (size_t j=0; j<N; j++) {
                double *rj = pos+(d*j);
                double hj = hsml[j];
                double dj = dist_periodic<d>(grid_r, rj, periodic);
                double Qj = qty[j];
                double avj = av[j];
                double Sj = S[j];

                if ( hj < dj )
                    continue;

                if ( Sj<1e-4 ) {
                    size_t i_near[d];
                    for (int k=0; k<d; k++)
                        i_near[(d-1)-k] = (rj[k]-extent[2*k]) / res[k];
                    // dismiss if out of grid
                    if ( out_of_grid<d>(i_near, Npx) )
                        continue;
                    size_t I_near = construct_linear_idx<d>(i_near, Npx);
                    if ( I_near != I )
                        continue;
                    Qs.push_back( Qj );
                    weights.push_back( avj * dV[j] );
                } else {
                    double Wj = kernel.proj_value(dj/hj, hj) / Sj;
                    Qs.push_back( Qj );
                    weights.push_back( avj * Wj );
                }
            }

            if ( not Qs.size() )
                grid[I] = empty_val;
            else
                grid[I] = reduction_func(Qs, weights);
        }
    }

    std::free(S);

    /*
    auto t_end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(t_end-t_start);
    printf("binning (projected) on %dD grid took %.6f s\n", d, diff.count());
    */
}

