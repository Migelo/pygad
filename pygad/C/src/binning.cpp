#include "binning.hpp"

double H_lim_out_of_grid = 0.3;

// helper funktion templates
template <int d>
bool out_of_grid(const size_t *i, const size_t *Npx) {
    for (int k=0; k<d; k++) {
        if (i[k]<0 or Npx[k]<=i[k])
            return true;
    }
    return false;
}
template <int d>
bool extents_out_of_grid(const size_t *i_min, const size_t *i_max, const size_t *Npx) {
    for (int k=0; k<d; k++) {
        if (i_min[k]==0 or i_max[k]==Npx[k])
            return true;
    }
    return false;
}
template <int d>
inline size_t construct_linear_idx(const size_t *i, const size_t *Npx) {
    size_t I = i[0];
    for (int k=1; k<d; k++)
        I = I*Npx[k] + i[k];
    return I;
}
// end of helper funktion templates

void sph_bin_3D(size_t N,
                double *pos,
                double *hsml,
                double *dV,
                double *qty,
                double *extent,
                size_t Npx[3],
                double *grid,
                const char *kernel_,
                double periodic) {
    //printf("initialze kernel...\n");
    Kernel<3> kernel(kernel_);

    //printf("initizalize grid...\n");
    size_t Ngrid = Npx[0]*Npx[1]*Npx[2];
    memset(grid, 0, Ngrid*sizeof(double));
    assert(grid[Ngrid-1]==0.0);

    //printf("bin %zu particles on grid...\n", N);
    double res[3];
    for (int k=0; k<3; k++)
        res[k] = (extent[2*k+1]-extent[2*k]) / Npx[k];
    double res_min = std::min(std::min(res[0], res[1]), res[2]);
    double dV_px = res[0]*res[1]*res[2];
#pragma omp parallel for default(shared) schedule(dynamic,10)
    for (size_t j=0; j<N; j++) {
        double *rj = pos+(3*j);
        double hj = hsml[j];
        double dVj = dV[j];
        double Qj = qty[j];

        size_t i_min[3], i_max[3];
        for (int k=0; k<3; k++) {
            i_min[k] = std::max<double>( (rj[k]-extent[2*k]-hj) / res[k],       0.0);
            i_min[k] = std::min(i_min[k], Npx[k]);
            i_max[k] = std::min<size_t>( (rj[k]-extent[2*k]+hj) / res[k] + 1.0, Npx[k]);
        }

        double W_int;
        // no correction for particles that extent out of the grid and the
        // integral is not over the entire kernel
        if (hj > H_lim_out_of_grid*res_min and extents_out_of_grid<3>(i_min, i_max, Npx)) {
            W_int = 1.0;
        } else {
            W_int = 0.0;
            size_t i[3];
            for (i[0]=i_min[0]; i[0]<i_max[0]; i[0]++) {
                double grid_r[3];
                grid_r[0] = extent[0] + (i[0]+0.5)*res[0];
                for (i[1]=i_min[1]; i[1]<i_max[1]; i[1]++) {
                    grid_r[1] = extent[2] + (i[1]+0.5)*res[1];
                    for (i[2]=i_min[2]; i[2]<i_max[2]; i[2]++) {
                        grid_r[2] = extent[4] + (i[2]+0.5)*res[2];
                        double dj = dist_periodic<3>(grid_r, rj, periodic);
                        W_int += dV_px * kernel.value(dj/hj, hj);
                    }
                }
            }
        }

        if (W_int<1e-4) {
            size_t i[3];
            for (int k=0; k<3; k++)
                i[k] = (rj[k]-extent[2*k]) / res[k];
            // dismiss if out of grid
            if ( out_of_grid<3>(i, Npx) )
                continue;
            size_t I = construct_linear_idx<3>(i, Npx);
            double VV = dVj / dV_px;
#pragma omp atomic
            grid[I] += VV * Qj;
        } else {
            size_t i[3];
            for (i[0]=i_min[0]; i[0]<i_max[0]; i[0]++) {
                double grid_r[3];
                grid_r[0] = extent[0] + (i[0]+0.5)*res[0];
                for (i[1]=i_min[1]; i[1]<i_max[1]; i[1]++) {
                    grid_r[1] = extent[2] + (i[1]+0.5)*res[1];
                    for (i[2]=i_min[2]; i[2]<i_max[2]; i[2]++) {
                        grid_r[2] = extent[4] + (i[2]+0.5)*res[2];
                        size_t I = construct_linear_idx<3>(i, Npx);
                        double dj = dist_periodic<3>(grid_r, rj, periodic);
                        double dVj_Wj = dVj / W_int * kernel.value(dj/hj, hj);
#pragma omp atomic
                        grid[I] += dVj_Wj * Qj;
                    }
                }
            }
        }
    }

}

void sph_3D_bin_2D(size_t N,
                   double *pos,
                   double *hsml,
                   double *dV,
                   double *qty,
                   double *extent,
                   size_t Npx[2],
                   double *grid,
                   const char *kernel_,
                   double periodic) {
    //printf("initialze kernel...\n");
    Kernel<3> kernel(kernel_);
    kernel.generate_projection(1024);

    //printf("initizalize grid...\n");
    size_t Ngrid = Npx[0]*Npx[1];
    memset(grid, 0, Ngrid*sizeof(double));
    assert(grid[Ngrid-1]==0.0);

    //printf("bin %zu particles on grid...\n", N);
    double res[2];
    for (int k=0; k<2; k++)
        res[k] = (extent[2*k+1]-extent[2*k]) / Npx[k];
    double res_min = std::min(res[0], res[1]);
    double dA_px = res[0]*res[1];
#pragma omp parallel for default(shared) schedule(dynamic,10)
    for (size_t j=0; j<N; j++) {
        double *rj = pos+(2*j);
        double hj = hsml[j];
        double dVj = dV[j];
        double Qj = qty[j];

        size_t i_min[2], i_max[2];
        for (int k=0; k<2; k++) {
            i_min[k] = std::max<double>( (rj[k]-extent[2*k]-hj) / res[k],       0.0);
            i_min[k] = std::min(i_min[k], Npx[k]);
            i_max[k] = std::min<size_t>( (rj[k]-extent[2*k]+hj) / res[k] + 1.0, Npx[k]);
        }

        double W_int;
        // no correction for particles that extent out of the grid and the
        // integral is not over the entire kernel
        if (hj > H_lim_out_of_grid*res_min and extents_out_of_grid<2>(i_min, i_max, Npx)) {
            W_int = 1.0;
        } else {
            W_int = 0.0;
            size_t i[2];
            for (i[0]=i_min[0]; i[0]<i_max[0]; i[0]++) {
                double grid_r[2];
                grid_r[0] = extent[0] + (i[0]+0.5)*res[0];
                for (i[1]=i_min[1]; i[1]<i_max[1]; i[1]++) {
                    grid_r[1] = extent[2] + (i[1]+0.5)*res[1];
                    double dj = dist_periodic<2>(grid_r, rj, periodic);
                    W_int += dA_px * kernel.proj_value(dj/hj, hj);
                }
            }
        }

        if (W_int<1e-4) {
            size_t i[2];
            for (int k=0; k<2; k++)
                i[k] = (rj[k]-extent[2*k]) / res[k];
            // dismiss if out of grid
            if ( out_of_grid<2>(i, Npx) )
                continue;
            size_t I = construct_linear_idx<2>(i, Npx);
            double VV = dVj / dA_px;
#pragma omp atomic
            grid[I] += VV * Qj;
        } else {
            size_t i[2];
            for (i[0]=i_min[0]; i[0]<i_max[0]; i[0]++) {
                double grid_r[2];
                grid_r[0] = extent[0] + (i[0]+0.5)*res[0];
                for (i[1]=i_min[1]; i[1]<i_max[1]; i[1]++) {
                    grid_r[1] = extent[2] + (i[1]+0.5)*res[1];
                    size_t I = construct_linear_idx<2>(i, Npx);
                    double dj = dist_periodic<2>(grid_r, rj, periodic);
                    double dVj_Wj = dVj / W_int * kernel.proj_value(dj/hj, hj);
#pragma omp atomic
                    grid[I] += dVj_Wj * Qj;
                }
            }
        }
    }
}

