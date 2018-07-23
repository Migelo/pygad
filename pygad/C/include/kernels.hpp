#pragma once
#include "general.hpp"

#include <gsl/gsl_integration.h>
#include <vector>

enum KernelType {
    UNDEFINED_KERNEL,

    CUBIC_SPLINE,
    QUARTIC_SPLINE,
    QUINTIC_SPLINE,
    WENDLAND_C2,
    WENDLAND_C4,
    WENDLAND_C6,

    NUM_DEF_KERNELS,
};
extern const char* kernel_name[NUM_DEF_KERNELS];
template<int d>
class Kernel {
    public:
        static const int dim = d;

        static_assert(0<d, "Dimension has to be positive!");

        Kernel();
        Kernel(KernelType type_);
        Kernel(const char *name);

        void init(KernelType type_);
        void init(const char *name);
        void generate_projection(int N);
        int proj_table_size() const {return _proj.size();}

        KernelType type() const {return _type;}
        KernelType norm() const {return _norm;}
        const char *name() const {return kernel_name[_type];}

        double value_ql1(double q, double H) const {
            assert(0.0 <= q and q <= 1.0);
            return pow(H,-d) * _norm * _w(q);
        }
        double value(double q, double H) const {
            return q<1.0 ? value_ql1(q,H) : 0.0;
        }

        double proj_value_ql1(double q, double H) const;
        double proj_value(double q, double H) const {
            return q<1.0 ? proj_value_ql1(q,H) : 0.0;
        }

        double operator()(double q, double H) const {return value(q,H);}

    private:
        KernelType _type;
        double _norm;
        double (*_w)(double q);

        std::vector<double> _proj;
};

extern "C" double cubic(double q, double H);
extern "C" double quartic(double q, double H);
extern "C" double quintic(double q, double H);
extern "C" double Wendland_C2(double q, double H);
extern "C" double Wendland_C4(double q, double H);
extern "C" double Wendland_C6(double q, double H);


double _cubic_kernel(double q);
double _quartic_kernel(double q);
double _quintic_kernel(double q);
double _Wendland_C2_1D(double q);
double _Wendland_C2_2D_3D(double q);
double _Wendland_C4_1D(double q);
double _Wendland_C4_2D_3D(double q);
double _Wendland_C6_1D(double q);
double _Wendland_C6_2D_3D(double q);

template<int d>
Kernel<d>::Kernel()
    : _type(UNDEFINED_KERNEL), _norm(1.0), _w([](double q){return q;})
{
    generate_projection(126);
}

template<int d>
Kernel<d>::Kernel(KernelType type_)
    : _type(type_), _norm(), _w()
{
    init(type_);
}

template<int d>
Kernel<d>::Kernel(const char *name)
    : _type(), _norm(), _w()
{
    init(name);
}

template<int d>
void Kernel<d>::init(KernelType type_)
{
    _type = type_;

    switch (_type) {
        case CUBIC_SPLINE:
            _w = _cubic_kernel;
            switch (d) {
                case 1: _norm = 8.0/3.0;            break;
                case 2: _norm = 80.0/(7.0*M_PI);    break;
                case 3: _norm = 16.0/M_PI;          break;
            }
            break;

        case QUARTIC_SPLINE:
            _w = _quartic_kernel;
            switch (d) {
                case 1: _norm = 3125.0/768.0;           break;
                case 2: _norm = 46875.0/(2398.0*M_PI);  break;
                case 3: _norm = 15625.0/(512.0*M_PI);   break;
            }
            break;

        case QUINTIC_SPLINE:
            _w = _quintic_kernel;
            switch (d) {
                case 1: _norm = 243.0/40.0;             break;
                case 2: _norm = 15309.0/(478.0*M_PI);   break;
                case 3: _norm = 2187.0/(40.0*M_PI);     break;
            }
            break;

        case WENDLAND_C2:
            switch (d) {
                case 1: _w = _Wendland_C2_1D;       break;
                case 2:
                case 3: _w = _Wendland_C2_2D_3D;    break;
            }
            switch (d) {
                case 1: _norm = 5.0/4.0;            break;
                case 2: _norm = 7.0/M_PI;           break;
                case 3: _norm = 21.0/(2.0*M_PI);    break;
            }
            break;

        case WENDLAND_C4:
            switch (d) {
                case 1: _w = _Wendland_C4_1D;       break;
                case 2:
                case 3: _w = _Wendland_C4_2D_3D;    break;
            }
            switch (d) {
                case 1: _norm = 3.0/2.0;            break;
                case 2: _norm = 9.0/M_PI;           break;
                case 3: _norm = 495.0/(32.0*M_PI);  break;
            }
            break;

        case WENDLAND_C6:
            switch (d) {
                case 1: _w = _Wendland_C6_1D;       break;
                case 2:
                case 3: _w = _Wendland_C6_2D_3D;    break;
            }
            switch (d) {
                case 1: _norm = 55.0/32.0;          break;
                case 2: _norm = 78.0/(7.0*M_PI);    break;
                case 3: _norm = 1365.0/(64.0*M_PI); break;
            }
            break;

        default:
            assert(false && "unknown kernel type!");
            fprintf(stderr, "ERROR: unkown kernel type %d!\n", type_);
            _w = [](double q){return q;};
            _norm = 1.0;
    }

    generate_projection(126);
}

template<int d>
void Kernel<d>::init(const char *name)
{
    for (int type=UNDEFINED_KERNEL+1; type<NUM_DEF_KERNELS; type++) {
        if (!strcmp(name, kernel_name[type])) {
            init((KernelType)type);
            return;
        }
    }

    fprintf(stderr, "ERROR: unknown kernel '%s'\n", name);
    init(UNDEFINED_KERNEL);
}

struct _gsl_integ_w_along_b_param_t {
    double (*w)(double);
    double b;
};
double _gsl_integ_w_along_b(double q, void *w);
template<int d>
void Kernel<d>::generate_projection(int N) {
    assert(0<N);
    //printf("projecting kernel '%s'\n", name());

    _proj.resize(N+1);
    _proj[N] = 0.0;
    size_t Nws = 4096;
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(Nws);
    gsl_function F;
    F.function = _gsl_integ_w_along_b;
    struct _gsl_integ_w_along_b_param_t params;
    params.w = _w;
    F.params = (void *)&params;
    double res, error;
    for (int i=0; i<N; i++) {
        params.b = double(i) / N;   // impact parameter to integrate at
        double l_max = std::sqrt(1.0 - std::pow(params.b, 2.0));
        gsl_integration_qag(&F, 0.0, l_max,
                            1e-13, 1e-13, Nws, GSL_INTEG_GAUSS61,
                            ws, &res, &error);
        if (error > 1e-12)
            fprintf(stderr, "WARNING: Error in kernel integration >1e-12!\n");
        assert(res >= 0.0 && "line-of-sight integration of kernel cannot be negative!");
        _proj[i] = 2.0 * res;
    }
    gsl_integration_workspace_free(ws);
}

template<int d>
double Kernel<d>::proj_value_ql1(double q, double H) const {
    assert(0.0 <= q and q <= 1.0);
    double qi = q * _proj.size();
    int i1=int(qi), i2=std::min<int>(i1+1, _proj.size()-1);
    assert(0<=i1 and (unsigned)i1 < _proj.size());
    double alpha = qi-i1;
    double proj_w = (1.0-alpha)*_proj[i1] + alpha*_proj[i2];
    return pow(H,-d+1) * _norm * proj_w;
}

