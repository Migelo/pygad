#pragma once
#include "general.hpp"

#include <gsl/gsl_integration.h>
#include <vector>
#include <map>

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
        Kernel(KernelType type_, unsigned proj_tbl_size=128,
                                 unsigned los_tbl_size=32);
        Kernel(const char *name, unsigned proj_tbl_size=128,
                                 unsigned los_tbl_size=32);

        void init(KernelType type_, unsigned proj_tbl_size, unsigned los_tbl_size);
        void init(const char *name, unsigned proj_tbl_size, unsigned los_tbl_size);

        void require_table_size(unsigned proj_tbl_size, unsigned los_tbl_size);

        KernelType type() const {return _type;}
        double norm() const {return _norm;}
        const char *name() const {return kernel_name[_type];}

        double value_ql1(double q, double H) const {
            assert(0.0 <= q and q <= 1.0);
            return pow(H,-d) * _norm * _w(q);
        }
        double value(double q, double H) const {
            return q<1.0 ? value_ql1(q,H) : 0.0;
        }
        double operator()(double q, double H) const {return value(q,H);}

        // the following function do not make sense for d == 2 and are only tested for
        // d == 3, no higher dimension

        void generate_projection(int N);
        int proj_table_size() const {return _proj.size();}
        void generate_los_integrals(int N, int M);
        int los_integrals_table_size() const {
            return _los_integ.empty() ? 0 : _los_integ.size()*_los_integ[0].size();
        }

        double proj_value_ql1(double q, double H) const;
        double proj_value(double q, double H) const {
            return q<1.0 ? proj_value_ql1(q,H) : 0.0;
        }

        double los_integ_value(double b, double x, double y, double H) const;

    private:
        KernelType _type;
        double _norm;
        double (*_w)(double q);

        std::vector<double> _proj;
        std::vector<std::vector<double>> _los_integ;

        double _los_integ_loockup(int b1, int b2, double alpha_b,
                                  int x1, int x2, double alpha_x) const;
};
extern std::map<std::string,Kernel<3>> kernels;

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
}

template<int d>
Kernel<d>::Kernel(KernelType type_, unsigned proj_tbl_size, unsigned los_tbl_size)
    : _type(type_), _norm(), _w()
{
    init(type_, proj_tbl_size, los_tbl_size);
}

template<int d>
Kernel<d>::Kernel(const char *name, unsigned proj_tbl_size, unsigned los_tbl_size)
    : _type(), _norm(), _w()
{
    init(name, proj_tbl_size, los_tbl_size);
}

template<int d>
void Kernel<d>::init(KernelType type_, unsigned proj_tbl_size, unsigned los_tbl_size)
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

    require_table_size(proj_tbl_size, los_tbl_size);
}

template<int d>
void Kernel<d>::init(const char *name, unsigned proj_tbl_size, unsigned los_tbl_size)
{
    for (int type=UNDEFINED_KERNEL+1; type<NUM_DEF_KERNELS; type++) {
        if (!strcmp(name, kernel_name[type])) {
            init((KernelType)type, proj_tbl_size, los_tbl_size);
            return;
        }
    }

    fprintf(stderr, "ERROR: unknown kernel '%s'\n", name);
    init(UNDEFINED_KERNEL, 0, 0);
}

template<int d>
void Kernel<d>::require_table_size(unsigned proj_tbl_size, unsigned los_tbl_size) {
    //printf("require table sizes of %d and %d\n", proj_tbl_size, los_tbl_size);
    if ( proj_tbl_size > _proj.size() )
        generate_projection(proj_tbl_size);
    if ( los_tbl_size > _los_integ.size() or los_tbl_size > _los_integ[0].size() )
        generate_los_integrals(los_tbl_size, los_tbl_size);
}

struct _gsl_integ_w_along_b_param_t {
    double (*w)(double);
    double b;
};
double _gsl_integ_w_along_b(double q, void *w);
template<int d>
void Kernel<d>::generate_projection(int N) {
    assert(0<N);
    //printf("projecting kernel '%s' - table with %d entries\n", name(), N);

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
void Kernel<d>::generate_los_integrals(int N, int M) {
    assert(0<N);
    assert(0<M);

    _los_integ.resize(N+1);
    size_t Nws = 4096;
    gsl_integration_workspace *ws = gsl_integration_workspace_alloc(Nws);
    gsl_function F;
    F.function = _gsl_integ_w_along_b;
    struct _gsl_integ_w_along_b_param_t params;
    params.w = _w;
    F.params = (void *)&params;
    double res, error;
    for (int i=0; i<=N; i++) {
        params.b = double(i) / N;   // impact parameter to integrate at
        double l_max = std::sqrt(1.0 - std::pow(params.b, 2.0));
        _los_integ[i].resize(M+1);
        for (int j=0; j<=M; j++) {
            double l = double(j) / M;   // upper integration bound
            if ( l > l_max )
                l = l_max;
            gsl_integration_qag(&F, 0.0, l,
                                1e-13, 1e-13, Nws, GSL_INTEG_GAUSS61,
                                ws, &res, &error);
            if (error > 1e-12)
                fprintf(stderr, "WARNING: Error in kernel integration >1e-12!\n");
            assert(res >= 0.0 && "line-of-sight integration of kernel cannot be negative!");
            _los_integ[i][j] = res;
        }
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

template<int d>
double Kernel<d>::_los_integ_loockup(int b1, int b2, double alpha_b,
                                     int x1, int x2, double alpha_x) const {
    assert( 0<=b1 and (unsigned)b1 < _los_integ.size() );
    assert( 0<=b2 and (unsigned)b2 < _los_integ.size() );
    assert( 0<=alpha_b and alpha_b<=1.0 );
    assert( 0<=x1 and (unsigned)x1 < _los_integ[0].size() );
    assert( 0<=x2 and (unsigned)x2 < _los_integ[0].size() );
    assert( 0<=alpha_x and alpha_x<=1.0 );

    // bi-linear interpolation
    double I_b1 = (1.0-alpha_x) * _los_integ[b1][x1] + alpha_x * _los_integ[b1][x2];
    double I_b2 = (1.0-alpha_x) * _los_integ[b2][x1] + alpha_x * _los_integ[b2][x2];
    return (1.0-alpha_b) * I_b1 + alpha_b * I_b2;
}
template<int d>
double Kernel<d>::los_integ_value(double b, double x, double y, double H) const {
    assert( d == 3 );
    // integral along a line from x to y at impact parameter b, using bi-linear
    // interpolation within the table
    assert( x <= y );

    if ( b >= 1.0 )
        return 0.0;
    // get b-indices and alpha
    assert( 0.0 <= b );
    double bi = b * _los_integ.size();
    int b1=int(bi), b2=std::min<int>(b1+1, _los_integ.size()-1);
    assert( 0<=b1 and (unsigned)b1 < _los_integ.size() );
    double alpha_b = bi-b1;

    // get |x|-indices and alpha
    double xi = std::min(std::abs(x),1.0) * _los_integ[0].size();
    int x1=std::min<int>(int(xi), _los_integ[0].size()-1);
    int x2=std::min<int>(x1+1,    _los_integ[0].size()-1);
    assert( 0<=x1 and (unsigned)x1 < _los_integ[0].size() );
    double alpha_x = xi-x1;

    // get |y|-indices and alpha
    double yi = std::min(std::abs(y),1.0) * _los_integ[0].size();
    int y1=std::min<int>(int(yi), _los_integ[0].size()-1);
    int y2=std::min<int>(y1+1,    _los_integ[0].size()-1);
    assert( 0<=y1 and (unsigned)y1 < _los_integ[0].size() );
    double alpha_y = yi-y1;

    // lookup the integrals from 0 to |x|/|y|
    double I_b_0x = _los_integ_loockup( b1,b2,alpha_b, x1,x2,alpha_x );
    double I_b_0y = _los_integ_loockup( b1,b2,alpha_b, y1,y2,alpha_y );

    // combine the integral to the integral from x to y
    assert( x <= y );
    double I;
    if ( 0.0 <= x ) {   // =>  0 <= y
        I = I_b_0y - I_b_0x;
    } else {    // x < 0
        if ( y < 0.0 ) {
            I = I_b_0x - I_b_0y;
        } else {    // 0 <= y
            I = I_b_0x + I_b_0y;
        }
    }

    return pow(H,-d+1) * _norm * I;
}

