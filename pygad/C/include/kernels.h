#include "general.h"

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

        KernelType type() const {return _type;}
        KernelType norm() const {return _norm;}
        const char *name() const {return kernel_name[_type];}

        double value_ql1(double q, double H) const {
            return pow(H,-d) * _norm * _w(q);
        }
        double value(double q, double H) const {
            return q<1.0 ? value_ql1(q,H) : 0.0;
        }

        double operator()(double q, double H) const {return value(q,H);}

    private:
        KernelType _type;
        double _norm;
        double (*_w)(double q);
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
}

template<int d>
Kernel<d>::Kernel(KernelType type_)
    : _type(type_)
{
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
            fprintf(stderr, "ERROR: unkown kernel!\n");
            _w = NULL;  // should never happen
    }
}

