#include "kernels.hpp"

static Kernel<3> cubic_kernel(CUBIC_SPLINE);
static Kernel<3> quartic_kernel(QUARTIC_SPLINE);
static Kernel<3> quintic_kernel(QUINTIC_SPLINE);
static Kernel<3> WC2_kernel(WENDLAND_C2);
static Kernel<3> WC4_kernel(WENDLAND_C4);
static Kernel<3> WC6_kernel(WENDLAND_C6);
double cubic(double q, double H) {return cubic_kernel(q,H);}
double quartic(double q, double H) {return quartic_kernel(q,H);}
double quintic(double q, double H) {return quintic_kernel(q,H);}
double Wendland_C2(double q, double H) {return WC2_kernel(q,H);}
double Wendland_C4(double q, double H) {return WC4_kernel(q,H);}
double Wendland_C6(double q, double H) {return WC6_kernel(q,H);}

const char* kernel_name[NUM_DEF_KERNELS] = {
    "<undefined>",

    "cubic",
    "quartic",
    "quintic",
    "Wendland C2",
    "Wendland C4",
    "Wendland C6",
};

std::map<std::string,Kernel<3>> kernels = {
    {"<undefined>", Kernel<3>()},

    {"cubic",       Kernel<3>(CUBIC_SPLINE)},
    {"quartic",     Kernel<3>(QUARTIC_SPLINE)},
    {"quintic",     Kernel<3>(QUINTIC_SPLINE)},
    {"Wendland C2", Kernel<3>(WENDLAND_C2)},
    {"Wendland C4", Kernel<3>(WENDLAND_C4)},
    {"Wendland C6", Kernel<3>(WENDLAND_C6)},
};

double _gsl_integ_w_along_b(double l, void *params) {
    struct _gsl_integ_w_along_b_param_t *param = (struct _gsl_integ_w_along_b_param_t *)params;
    return param->w( std::sqrt(l*l + std::pow(param->b,2.0)) );
}

double _cubic_kernel(double q) {
    double w = pow(1.0-q,3.0);
    if (q<0.5) {
        w -= 4.0 * pow(0.5-q,3);
    }
    return w;
}

double _quartic_kernel(double q) {
    double w = pow(1.0-q,4.0);
    if (q<3.0/5.0) {
        w -= 5.0 * pow(3.0/5.0-q,4);
        if (q<1.0/5.0) {
            w += 10.0 * pow(1.0/5.0-q,4);
        }
    }
    return w;
}

double _quintic_kernel(double q) {
    double w = pow(1.0-q,5.0);
    if (q<2.0/3.0) {
        w -= 6.0 * pow(2.0/3.0-q,5);
        if (q<1.0/3.0) {
            w += 15.0 * pow(1.0/3.0-q,5);
        }
    }
    return w;
}

double _Wendland_C2_1D(double q) {
    return pow(1.0-q,3) * (1.0+3.0*q);
}
double _Wendland_C2_2D_3D(double q) {
    return pow(1.0-q,4) * (1.0+4.0*q);
}

double _Wendland_C4_1D(double q) {
    return pow(1.0-q,5) * (1.0+5.0*q+8.0*q*q);
}
double _Wendland_C4_2D_3D(double q) {
    return pow(1.0-q,6) * (1.0+6.0*q+35.0/3.0*q*q);
}

double _Wendland_C6_1D(double q) {
    return pow(1.0-q,7) * (1.0+7.0*q+19.0*q*q+21.0*q*q*q);
}
double _Wendland_C6_2D_3D(double q) {
    return pow(1.0-q,8) * (1.0+8.0*q+25.0*q*q+32.0*q*q*q);
}

