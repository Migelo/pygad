#include "voigt.hpp"

typedef std::complex<double> cmplx;

double Voigt(double x, double sigma, double gamma) {
    cmplx z =
        (cmplx(0.,gamma) + x) / (std::sqrt(2) * sigma);
    return Faddeeva::w(z).real() / (std::sqrt(2*M_PI) * sigma);
}

