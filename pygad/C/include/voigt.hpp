#include "Faddeeva.hpp"

inline double Gaussian(double x, double sigma) {
    return std::exp(-x*x/(2.*sigma*sigma)) / sigma / std::sqrt(2*M_PI);
}
inline double Lorentzian(double x, double gamma) {
    return gamma / M_PI / (x*x + gamma*gamma);
}

extern "C" double Voigt(double x, double sigma, double gamma);

