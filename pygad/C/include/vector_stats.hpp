#include <vector>
#include <cmath>

template <typename T>
T sum(const std::vector<T> &X) {
    T s(0);
    for ( const T &x : X ) s += x;
    return s;
}

template <typename T>
T mean(const std::vector<T> &X) {
    return sum(X) / X.size();
}
template <typename T>
T mean(const std::vector<T> &X, const std::vector<T> &w) {
    assert( X.size() == w.size() );
    T EX(0), norm(0);
    for ( size_t i=0; i<X.size(); i++ ) {
        EX += X[i] * w[i];
        norm += w[i];
    }
    return EX / norm;
}

template <typename T>
T variance(const std::vector<T> &X) {
    T var(0);
    T EX = mean(X);
    for ( const T &x : X ) {
        T tmp = x - EX;
        var += tmp*tmp;
    }
    return var / X.size();
}
template <typename T>
T variance(const std::vector<T> &X, const std::vector<T> &w) {
    assert( X.size() == w.size() );
    T EX(0), EX2(0), norm(0);
    for ( size_t i=0; i<X.size(); i++ ) {
        EX  += X[i]      * w[i];
        EX2 += X[i]*X[i] * w[i];
        norm += w[i];
    }
    EX /= norm;
    EX2 /= norm;
    return EX2 - EX*EX;
}

template <typename T>
T stddev(const std::vector<T> &X) {
    return std::sqrt(variance(X));
}
template <typename T>
T stddev(const std::vector<T> &X, const std::vector<T> &w) {
    return std::sqrt(variance(X,w));
}

