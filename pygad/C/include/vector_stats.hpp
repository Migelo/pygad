#include <vector>
#include <cmath>
#include <numeric>

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
T median(const std::vector<T> &X) {
    size_t size = X.size();
    std::vector<T> vals = X;
    std::sort(vals.begin(), vals.end());

    if (size  % 2 == 0)
        return (vals[size / 2 - 1] + vals[size / 2]) / 2;
    else
        return vals[size / 2];
}
template <typename T>
T median(const std::vector<T> &X, const std::vector<T> &w) {
    assert( X.size() == w.size() );
    size_t size = X.size();
    if ( size == 0) {
        return T(0);
    } else if ( size == 1 ) {
        return X[0];
    }

    // "argsort" the vector X, i.e. create vector idx such that:
    //      i < j  =>  X[idx[i]] < X[idx[j]]
    std::vector<size_t> idx(X.size());
    std::iota(idx.begin(), idx.end(), 0u);
    std::sort(idx.begin(), idx.end(),
       [&X](size_t i1, size_t i2) {return X[i1] < X[i2];});

    // find median to index precision
    T w_half = sum(w) / 2;
    T w_sum = T(0);
    size_t i;
    for ( i=0; w_sum<=w_half and i<size; i++ ) {
        assert( w[idx[i]] >= 0 );
        w_sum += w[idx[i]];
    }
    --i;    // make i the last incremented index again
    assert( i>=0 and i!=T(-1) );

    // interpolate for more precise median
    T w_last = w[idx[i]];
    T alpha = (w_half - (w_sum - w_last)) / w_last;
    return alpha * X[idx[i]] + (1-alpha) * X[idx[i-1]];
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

