#pragma once
#include "general.hpp"

typedef size_t idx_t;

// Pass a function for each level, if less functions are provieded, only the
// innermost levels have a function passed, if more are provided the last ones are
// just without effect.
// If, however, exactly to functions are passed, the first one is executed at each
// level except the innermost level, where the second function is executed.
// Call signature of the functions:
//      (unsigned n, idx_t *i, idx_t *i_min, idx_t *i_max)
template <unsigned n>
struct nested_loops {
        template <typename F, typename... functions>
        static void do_loops(idx_t *i, idx_t *i_min, idx_t *i_max, F f, functions... funcs) {
            for (i[n-1] = i_min[n-1]; i[n-1] < i_max[n-1]; i[n-1]++) {
                if (sizeof...(functions) == n-1  or  (sizeof...(functions) == 1 and n==2)) {
                    f(n-1, i, i_min, i_max);
                    nested_loops<n-1>::do_loops(i, i_min, i_max, funcs...);
                } else if (sizeof...(functions) == 1) {
                    // it is n != 2
                    f(n-1, i, i_min, i_max);
                    nested_loops<n-1>::do_loops(i, i_min, i_max, f, funcs...);
                } else {
                    // it is sizeof...(functions) != n-1 and != 1
                    nested_loops<n-1>::do_loops(i, i_min, i_max, f, funcs...);
                }
            }
        }

        friend struct nested_loops<n+1>;

    private:
        static void do_loops(idx_t *i, idx_t *i_min, idx_t *i_max) {
            // never gets called
        }
};

template <>
struct nested_loops<1> {
        template <typename F, typename... functions>
        static void do_loops(idx_t *i, idx_t *i_min, idx_t *i_max, F f, functions... funcs) {
            for (i[0] = i_min[0]; i[0] < i_max[0]; i[0]++) {
                f(0, i, i_min, i_max);
            }
        }

        friend struct nested_loops<2>;

    private:
        static void do_loops(idx_t *i, idx_t *i_min, idx_t *i_max) {
            // never gets called
        }
};

// this is basically no looping at all
template <>
struct nested_loops<0> {
        template <typename F, typename... functions>
        static void do_loops(idx_t *i, idx_t *i_min, idx_t *i_max, F f, functions... funcs) {
            f(0, i, i_min, i_max);
        }

    private:
        static void do_loops(idx_t *i, idx_t *i_min, idx_t *i_max) {
            // never gets called
        }
};

