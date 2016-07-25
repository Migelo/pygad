__all__ = ['UVB']

import numpy as np

UVB = {
    'FG11':{
        'logz':np.array([0.   ,  0.005,  0.01 ,  0.015,  0.02 ,  0.025,  0.03 ,  0.035,
                         0.04 ,  0.045,  0.05 ,  0.055,  0.06 ,  0.065,  0.07 ,  0.075,
                         0.08 ,  0.085,  0.09 ,  0.095,  0.1  ,  0.105,  0.11 ,  0.115,
                         0.12 ,  0.125,  0.13 ,  0.135,  0.14 ,  0.145,  0.15 ,  0.155,
                         0.16 ,  0.165,  0.17 ,  0.175,  0.18 ,  0.185,  0.19 ,  0.195,
                         0.2  ,  0.205,  0.21 ,  0.215,  0.22 ,  0.225,  0.23 ,  0.235,
                         0.24 ,  0.245,  0.25 ,  0.255,  0.26 ,  0.265,  0.27 ,  0.275,
                         0.28 ,  0.285,  0.29 ,  0.295,  0.3  ,  0.305,  0.31 ,  0.315,
                         0.32 ,  0.325,  0.33 ,  0.335,  0.34 ,  0.345,  0.35 ,  0.355,
                         0.36 ,  0.365,  0.37 ,  0.375,  0.38 ,  0.385,  0.39 ,  0.395,
                         0.4  ,  0.405,  0.41 ,  0.415,  0.42 ,  0.425,  0.43 ,  0.435,
                         0.44 ,  0.445,  0.45 ,  0.455,  0.46 ,  0.465,  0.47 ,  0.475,
                         0.48 ,  0.485,  0.49 ,  0.495,  0.5  ,  0.505,  0.51 ,  0.515,
                         0.52 ,  0.525,  0.53 ,  0.535,  0.54 ,  0.545,  0.55 ,  0.555,
                         0.56 ,  0.565,  0.57 ,  0.575,  0.58 ,  0.585,  0.59 ,  0.595,
                         0.6  ,  0.605,  0.61 ,  0.615,  0.62 ,  0.625,  0.63 ,  0.635,
                         0.64 ,  0.645,  0.65 ,  0.655,  0.66 ,  0.665,  0.67 ,  0.675,
                         0.68 ,  0.685,  0.69 ,  0.695,  0.7  ,  0.705,  0.71 ,  0.715,
                         0.72 ,  0.725,  0.73 ,  0.735,  0.74 ,  0.745,  0.75 ,  0.755,
                         0.76 ,  0.765,  0.77 ,  0.775,  0.78 ,  0.785,  0.79 ,  0.795,
                         0.8  ,  0.805,  0.81 ,  0.815,  0.82 ,  0.825,  0.83 ,  0.835,
                         0.84 ,  0.845,  0.85 ,  0.855,  0.86 ,  0.865,  0.87 ,  0.875,
                         0.88 ,  0.885,  0.89 ,  0.895,  0.9  ,  0.905,  0.91 ,  0.915,
                         0.92 ,  0.925,  0.93 ,  0.935,  0.94 ,  0.945,  0.95 ,  0.955,
                         0.96 ,  0.965,  0.97 ,  0.975,  0.98 ,  0.985,  0.99 ,  0.995,
                         1.   ,  1.005,  1.01 ,  1.015,  1.02 ,  1.025,  1.03 ,  1.035,
                         1.04 ,  1.045,  1.05 ,  1.055,  1.06 ,  1.065,  1.07]),
        'gH0':np.array([3.76244000e-14,   3.83213000e-14,   3.90303000e-14,
                        4.01290000e-14,   4.16007000e-14,   4.31222000e-14,
                        4.47007000e-14,   4.63453000e-14,   4.80462000e-14,
                        4.98113000e-14,   5.16490000e-14,   5.35503000e-14,
                        5.55239000e-14,   5.75769000e-14,   5.97016000e-14,
                        6.19076000e-14,   6.42009000e-14,   6.65751000e-14,
                        6.90410000e-14,   7.16025000e-14,   7.42548000e-14,
                        7.70100000e-14,   7.98695000e-14,   8.28307000e-14,
                        8.59068000e-14,   8.90972000e-14,   9.24012000e-14,
                        9.58331000e-14,   9.94039000e-14,   1.03087000e-13,
                        1.06911000e-13,   1.10885000e-13,   1.14984000e-13,
                        1.19226000e-13,   1.23641000e-13,   1.28194000e-13,
                        1.32903000e-13,   1.37801000e-13,   1.42848000e-13,
                        1.48065000e-13,   1.53480000e-13,   1.59057000e-13,
                        1.64817000e-13,   1.70784000e-13,   1.76923000e-13,
                        1.83257000e-13,   1.89805000e-13,   1.96533000e-13,
                        2.03467000e-13,   2.10616000e-13,   2.17951000e-13,
                        2.25498000e-13,   2.33258000e-13,   2.41203000e-13,
                        2.49362000e-13,   2.57724000e-13,   2.66267000e-13,
                        2.75018000e-13,   2.83999000e-13,   2.93102000e-13,
                        3.02401000e-13,   3.11903000e-13,   3.21502000e-13,
                        3.31240000e-13,   3.41165000e-13,   3.51160000e-13,
                        3.61251000e-13,   3.71488000e-13,   3.81738000e-13,
                        3.92027000e-13,   4.02401000e-13,   4.12718000e-13,
                        4.22998000e-13,   4.33286000e-13,   4.43424000e-13,
                        4.53438000e-13,   4.63352000e-13,   4.73011000e-13,
                        4.82441000e-13,   4.91671000e-13,   5.00591000e-13,
                        5.09267000e-13,   5.17733000e-13,   5.25902000e-13,
                        5.33842000e-13,   5.41633000e-13,   5.48998000e-13,
                        5.56077000e-13,   5.62935000e-13,   5.69306000e-13,
                        5.75296000e-13,   5.80996000e-13,   5.86202000e-13,
                        5.90973000e-13,   5.95411000e-13,   5.99308000e-13,
                        6.02729000e-13,   6.05783000e-13,   6.08265000e-13,
                        6.10246000e-13,   6.11843000e-13,   6.12855000e-13,
                        6.13357000e-13,   6.13477000e-13,   6.13017000e-13,
                        6.12060000e-13,   6.10742000e-13,   6.08870000e-13,
                        6.06530000e-13,   6.03867000e-13,   6.00695000e-13,
                        5.97115000e-13,   5.93252000e-13,   5.88951000e-13,
                        5.84315000e-13,   5.79553000e-13,   5.74327000e-13,
                        5.68850000e-13,   5.63301000e-13,   5.57385000e-13,
                        5.51270000e-13,   5.45110000e-13,   5.38750000e-13,
                        5.32273000e-13,   5.25828000e-13,   5.19261000e-13,
                        5.12652000e-13,   5.06142000e-13,   4.99581000e-13,
                        4.93041000e-13,   4.86655000e-13,   4.80274000e-13,
                        4.73961000e-13,   4.67838000e-13,   4.61761000e-13,
                        4.55782000e-13,   4.50010000e-13,   4.44306000e-13,
                        4.38712000e-13,   4.33323000e-13,   4.28006000e-13,
                        4.22791000e-13,   4.17762000e-13,   4.12791000e-13,
                        4.07901000e-13,   4.03235000e-13,   3.98514000e-13,
                        3.93836000e-13,   3.89305000e-13,   3.84682000e-13,
                        3.80033000e-13,   3.75380000e-13,   3.70629000e-13,
                        3.65741000e-13,   3.60711000e-13,   3.55487000e-13,
                        3.50126000e-13,   3.44716000e-13,   3.39250000e-13,
                        3.33815000e-13,   3.28484000e-13,   3.23176000e-13,
                        3.17893000e-13,   3.12648000e-13,   3.07359000e-13,
                        3.02040000e-13,   2.96708000e-13,   2.91291000e-13,
                        2.85810000e-13,   2.80289000e-13,   2.74664000e-13,
                        2.68965000e-13,   2.63257000e-13,   2.57409000e-13,
                        2.51476000e-13,   2.45535000e-13,   2.39468000e-13,
                        2.33338000e-13,   2.27174000e-13,   2.20960000e-13,
                        2.14712000e-13,   2.08469000e-13,   2.02191000e-13,
                        1.95911000e-13,   1.89667000e-13,   1.83417000e-13,
                        1.77197000e-13,   1.71041000e-13,   1.64910000e-13,
                        1.58837000e-13,   1.52855000e-13,   1.46927000e-13,
                        1.41081000e-13,   1.35349000e-13,   1.29695000e-13,
                        1.24146000e-13,   1.18727000e-13,   1.13406000e-13,
                        1.07857000e-13,   1.00976000e-13,   9.28557000e-14,
                        8.37879000e-14,   7.40917000e-14,   6.40348000e-14,
                        5.39463000e-14,   4.41478000e-14,   3.48885000e-14,
                        2.64282000e-14,   1.89767000e-14,   1.26822000e-14,
                        7.64449000e-15,   3.90686000e-15,   1.45165000e-15,
                        2.10205000e-16,   0.00000000e+00])
    }
}
