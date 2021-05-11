from Ej2.constants import *

def gen_hh(i):
    def HH(t, vars):
        v, n, m, h = vars
        out_v = (i(t) - g_Na * m**3 * h * (v - v_Na) - g_K * n**4 * (v - v_K) - g_L * (v - v_L)) / C
        out_n = a_n(v) * (1 - n) - n * b_n(v)
        out_m = a_m(v) * (1 - m) - m * b_m(v)
        out_h = a_h(v) * (1 - h) - h * b_h(v)

        return np.array([out_v, out_n, out_m, out_h])

    return HH

