import matplotlib.pyplot as plt

from Taylor import taylor
from RK4 import ruku4
import numpy as np


def derivs(t, _x):
    k, x = _x

    res_k = [x - k]
    res_k.append(k - res_k[0])
    res_k.append(res_k[0] - res_k[1])

    res_x = [k, res_k[0], res_k[1]]

    return [res_k, res_x]

def func(t, _x):
    k, x = _x
    return [x - k, k]

if __name__ == '__main__':
    t_0, t_f = 0, 3
    delta_t = 1e-3
    x_0 = [1, 1]

    t_taylor, _x_taylor = taylor(derivs, t_0, t_f, delta_t, x_0)
    k_t, x_t = _x_taylor.T

    t_rk4, _x_rk4 = ruku4(func, t_0, t_f, delta_t, x_0)
    k_rk4, x_rk4 = _x_rk4.T

    real_expos = np.roots([1, 1, -1])
    real_coefs = np.linalg.solve([real_expos, [1, 1]], x_0)
    real = real_coefs[0] * np.exp(t_rk4 * real_expos[0]) + real_coefs[1] * np.exp(t_rk4 * real_expos[1])

    plt.plot(t_taylor, x_t, label = 'Taylor')
    plt.plot(t_rk4, x_rk4, label = 'RK4')
    plt.plot(t_rk4, real, label = 'Real')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    fig = plt.figure()
    plt.plot(t_taylor, real - x_t, label = 'Taylor')
    plt.plot(t_rk4, real - x_rk4, label = 'RK4')
    plt.legend()
    plt.grid()
    plt.title('Errores')
    plt.tight_layout()

    plt.show()