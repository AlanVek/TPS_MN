from Cauchy import cauchy
from Euler import euler
from Heun import heun
from RK4 import ruku4
from Taylor import taylor
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    f = lambda t, x: [x + np.sin(t)]

    def derivs(t, _x):
        x, = _x
        first = x + np.sin(t)
        second = first + np.cos(t)
        third = second - np.sin(t)
        fourth = third - np.cos(t)

        return [[first, second, third, fourth]]

    x_0 = [-1/2]
    t_0 = 0
    tf = 2
    N = 100
    delta_t = (tf - t_0) / (N - 1)
    args = (f, t_0, tf, delta_t, x_0)

    t_e, x_e = euler(*args)
    t_h, x_h = heun(*args)
    t_c, x_c = cauchy(*args)
    t_rk4, x_rk4 = ruku4(*args)
    t_t, x_t = taylor(derivs, *args[1:])

    K = (x_0 + np.cos(t_0)/2 + np.sin(t_0)/2) * np.exp(-t_0)
    real = K * np.exp(t_e) - np.sin(t_e) / 2 - np.cos(t_e) / 2

    fig = plt.figure()
    plt.plot(t_e, x_e, label = 'Euler')
    plt.plot(t_h, x_h, label = 'Heun')
    plt.plot(t_c, x_c, label = 'Cauchy')
    plt.plot(t_rk4, x_rk4, label='RK4')
    plt.plot(t_t, x_t, label='Taylor 4')
    plt.plot(t_e, real, label = 'Real')
    plt.grid()
    plt.legend()
    plt.title('Resultados')
    plt.tight_layout()

    fig2 = plt.figure()
    plt.plot(t_e, np.abs(real - x_e.reshape(-1)), label = 'Euler')
    plt.plot(t_h, np.abs(real - x_h.reshape(-1)), label = 'Heun')
    plt.plot(t_c, np.abs(real - x_c.reshape(-1)), label = 'Cauchy')
    plt.plot(t_rk4, np.abs(real - x_rk4.reshape(-1)), label = 'RK4')
    plt.plot(t_t, np.abs(real - x_t.reshape(-1)), label = 'Taylor 4')
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.title('Errores')
    plt.tight_layout()

    plt.show()



