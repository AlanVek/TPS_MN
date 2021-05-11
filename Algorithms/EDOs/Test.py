from Cauchy import cauchy
from Euler import euler
from Heun import heun
from RK4 import ruku4
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    f = lambda t, x: x + np.sin(t)

    x_0 = -1/2#-2 + 4 * np.random.rand()
    t_0 = 0#np.random.rand() * 2
    tf = 10#t_0 + np.random.rand() * 2
    N = 100
    delta_t = (tf - t_0) / N
    args = (f, t_0, tf, delta_t, x_0)

    t_e, x_e = euler(*args)
    t_h, x_h = heun(*args)
    t_c, x_c = cauchy(*args)
    t_rk4, x_rk4 = ruku4(f, t_0, tf, delta_t, [x_0])

    K = (x_0 + np.cos(t_0)/2 + np.sin(t_0)/2) * np.exp(-t_0)
    print(K)
    real = K * np.exp(t_e) - np.sin(t_e) / 2 - np.cos(t_e) / 2

    # plt.plot(t_e, x_e, label = 'Euler')
    # plt.plot(t_h, x_h, label = 'Heun')
    # plt.plot(t_c, x_c, label = 'Cauchy')
    plt.plot(t_rk4, x_rk4, label = 'RK4')
    plt.plot(t_e, real, label = 'Real')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



