from Taylor import taylor
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    def d1(t, _x):
        x, = _x
        return -x**2

    def d2(t, _x):
        x, = _x
        return 2 * x**3

    x_0 = [3 * 59098 / 10**5]
    t_0 = 0
    tf = 2
    dt = 0.002
    N = int(np.round((tf - t_0)/dt)) + 1

    t_t, x_t = taylor([d1, d2], x_0 = x_0, t_0 = t_0, t_f = tf, N = N)

    K = (1/x_0[0] - t_0)
    real = 1 / (K + t_t)

    fig = plt.figure()
    plt.plot(t_t, x_t, label='Taylor 2', linewidth = 3)
    plt.plot(t_t, real, label = 'Real')
    plt.grid()
    plt.legend()
    plt.title('Resultados')
    plt.tight_layout()

    plt.show()



