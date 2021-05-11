import numpy as np
import matplotlib.pyplot as plt

def heun(f, t_0, tf, delta_t, x_0):
    N = int((tf - t_0) / delta_t) + 1 # cantidad de iteraciones
    x_k = np.zeros(N)
    x_k[0] = x_0
    t_k = t_0 + delta_t * np.arange(N)

    for i in range(N-1):
        val = f(x_k[i], t_k[i])
        x_k[i+1] = x_k[i] + val * delta_t/2 + f(t_k[i] + delta_t, x_k[i] + val * delta_t) * delta_t/2

    return t_k, x_k


if __name__ == '__main__':
    f = lambda t, x: 3 * (x + t)

    x_0 = -2 + 4 * np.random.rand()
    t_0 = np.random.rand() * 2
    tf = t_0 + np.random.rand() * 2
    N = 10e3
    delta_t = (tf - t_0) / N

    t, x = heun(f, t_0, tf, delta_t, x_0)

    K = (x_0 + (3 * t_0 + 1) / 3) * np.exp(-3 * t_0)
    real = K * np.exp(3 * t) - (3 * t + 1) / 3

    plt.plot(t, x, label='Heun')
    plt.plot(t, real, label='Real')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
