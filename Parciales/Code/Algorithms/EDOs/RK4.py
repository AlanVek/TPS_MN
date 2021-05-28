import numpy as np
import matplotlib.pyplot as plt

def ruku4(f, t_0, t_f, delta_t, x_0):

    N = int(np.round((t_f - t_0) / delta_t)) + 1

    x_k = np.zeros((N, len(x_0)))
    x_k[0] = x_0
    t_k = t_0 + delta_t * np.arange(N)

    for i in range(N-1):
        k1 = np.asarray(f(t_k[i], x_k[i]))
        k2 = np.asarray(f(t_k[i] + delta_t/2, x_k[i] + k1 * delta_t/2))
        k3 = np.asarray(f(t_k[i] + delta_t/2, x_k[i] + k2 * delta_t/2))
        k4 = np.asarray(f(t_k[i] + delta_t, x_k[i] + k3 * delta_t))

        x_k[i+1] = x_k[i] + (k1 + 2 * k2 + 2 * k3 + k4) * delta_t / 6

    return t_k, x_k

if __name__ == '__main__':
    f = lambda t, x: [(x + 1) * np.sin(t)]

    x_0 = -10 + 20 * np.random.rand()
    t_0 = np.random.rand() * 2
    tf = t_0 + np.random.rand() * 10
    N = 100
    delta_t = (tf - t_0) / N

    t, _x = ruku4(f, t_0, tf, delta_t, [x_0])
    x, = _x.T
    K = (x_0 + 1) * np.exp(np.cos(t_0))
    real = K * np.exp(-np.cos(t)) - 1

    # fig, (ax1, ax2) = plt.subplots(2)
    fig, ax1 = plt.subplots()
    ax1.plot(t, x, label='RK4', linewidth = 3)
    ax1.plot(t, real, label='Real')
    print(delta_t)
    print(np.abs(x.reshape(-1) - real).max())
    # ax1.plot(t, x.reshape(-1) - real, label = 'Error')
    ax1.grid(); #ax2.grid()
    ax1.legend();# ax2.legend()
    fig.tight_layout()
    plt.show()

    # plt.show()