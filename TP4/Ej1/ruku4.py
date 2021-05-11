import numpy as np

def ruku4(f, t_0, t_f, delta_t, x_0):
    """ Solves dx/dt = f(t, x)

    t_0 -> Number
    t_f -> Number
    x_0 -> List or array

    """

    N = int((t_f - t_0) / delta_t) + 1

    x_k = np.zeros((N, len(x_0)))
    x_k[0] = x_0
    t_k = t_0 + delta_t * np.arange(N)

    for i in range(N-1):
        k1 = f(t_k[i], x_k[i])
        k2 = f(t_k[i] + delta_t/2, x_k[i] + k1 * delta_t/2)
        k3 = f(t_k[i] + delta_t/2, x_k[i] + k2 * delta_t/2)
        k4 = f(t_k[i] + delta_t, x_k[i] + k3 * delta_t)

        x_k[i+1] = x_k[i] + (k1 + 2 * k2 + 2 * k3 + k4) * delta_t / 6

    return t_k, x_k