import numpy as np
import matplotlib.pyplot as plt

def cauchy(f, t_0, tf, delta_t, x_0):
    N = int((tf - t_0) / delta_t) + 1 # cantidad de iteraciones
    x_k = np.zeros((N, len(x_0)))
    x_k[0] = x_0
    t_k = t_0 + delta_t * np.arange(N)

    for i in range(N-1):
        f1 = f(t_k[i] + delta_t/2, x_k[i] + np.asarray(f(t_k[i], x_k[i])) * delta_t/2)
        x_k[i+1] = x_k[i] + np.asarray(f1) * delta_t

    return t_k, x_k

