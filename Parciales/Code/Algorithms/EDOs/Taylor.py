import numpy as np
from scipy.special import factorial

def taylor(derivs, x_0, t_0, t_f, N):
    delta_t = (t_f - t_0) / (N - 1)
    x_k = np.zeros((N, len(x_0)))
    x_k[0] = x_0
    t_k = t_0 + delta_t * np.arange(N)

    order = len(derivs)
    expos = np.arange(1, order + 1)
    factor = delta_t**expos / factorial(expos)
    
    for i in range(N-1):
        ds = [d(t_k[i], x_k[i]) for d in derivs]
        x_k[i + 1] = x_k[i] + np.dot(ds, factor)

    return t_k, x_k
#################################################################################################