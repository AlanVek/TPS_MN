import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# derivs: Función que devuelve una lista con las n derivadas evaluadas
def taylor(derivs, t_0, t_f, delta_t, x_0):

    N = int((t_f - t_0) / delta_t) + 1
    x_k = np.zeros(N)
    x_k[0] = x_0
    t_k = t_0 + delta_t * np.arange(N)
    
    for i in range(N-1):
        adder = np.asarray(derivs(x_k[i], t_k[i]))
        expos = np.arange(1, adder.size + 1)
        x_k[i + 1] = x_k[i] + adder.dot(delta_t**expos / factorial(expos))

    return t_k, x_k
#################################################################################################

if __name__ == '__main__':

    x_0, t_0, t_f = 1, 0, .8
    delta_t = 1e-2

    n = 10
    def derivs(x, t):
        first = 3 * (x + t)
        second = 3 * (first + 1)
        res = [first, second]

        for _ in range(n - 2): res += [3 * res[-1]]
        return res

    t, x = taylor(derivs, t_0 = t_0, t_f = t_f, delta_t = delta_t, x_0 = x_0)

    plt.plot(t, x, label = 'Taylor')
    plt.plot(t, 1.33 * np.exp(3*t) - (3*t + 1)/3, label='Real')
    plt.legend()

    plt.title(f'Taylor -- n = {n}')
    plt.xlabel('Time [s]')
    plt.grid()
    plt.tight_layout()
    plt.show()
