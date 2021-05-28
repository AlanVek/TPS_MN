import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# derivs: Función que devuelve una lista con las n derivadas evaluadas
def taylor(derivs, t_0, t_f, delta_t, x_0):

    N = int(np.round((t_f - t_0) / delta_t)) + 1
    x_k = np.zeros((N, len(x_0)))
    x_k[0] = x_0
    t_k = t_0 + delta_t * np.arange(N)
    
    for i in range(N-1):
        adder = np.asarray(derivs(t_k[i], x_k[i]))
        expos = np.arange(1, adder.shape[1] + 1)
        x_k[i + 1] = x_k[i] + adder.dot(delta_t**expos / factorial(expos))

    return t_k, x_k
#################################################################################################

if __name__ == '__main__':

    x_0, t_0, t_f = [1, 1], 0, .3
    delta_t = .1

    def derivs(t, x):
        k, y = x
        first_k = -(2*k + 10 + np.sin(t))
        second_k = -2 * first_k - 10 * np.cos(10 * t)

        first_y = k
        second_y = first_k

        return [[first_k, second_k], [first_y, second_y]]

    t, x = taylor(derivs, t_0 = t_0, t_f = t_f, delta_t = delta_t, x_0 = x_0)
    k, y = x.T

    plt.plot(t, y, label = 'Taylor')
    # plt.plot(t, 1.33 * np.exp(3*t) - (3*t + 1)/3, label='Real')
    plt.legend()

    plt.title(f'Taylor')
    plt.xlabel('Time [s]')
    plt.grid()
    plt.tight_layout()
    plt.show()

    print(y[np.argmin(np.abs(t - .2))])

