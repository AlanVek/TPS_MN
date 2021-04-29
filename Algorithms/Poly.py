import numpy as np
from LU import leastsq_lu
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.misc import derivative as scipy_deriv

def find_poly(data_in, data_out, order):
    A = data_in.reshape(-1, 1) ** range(order, -1, -1)
    return leastsq_lu(A, data_out).ravel()

def derivate(t, f, order, poly_order):
    if order > poly_order: return np.zeros(1)
    taylor = find_poly(t, f, poly_order)
    i = np.arange(taylor.size - order)
    return taylor[ : i[-1] + 1] * factorial(taylor.size - i - 1) / factorial(taylor.size - i - 1 - order)


if __name__ == '__main__':
    t = np.linspace(-5, 5, 10000)

    n = 1

    y = lambda t: np.cos(t) * np.exp(-np.abs(t)) * t

    fig, (ax1, ax2) = plt.subplots(2)

    poly = np.poly1d(find_poly(t, y(t), 40))
    polyd = np.poly1d(derivate(t, y(t), n, 40))

    ax1.plot(t, y(t), label = 'Input', linewidth = 4)
    ax1.plot(t, poly(t), label = 'Poly')
    ax1.grid()
    ax1.legend()

    ax2.plot(t,scipy_deriv(y, t, n = n, dx = t[1] - t[0], order = n + 1 + n%2), label = f'SciPy deriv, n = {n}', linewidth = 4)
    ax2.plot(t, polyd(t), label = f'Poly deriv, n = {n}')
    ax2.grid()
    ax2.legend()

    fig.tight_layout()
    plt.show()
