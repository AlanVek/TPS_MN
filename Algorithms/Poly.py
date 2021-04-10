import numpy as np
from LU import leastsq_lu
import matplotlib.pyplot as plt
from scipy.special import factorial

def find_poly(data_in, data_out, order):
    A = data_in.reshape(-1, 1) ** np.arange(order, -1, -1)
    return leastsq_lu(A, data_out).reshape(-1)

def derivate(t, f, order, poly_order):
    if order > poly_order: return np.zeros(1)
    taylor = find_poly(t, f, poly_order)
    i = np.arange(taylor.size - order)
    return taylor[ : i[-1] + 1] * factorial(taylor.size - i - 1) / factorial(taylor.size - i - 1 - order)


if __name__ == '__main__':
    t = np.linspace(1, 7, 10000)

    y = np.exp(np.sqrt(t**2 + .5)) / t

    fig, (ax1, ax2) = plt.subplots(2)

    poly = np.poly1d(find_poly(t, y, 50))
    polyd = np.poly1d(derivate(t, y, 1, 50))

    ax1.plot(t, y, label = 'Input', linewidth = 4)
    ax1.plot(t, poly(t), label = 'Poly')
    ax1.grid()
    ax1.legend()

    ax2.plot(t[1:], np.diff(y) / np.diff(t), label = 'NumPy deriv', linewidth = 4)
    ax2.plot(t, polyd(t), label = 'Poly deriv')
    ax2.grid()
    ax2.legend()

    fig.tight_layout()
    plt.show()
