# Ejercicio 1
##################################################

from Code.Algorithms.EDOs.RK4 import ruku4
import numpy as np
import matplotlib.pyplot as plt


def f(t, _x):
    k, x = _x
    out_k = -np.tanh(np.cos(t) * x * k) - x
    out_x = k

    return np.array([out_k, out_x])

maxerr = 1e-6
n = 4

x_0 = [1, 1]
t_0, t_f = 0, 50

delta_t1 = .5
_, x1 = ruku4(f, t_0, t_f, delta_t1, x_0)
_, x2 = ruku4(f, t_0, t_f, delta_t1/2, x_0)

c = np.max(np.abs(x1[:, 1] - x2[::2, 1]) / (delta_t1**n * (1 - 1/2**n)))

delta_t = .85 * (maxerr / c)**(1/n)

t, _x = ruku4(f, t_0, t_f, delta_t, x_0)
k, x = _x.T

fig, ax1 = plt.subplots()
ax1.plot(t, x, label='RK4', linewidth=3)
ax1.grid()
ax1.legend()
fig.tight_layout()
plt.show()

_, _x2 = ruku4(f, t_0, t_f, delta_t/2, x_0)
k2, x2 = _x2.T

real_err = np.max(np.abs(x2[::2] - x) / (1 - 1/2**n))
derv_err = np.max(np.abs(k2[::2] - k) / (1 - 1/2**n))

print('Ejercicio 1:')
print(f'\tError en x: {real_err}')
print(f"\tError en x': {derv_err}")


# Ejercicio 2
##################################################

from Code.Algorithms.Nonlinear_Eqs.Bisec import bisec

tol, maxiter, x_ini, x_fin = 2e-16, 100, 0, 3

f = lambda x: x ** 4 + x ** 3 + x ** 2 + x - 40
print('\nEjercicio 2:', bisec(f, x_ini, x_fin, tol, maxiter))


# Ejercicio 3
##################################################

from Code.Algorithms.Least_Squares.QR2_Householder import leastsq_qr
import pandas as pd

df = pd.read_csv('Code/p53.csv').sort_values(by ='x')
x, y = df['x'].to_numpy(), df['y'].to_numpy().reshape(-1, 1)

A = np.zeros((x.size, 3))
A[:, 0] = np.sqrt(np.abs(x))
A[:, 1] = np.cos(A[:, 0])
A[:, 2] = 1

res = leastsq_qr(A, y)
print('\nEjercicio 3:', res.reshape(-1))

plt.plot(x, y, label = 'Original', linewidth = 0, marker = 'o')
plt.plot(x, A.dot(res), label = 'Cuadrados Mínimos')
plt.legend()
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

