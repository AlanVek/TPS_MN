# Ejercicio 1
##################################################

import numpy as np
from Code.Algorithms.EDOs.RK4 import ruku4
import matplotlib.pyplot as plt

def func(t, _x):
    k, x = _x
    out_k = 1 - x - 2 * (-1 if abs(x) < 1 else 1) * x
    out_x = k

    return np.array([out_k, out_x])

x_0 = [1, 5]
t_0, t_f = 0, 15
n = 4
maxerr = 1e-4

delta_t1 = .01
_, x1 = ruku4(func, t_0, t_f, delta_t1, x_0)
_, x2 = ruku4(func, t_0, t_f, delta_t1/2, x_0)

c = np.max(np.abs(x1[:, 1] - x2[::2, 1]) / (delta_t1**n * (1 - 1/2**n)))

delta_t = .1 * (maxerr / c)**(1/n)
print(delta_t)
t, _x = ruku4(func, t_0, t_f, delta_t, x_0)
k, x = _x.T

fig, ax1 = plt.subplots()
ax1.plot(t[t <= 15], x[t <= 15], label='RK4', linewidth=3)
ax1.grid()
ax1.legend()
fig.tight_layout()
plt.show()

_, _x2 = ruku4(func, t_0, t_f, delta_t/2, x_0)
k2, x2 = _x2.T

real_err = np.max(np.abs(x2[::2] - x) / (1 - 1/2**n))
derv_err = np.max(np.abs(k2[::2] - k) / (1 - 1/2**n))

print('Ejercicio 1:')
print(f'\tError en x: {real_err}')
print(f"\tError en x': {derv_err}")

per = t[np.isclose(x, 0, atol = 1e-2)]
print(f'\tPeríodo: {np.max(2 * (per[1:] - per[:-1]))}')


# Ejercicio 2
##################################################
from Code.Algorithms.Nonlinear_Eqs.Newton_Raphson import newton_raphson

f = lambda x: 2 * np.sqrt(x) + 4 * np.cos(np.sqrt(x)) - 3
deriv = lambda x: (1 - 2 * np.sin(np.sqrt(x))) / np.sqrt(x)
interval = [2, 3]

x_ini = .5 * (interval[1] - interval[0])
maxiter = 100
tol = 1e-16

table = newton_raphson(f, deriv, x_ini = x_ini, maxiter = maxiter, tol = tol, table = True)

print('\nEjercicio 2:')
print(f'\tIntervalo = {interval}')
print(f'\tResultado = {table[-1]}')
print(f'\tValores intermedios = {table} con máximo error {2 * delta_t}')


# Ejercicio 3
##################################################

from Code.Algorithms.Least_Squares.SVD import leastsq_svd
import pandas as pd

df = pd.read_csv('Code/p53.csv').sort_values(by ='x')
x, y = df['x'].to_numpy(), df['y'].to_numpy().reshape(-1, 1)

A = np.zeros((x.size, 3))
A[:, 0] = np.sqrt(np.abs(x))
A[:, 1] = np.cos(A[:, 0])
A[:, 2] = 1

res = leastsq_svd(A, y)
print('\nEjercicio 3:', res.reshape(-1))

plt.plot(x, y, label = 'Original', linewidth = 0, marker = 'o')
plt.plot(x, A.dot(res), label = 'Cuadrados Mínimos')
plt.legend()
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()




