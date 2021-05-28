# Ejercicio 1
##################################################

import numpy as np
from Code.Algorithms.EDOs.RK4 import ruku4
import matplotlib.pyplot as plt

def f(t, _x):
    k, x = _x
    out_k = 1 - 2 * (x ** 4 - 1) * k - x
    out_x = k

    return np.array([out_k, out_x])

x_0 = [1, 1.5]
t_0, t_f = 0, 250

maxerr = 1e-5
n = 4

# Resolvemos para un delta_t1 y con delta_t1/2
delta_t1 = .1
_, x1 = ruku4(f, t_0, t_f, delta_t1, x_0)
_, x2 = ruku4(f, t_0, t_f, delta_t1/2, x_0)

# Buscamos el valor de la constante del error
c = np.max(np.abs(x1[:, 1] - x2[::2, 1]) / (delta_t1**n * (1 - 1/2**n)))

# Nos quedamos con el 80% del error máximo estimado
delta_t = .8 * (maxerr / c)**(1/n)

# Resolvemos con el nuevo delta_t
t, _x = ruku4(f, t_0, t_f, delta_t, x_0)
k, x = _x.T


plt.plot(t[t <= 75], x[t <= 75], label='RK4', linewidth = 3)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# # FFT para buscar la frecuencia
transf = np.fft.rfft(x[t > 60])
freqs = np.fft.rfftfreq(t[t > 60].size, t[1] - t[0])

# Resolvemos con delta_t/2
_, _x2 = ruku4(f, t_0, t_f, delta_t/2, x_0)
k2, x2 = _x2.T

# Buscamos los errores en x y en x'
real_err = np.max(np.abs(x2[::2] - x) / (1 - 1/2**n))
derv_err = np.max(np.abs(k2[::2] - k) / (1 - 1/2**n))

print('Ejercicio 1:')
print('\tErrores:')
print("\t\tError en x:", real_err)
print("\t\tError en x':", derv_err)
print('\tPeríodo:', 1/freqs[1 + np.argmax(np.abs(transf[1:]))], 'con error máximo de', 2 * delta_t)
# Ejercicio 2
##################################################

from Code.Algorithms.Nonlinear_Eqs.Newton_Raphson import newton_raphson

# Resuelvo para encontrar k = np.sqrt(x/30), y el resultado será x = k**2 * 30

f = lambda k: np.cos(2 * k) - 4 * np.sin(k)
deriv = lambda k: (-2 * np.sin(2 * k) - 4 * np.cos(k))

tol, maxiter, x_ini = 1e-16, 1000, np.sqrt(1 / 30)
vals = newton_raphson(f, deriv, x_ini, tol, maxiter, table = True)

print('\nEjercicio 2:')
print(f'\tIntervalo: [0, {x_ini**2 * 30 * 2}]')
print('\tResultado:', vals[-1]**2 * 30)
print('\tValores intermedios:', vals**2 * 30)

# Ejercicio 3
##################################################

import pandas as pd
from Code.Algorithms.Least_Squares.SVD import leastsq_svd

df = pd.read_csv('Code/p43.csv').sort_values(by ='x')
x, y = df['x'].to_numpy(), df['y'].to_numpy().reshape(-1, 1)

A = np.zeros((x.size, 3))
A[:, 0] = np.cos(np.sqrt(np.abs(x)))
A[:, 1] = np.sin(np.sqrt(np.abs(x)))
A[:, 2] = 1

# SVD
###################################
# Tarda demasiado, pero lo corrí en Colab con GPU y está chequeado que da la respuesta correcta.
# Con LU o QR tarda pocos segundos.

res = leastsq_svd(A, y)
# print('\nEjercicio 3:', res.reshape(-1))
###################################

# from Code.Algorithms.LU import leastsq_lu

# res = leastsq_lu(A, y)

print('\nEjercicio 3:', res.reshape(-1))

plt.plot(x, y, label = 'Original', linewidth = 0, marker = 'o')
plt.plot(x, A.dot(res), label = 'Cuadrados Mínimos')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.tight_layout()
plt.show()