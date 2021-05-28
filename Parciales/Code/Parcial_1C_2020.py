# Ejercicio 1
##################################################

from Code.Algorithms.EDOs.Euler import euler
import numpy as np

def f(t, _x):
    k, x = _x
    out_k = -10 - np.sin(10 * t) - 2 * k
    out_x = k

    return [out_k, out_x]

delta_t = .1
t_0, t_f = 0, .3
x_0 = [1, 1]

t, _x = euler(f = f, t_0 = t_0, tf = t_f, delta_t = delta_t, x_0 = x_0)
k, y = _x.T
print('Ejercicio 1:', y[np.argmin(np.abs(t - .2))])


# Ejercicio 2
##################################################

from Code.Algorithms.dec2bin import binf2dec

num = np.array([0, 0] + [1] * 7 + [0] + [1] * 2 + [0] * 52)
print('\nEjercicio 2:', binf2dec(num, ne = 11))

# Ejercicio 4
##################################################

A = np.array([
    [1, -2],
    [2, -4],
    [3, -6]
])
vals = np.linalg.eigvals(A.T.dot(A))
print('\nEjercicio 4: ')
print('\tValores singulares:', np.sqrt(vals))
print('\tRango:', np.sum(~np.isclose(vals, 0)))


# Ejercicio 5
##################################################

from Code.Algorithms.Nonlinear_Eqs.Secante import secante

f = lambda x: x**3 - 3

x_0, x_1 = 1, 2

# Pongo maxiter = 2 para que me devuelva x3.
print('\nEjercicio 5:', secante(f = f, x_ini1 = x_0, x_ini2 = x_1, maxiter = 2, tol = 1e-3, table = True))