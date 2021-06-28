import numpy as np
from scipy.optimize import minimize
import pandas as pd


# Minimización monovariable con proporción áurea
def findalpha(f, a, b, tol, maxiter):
    r = (-1 + np.sqrt(5))/2
    c, d = a + (1 - r) * (b - a), a + r * (b - a)
    f1, f2 = f(c), f(d)

    for i in range(maxiter):
        if f2 > f1:
            b, d, f2 = d, c, f1
            c = a + (1 - r) * (b - a)
            f1 = f(c)
        else:
            a, c, f1 = c, d, f2
            d = a + r * (b - a)
            f2 = f(d)

        if abs(b - a) < 2 * tol: break

    return (b + a)/2


def minimi(func, grad, x0, tol, maxiter):
    x_k = x_k1 = x0
    for i in range(maxiter):
        descent_dir = -grad(x_k)

        alpha_k = findalpha(lambda a: func(x_k + a * descent_dir), 0, 5, 1e-10, 200)

        x_k1 = x_k + alpha_k * descent_dir
        if np.linalg.norm(x_k - x_k1) < tol: break
        x_k = x_k1

    return x_k1


def test():
    base1 = lambda x: x[0] + 2 * x[1] - 7
    base2 = lambda x: 2 * x[0] + x[1] - 5
    func = lambda x: base1(x)**2 + base2(x)**2

    def gr(x):
        bx1, bx2 = base1(x), base2(x)
        return np.array([
            2 * bx1 + 4 * bx2,
            4 * bx1 + 2 * bx2
        ])

    tol, maxiter = 1e-10, 100
    x0 = [-1.5, 2.5]

    result = minimi(func, gr, x0, tol = tol, maxiter = maxiter)
    verif = minimize(func, x0 = x0, tol = tol)['x']

    print('Test general: ' + 'Worked' if np.allclose(result, verif) else 'Failed')

data = pd.read_csv('data_métodos.txt', sep=' ')
ti, yi = data['ti'].to_numpy(), data['yi'].to_numpy()

def base(x):
    a, b, T1, c, T2 = x
    return (yi - a - b * np.cos(2 * np.pi * ti / T1) - c * np.cos(2 * np.pi * ti / T2))

def f(x):
    bx = base(x)
    return bx.dot(bx)

def gradient(x):
    a, b, T1, c, T2 = x
    bx = base(x)
    da = -2 * bx.sum()
    db = (-2 * bx * np.cos(2 * np.pi * ti / T1)).sum()
    dT1 = (2 * bx * b * np.sin(2 * np.pi * ti / T1) * -2 * np.pi * ti / T1 ** 2).sum()
    dc = (-2 * bx * np.cos(2 * np.pi * ti / T2)).sum()
    dT2 = (2 * bx * c * np.sin(2 * np.pi * ti / T2) * -2 * np.pi * ti / T2 ** 2).sum()

    return np.array([da, db, dT1, dc, dT2])

#     a   b   T1   c   T2
x0 = [36, 0, 85e3, 0, 85e3]
tol = 1e-10
maxiter = 200

def temperatura(): 

    result =  minimi(f, gradient, x0, tol, maxiter)
    verif =  minimize(f, x0=x0, tol=tol)['x']

    return result, np.abs(result - x0)

def test_temp():
    result, error = temperatura()
    verif = minimize(f, x0 = x0, tol = tol)['x']
    print('Test2: ' + 'Worked' if np.allclose(f(result), f(verif)) else 'Failed')


test()
test_temp()
input()