import numpy as np

def bisec(f, a : float, b : float, tol : float = 1e-15, maxiter : int = 100):

    """ Solves f(x) = 0 """

    fa, fb = f(a), f(b)
    if not fa: return a
    if not fb: return b
    if np.sign(fa) == np.sign(fb) or maxiter <= 0: return None

    for _ in range(maxiter):
        c = (a + b) / 2
        fc = f(c)
        if not fc: break

        if np.sign(fa) != np.sign(fc) < 0: b, fb = c, fc
        else: a, fa = c, fc

        if abs(a - b) <= 2 * tol: break

    return c


def fixed_point(f, x, tol, maxiter):

    """ Solves x = f(x) """

    for k in range(maxiter):
        xk = f(x)
        if abs(xk - x) <= tol: break
        x = xk

    return x


def newton_raphson(f, deriv, x_ini, tol, maxiter):

    """ Solves f(x) = 0 """

    for i in range(maxiter):
        x_k = x_ini - f(x_ini) / deriv(x_ini)
        if abs(x_k - x_ini) <= tol: return x_k
        x_ini = x_k

    return x_ini


def secante(f, x_ini1, x_ini2, tol, maxiter):

    """ Solves f(x) = 0 """

    f_1, f_2 = f(x_ini1), f(x_ini2)
    for i in range(maxiter):
        x_k = x_ini2 - f_2 / (f_2 - f_1) * (x_ini2 - x_ini1)
        if abs(x_k - x_ini2) <= tol: return x_k

        x_ini1, f_1, x_ini2, f_2 = x_ini2, f_2, x_k, f(x_k)

    return x_ini2

if __name__ == '__main__':

    f = lambda x: x**3
    deriv = lambda x: 3 * x**2

    g = lambda x: 3 / x**2

    tol, maxiter, x_ini = 1e-9, 1000, -1
    x_ini2 = 3

    # print(secante(f, x_ini, x_ini2, tol, maxiter))
    # print(newton_raphson(f, deriv, x_ini, tol, maxiter))
    print(bisec(f, x_ini, x_ini2, tol, maxiter))
    # print(fixed_point(g, x_ini, tol, maxiter))

