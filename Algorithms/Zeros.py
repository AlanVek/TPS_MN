import numpy as np

def bisec(f, a, b, tol, maxiter):

    fa, fb = f(a), f(b)
    if fa * fb > 0 or maxiter <= 0:
        return None

    for k in range(maxiter):
        c = (a + b) / 2
        fc = f(c)

        if fa * fc < 0: b, fb = c, fc
        else: a, fa = c, fc

        if abs(a - b) < 2 * tol:
            break

    return c, k + 1


def fixed_point(f, maxiter = 1000, x = 1, tol = 1e-6):
    xk_1 = xk = x
    for k in range(maxiter):
        xk = f(xk_1)
        if abs(xk - xk_1) < tol: break
        xk_1 = xk

    return xk, k

func_bisec = lambda x: x - np.exp(np.sqrt(x**2 + .5)) - x * (1 + x**10) / (1 + x) - 9.115 * x

func_fp = lambda x: np.exp(np.sqrt(x**2 + .5)) - x * (1 + x**10) / (1 + x) - 9.115 * x



res_b = bisec(func_bisec, 0, 3, tol = 1e-9, maxiter = 100)
print(res_b)



res_fp = fixed_point(func_fp, tol = 1e-9, maxiter = 100, x = 0)
print(res_fp)

print(np.sin(res_fp[0]))
