import numpy as np

def secante(f, x_ini1, x_ini2, tol, maxiter, table=False):
    xk = [np.asarray(x_ini1), np.asarray(x_ini2)]
    fk = [np.asarray(f(x_ini1)), np.asarray(f(x_ini2))]

    for i in range(maxiter):
        xk = np.append(xk, [xk[-1] - fk[-1] / (fk[-1] - fk[0]) * (xk[-1] - xk[-2])], axis=0)
        if np.linalg.norm(xk[-1] - xk[-2]) <= tol: break
        fk = [fk[1], np.asarray(f(xk[-1]))]

    return xk if table else xk[-1]


if __name__ == '__main__':
    f = lambda x: [x[0] ** 2 - np.pi * 2 * x[0] + 3, x[0] - 2 * x[1] ** 2]
    x_ini1, x_ini2 = [1, 1],[2, 2]
    maxiter = 100
    tol = 1e-6

    res = secante(f, x_ini1, x_ini2, tol, maxiter)

    print(res)
    print(f(res))