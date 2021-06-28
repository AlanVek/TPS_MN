import numpy as np
from Useful.LU import leastsq_lu

def newton_raphson(f, deriv, x_ini, tol, maxiter, table = False):
    xk = [x_ini]
    for i in range(maxiter):
        xk = np.append(xk, xk[-1] - f(xk[-1]) / deriv(xk[-1]))
        if abs(xk[-1] - xk[-2]) <= tol: break
    return xk if table else xk[-1]

# f debe decibir un array y devolver un array del mismo largo
# jac debe recibir un array devolver una matriz de largo x largo.
def newton_raphson_multi(f, jac, x_ini, tol, maxiter, table = False):
    xk = [np.asarray(x_ini)]
    for i in range(maxiter):
        jacmatrix = np.asarray(jac(xk[-1]))
        new_fk = np.asarray(f(xk[-1])).reshape(-1, 1)
        xk = np.append(xk, [xk[-1] - leastsq_lu(jacmatrix, new_fk).reshape(-1)], axis = 0)
        if np.linalg.norm(xk[-1] - xk[-2]) <= tol: break
    return xk if table else xk[-1]


if __name__ == '__main__':
    f = lambda x: np.array([
        x[0]**2 - 2 * x[1]**2 * np.exp(x[2]),
        - x[0] + x[1]**3  - np.abs(x[2]) * x[1] / x[0],
        x[0] * x[2] * np.exp(x[1]) - np.abs(x[1])**1.3 + np.pi
    ])

    jac = lambda x: np.array([
        # (df1/dx,        df1/dy,                     df1/dz)
        [2 * x[0], -4 * x[1] * np.exp(x[2]), -2 * x[1]**2 * np.exp(x[2])],

        #             (df2/dx,                     df2/dy,                              df2/dz)
        [-1 + np.abs(x[2]) * x[1] / x[0]**2, 3*x[1]**2 - np.abs(x[2])/x[0], -np.sign(x[2]) * x[1]/x[0]],

        # (df3/dx,                               df3/dy,                                       df3/dz)
        [x[2]*np.exp(x[1]), x[0]*x[2]*np.exp(x[1]) - 1.3*np.abs(x[1])**.3 * np.sign(x[1]), x[0] * np.exp(x[1])]])

    maxiter = 50
    tol = 1e-6
    x_ini = np.array([1, 1, 2])

    res = newton_raphson_multi(f, jac, x_ini, tol, maxiter, table = False)

    print(res)
    print(f(res))
