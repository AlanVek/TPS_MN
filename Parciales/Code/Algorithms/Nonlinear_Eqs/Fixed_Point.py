import numpy as np

def fixed_point(f, x, tol, maxiter, table = False):
    xk = [x]
    for k in range(maxiter):
        xk = np.append(xk, f(x))
        if abs(xk[-1] - xk[-2]) <= tol: break

    return xk if table else xk[-1]
