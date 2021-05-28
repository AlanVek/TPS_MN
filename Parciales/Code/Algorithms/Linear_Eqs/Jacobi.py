import numpy as np

def jacobi(coef: np.array, y: np.array, tol : [int, float], maxiter : [int], x: np.array = None, table = False):

    if x is None: x_k = np.zeros(coef.shape[0])
    else: x_k = x.astype(float)

    diag = coef.diagonal()

    if table: iters = np.zeros((0, x_k.size))

    for k in range(maxiter):
        if table: iters = np.append(iters, [x_k], axis = 0)

        sums_ = coef.dot(x_k.reshape(-1, 1)).reshape(-1) - diag * x_k
        x_temp = (y - sums_) / diag
        if np.all(np.abs(x_k - x_temp) <= tol): break
        x_k = x_temp

    return (x_k, iters) if table else x_k


if __name__ == '__main__':
    coef = np.array([[6, 2, 1],
                     [-1, 8, 2],
                     [1, -1, 6]])

    y = np.array([22, 30, 23])
    print('Solution:', np.linalg.solve(coef, y))

    print('Jacobi  :', jacobi(coef, y, 1e-9, 100))
