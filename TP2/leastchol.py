import numpy as np
from scipy.linalg import lstsq
from Solve_Triangular import solve_triangular

def cholesky(A : np.ndarray) -> np.ndarray:
    h, w = A.shape
    G = np.zeros((w, h))

    if h == w:
        # if np.allclose(A, A.T) and np.all(np.linalg.eigvals(A) > 0):
        for i in range(h):
            G[i, i] = np.sqrt(A[i, i] - np.dot(G[i, :i], G[i, :i]))
            G[i + 1:w, i] = (A[i, i+1:w] - G[i + 1:w, :i].dot(G[i, :i])) / G[i, i]
        # else:
        #     raise Exception('Input error: Matrix must be positive semidefinite')
    else:
        raise Exception('Dimension error: Matrix must be square')
    return G

def leastsq(A : np.ndarray, b : np.ndarray) -> np.ndarray:
    G = cholesky(A.T.dot(A))
    w = solve_triangular(G, A.T.dot(b), lower=True)
    return solve_triangular(G.T, w, lower=False)

def test() -> None:

    h = 200
    w = 200
    minlim = -50
    maxlim = 50
    num_tests = 50

    for i in range(num_tests):
        A = np.random.randint(minlim, maxlim, (h, w))
        y = np.random.randint(minlim, maxlim, (h, 1))

        # Verificación con SciPy
        x2 = lstsq(A, y)[0]

        # Implementación
        x1 = leastsq(A, y)

        if not np.allclose(x1, x2):
            print('Failed')
            return

    print('Ok!')

if __name__ == '__main__':

    test()
