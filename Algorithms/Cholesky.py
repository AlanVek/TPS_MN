import numpy as np
from Solve_Triangular import solve_triangular
from scipy.linalg import lstsq

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

def leastsq_chol(A : np.ndarray, b : np.ndarray) -> np.ndarray:
    G = cholesky(A.T.dot(A))
    w = solve_triangular(G, A.T.dot(b), lower=True)
    return solve_triangular(G.T, w, lower=False)


if __name__ == '__main__':
    tests = 1000

    for i in range(tests):

        A = np.random.randint(-10, 10, (np.random.randint(50, 200), np.random.randint(2, 51)))
        b = np.random.randint(-10, 10, (A.shape[0], 1))

        worked = np.allclose(lstsq(A, b)[0], leastsq_chol(A, b))

        if not worked:
            print('Failed')
            exit()

    print('Worked')