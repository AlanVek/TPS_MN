import numpy as np
from Algorithms.Solve_Triangular import solve_triangular

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
    AT_A = np.dot(A.T, A)
    AT_b = np.dot(A.T, b)

    G = cholesky(AT_A)
    w = solve_triangular(G, AT_b, lower=True)
    return solve_triangular(G.T, w, lower=False)


if __name__ == '__main__':

    A = np.random.randint(-10, 10, (5, 5))

    G = cholesky(A.T.dot(A))

    print(np.allclose(G.dot(G.T), A.T.dot(A)))

    print(A.T.dot(A))
    print(G.dot(G.T))

    #input()