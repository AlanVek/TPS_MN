import numpy as np
from scipy.linalg import lstsq

def cholesky(A : np.array) -> np.array:
    h, w = A.shape
    G = np.zeros((w, h))

    if h == w:
        if np.all(np.linalg.eig(A)[0] > 0):
            for i in range(h):
                G[i, i] = np.sqrt(A[i, i] -  G[i, :i].dot(G[i, :i]))
                G[i+1:w, i] = (A[i, i+1:w] - G[i+1:w, :i].dot(G[i, :i])) / G[i, i]
        else:
            raise ValueError('Input error: Matrix must be positive semidefinite')
    else:
        raise ValueError('Input error: Matrix must be square')
    return G

def leastsq(A : np.array, b : np.array) -> np.array:
    if len(b.shape) == 1 or not b.shape[1] == 1: 
        raise ValueError('Input error: Independent terms must be a column vector')
        
    elif A.shape[0] < A.shape[1]:
        raise ValueError ("Input error: Matrix can't be rank-deficient")
    
    G = cholesky(A.T.dot(A))
    w = solve_triangular(G, A.T.dot(b), lower=True)
    return solve_triangular(G.T, w, lower=False)

def solve_triangular(G : np.array, y : np.array, lower = False) -> np.array:
    h, w = G.shape
    res = np.zeros((h, 1))

    if h == w:
        for i in np.arange(h)[::(-1) ** (not lower)]:
            res[i] = (y[i] - np.dot(G[i], res)) / G[i, i]

    return res

def test() -> None:

    num_tests = 150

    for i in range(num_tests):
        h = np.random.randint(100, 201)
        w = np.random.randint(2, 101)
        
        A = -100 + 200 * np.random.rand(h, w)
        y = -100 + 200 * np.random.rand(h, 1)
        
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
    input()