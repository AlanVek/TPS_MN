import numpy as np
from Code.Algorithms.Least_Squares.Solve_Triangular import solve_triangular

# Si la matriz no es semidefinida positiva, dará error en algún paso.
def cholesky(A : np.array) -> np.array:
    h, w = A.shape
    G = np.zeros((w, h))

    if h == w:
        for i in range(h):
            G[i, i] = np.sqrt(A[i, i] - G[i, :i].dot(G[i, :i]))
            G[i + 1:, i] = (A[i, i+1:] - G[i + 1:, :i].dot(G[i, :i])) / G[i, i]
    else:
        raise Exception('Dimension error: Matrix must be square')
    return G

def leastsq_chol(A : np.array, b : np.array) -> np.array:
    G = cholesky(A.T.dot(A))
    w = solve_triangular(G, A.T.dot(b), lower=True)
    return solve_triangular(G.T, w, lower=False)

def det_chol_abs(A : np.array) -> float:
    try:
        G = cholesky(A.T.dot(A))
        return np.prod(np.diagonal(G))
    except RuntimeWarning: return 0


def inverse_chol(A : np.array) -> np.array:
    if A.shape[0] != A.shape[1]: raise Exception ('Matrix must be square')

    G_inv = solve_triangular(cholesky(A.T.dot(A)), np.eye(*A.shape), lower = True)
    return G_inv.T.dot(G_inv.dot(A.T))


if __name__ == '__main__':
     from scipy.linalg import lstsq
     tests = 1000
    
     for i in range(tests):
    
         A = np.random.randint(-10, 10, (np.random.randint(50, 200), np.random.randint(2, 51)))
         b = np.random.randint(-10, 10, (A.shape[0], 1))
    
         worked = np.allclose(lstsq(A, b)[0], leastsq_chol(A, b))
    
         if not worked:
             print('Failed')
             exit()
    
     print('Worked')
     input()