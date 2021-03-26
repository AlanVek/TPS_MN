import numpy as np
from scipy.linalg import lstsq

def cholesky(A : np.ndarray) -> np.ndarray:
    h, w = A.shape
    G = np.zeros((w, h))

    if h == w:
        for i in range(h):
            for j in range(i, w):
                G[j, i] = A[i, j] - np.dot(G[i], G[j])
                G[j, i] /= np.sqrt(G[i, i]) ** (1 + int(i != j))

    return G

def solve_triangular(G : np.ndarray, y : np.ndarray, lower = False) -> np.ndarray:
    h, w = G.shape
    res = np.zeros((h, 1))

    if h == w:
        for i in np.arange(h)[::(-1) ** (not lower)]:
            res[i] = (y[i] - np.dot(G[i], res)) / G[i, i]

    return res

def leastsq(A : np.ndarray, b : np.ndarray) -> np.ndarray:
    AT_A = np.dot(A.T, A)
    AT_b = np.dot(A.T, b)

    G = cholesky(AT_A)
    w = solve_triangular(G, AT_b, lower=True)
    return solve_triangular(G.T, w, lower=False)

def test(h : int, w : int, minlim : [int, float], maxlim : [int, float]) -> bool:

    if h <= 0 or w <= 0: return False

    A = np.random.randint(minlim, maxlim, (h, w))
    y = np.random.randint(minlim, maxlim, (h, 1))

    # Verificación con NumPy
    x2 = np.linalg.lstsq(A, y, rcond = None)[0]

    # Verificación con SciPy
    x3 = lstsq(A, y)[0]

    # Implementación
    x1 = leastsq(A, y)

    return np.allclose(x1, x2) and np.allclose(x1, x3)

if __name__ == '__main__':

    h, w, minlim, maxlim = 120, 120, -50, 50

    tests = 50

    print('###############################')

    print(f'Tests amount | {tests}')

    print(f'Matrix shape | {h}x{w}')

    print(f'Value limits | [{minlim}, {maxlim})')

    print('###############################')

    print('\nResult: ', end = '')

    for i in range(tests):
        if not test(h, w, minlim, maxlim):
            print('Failed')
            exit()

    print('Worked')
