import numpy as np

def solve_triangular(G : np.ndarray, y : np.ndarray, lower = False) -> np.ndarray:
    h, w = G.shape
    res = np.zeros(y.shape)

    if h == w:
        for i in np.arange(h)[::(-1) ** (not lower)]:
            res[i] = (y[i] - np.dot(G[i], res)) / G[i, i]

    return res

def frwd_subs(A : np.ndarray, b : np.ndarray) -> np.ndarray:
    return solve_triangular(A, b, lower = True)

def bkwd_subs(A : np.ndarray, b : np.ndarray) -> np.ndarray:
    return solve_triangular(A, b, lower = False)


if __name__ == '__main__':
    A = np.array([[1, 0, 0], [2, 3, 0], [10, 4, 8]])
    b = np.array([[2], [4], [6]])

    print(frwd_subs(A, b))

    print(bkwd_subs(A[::-1, ::-1], b[::-1])[::-1])

    print(np.linalg.solve(A, b))