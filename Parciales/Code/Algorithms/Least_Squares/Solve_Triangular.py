import numpy as np

def solve_triangular(G : np.array, y : np.array, lower = False) -> np.array:
    h, w = G.shape
    res = np.zeros(y.shape)

    if h == w:
        for i in np.arange(h)[::(-1) ** (not lower)]:
            res[i] = (y[i] - G[i].dot(res)) / G[i, i]

    return res

def frwd_subs(A : np.array, b : np.array) -> np.array:
    return solve_triangular(A, b, lower = True)

def bkwd_subs(A : np.array, b : np.array) -> np.array:
    return solve_triangular(A, b, lower = False)


if __name__ == '__main__':
    pass