import numpy as np
from scipy.sparse import eye as sparse_eye
from Solve_Triangular import solve_triangular
from scipy.linalg import lstsq

def lu(A : np.array, pivot_nonzero = False):
    h, w = A.shape
    U = A.astype(float)
    P = sparse_eye(h, dtype = int).tolil()

    for j in range(min(w, h)):
        if pivot_nonzero or np.isclose(U[j, j], 0):
            f_max = np.argmax(np.abs(U[j:, j])) + j
            U[[j, f_max]] = U[[f_max, j]]
            inter_j, inter_f = P[[j, f_max]].nonzero()[1]
            P[j, inter_j] = P[f_max, inter_f] = 0
            P[j, inter_f] = P[f_max, inter_j] = 1

        U[j + 1 :, j] = U[j + 1:, j] / U[j, j]
        U[j + 1:, j + 1:] = U[j + 1 :, j + 1 :] - U[j, j + 1:] * U[j + 1 :, [j]]

    L = np.tril(U, k = -1)
    if h > w: U = U[:w - h]
    if h < w: L = L[:, :h - w]
    return P.tocsr().T, L + np.eye(*L.shape), np.triu(U)


def leastsq_lu(A : np.array, b : np.array, pivot_nonzero = False) -> np.array:
    p, l, u = lu(A.T.dot(A), pivot_nonzero = pivot_nonzero)
    y = p.T.dot(A.T.dot(b))
    return solve_triangular(u, solve_triangular(l, y, lower = True), lower = False)

def det_p(p):
    max_row, tot = p.nonzero()[1], 1
    for i in range(max_row.size):
        k = max_row[i]
        if k != i:
            max_row[[k, i]], tot = max_row[[i, k]], -tot
    return tot

def det_lu(A : np.array, pivot_nonzero = False) -> float:
    if A.shape[0] != A.shape[1]: raise Exception('Matrix must be square')
    p, l, u = lu(A, pivot_nonzero = pivot_nonzero)
    return det_p(p) * np.prod(np.diagonal(u))


def test_leastsq():
    tests = 1000

    for i in range(tests):

        A = np.random.randint(-10, 10, (np.random.randint(100, 201),np.random.randint(2, 101)))
        b = np.random.randint(-10, 10, (A.shape[0], 1))

        worked = np.allclose(lstsq(A, b)[0], leastsq_lu(A, b))

        if not worked:
            print('Failed leastsq')
            return

    print('Worked leastsq')


def test_det():
    tests = 1000

    for i in range(tests):
        h = np.random.randint(2, 201)
        A = np.random.rand(h, h)

        worked = np.isclose(det_lu(A), np.linalg.det(A), atol = 1e-3)

        if not worked:
            print('Failed det')
            return

    print('Worked det')

if __name__ == '__main__':
    # test_leastsq()
    test_det()

