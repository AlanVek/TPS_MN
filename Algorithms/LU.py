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
            f_max = np.argmax(np.abs(U[j+1:, j])) + j + 1
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



if __name__ == '__main__':
    tests = 1000

    for i in range(tests):

        A = np.random.randint(-10, 10, (np.random.randint(50, 200), np.random.randint(2, 51)))
        b = np.random.randint(-10, 10, (A.shape[0], 1))

        worked = np.allclose(lstsq(A, b)[0], leastsq_lu(A, b))

        if not worked:
            print('Failed')
            exit()

    print('Worked')

#input()


