import numpy as np
from scipy.sparse import eye as sparse_eye
from Algorithms.Solve_Triangular import solve_triangular

def lu(A : np.array, pivot_nonzero = False):

    h, w = A.shape
    U = A.astype(float)
    
    P = sparse_eye(h, dtype = int).tolil()

    for j in range(min(w-1, h-1)):
        if pivot_nonzero or not U[j, j]:
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


def leastsq_lu(A : np.array, b : np.array) -> np.array:
    p, l, u = lu(A)
    y = p.T.dot(b)
    return solve_triangular(u, solve_triangular(l, y, lower = True), lower = False)


if __name__ == '__main__':
    print('Working...')
    for i in range(10):

         mat = np.random.randint(-100, 100, (5, 5))
         P, L, U = lu(mat, pivot_nonzero=True)


         recreated = P.dot(L.dot(U))
         worked = np.allclose(mat, recreated, atol=.1)

         if not worked:
             print(f'Failed at i = {i}')
             print(mat)
             print(recreated)
             exit()

    print('Worked!')
#input()


