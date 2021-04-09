import numpy as np
from Solve_Triangular import solve_triangular
from scipy.linalg import lstsq

def gram_schmidt(base):
    h, w = base.shape
    GS, R = np.zeros((h, min(h, w))), np.zeros((h, w))

    R[0, 0] = np.linalg.norm(base[:, 0])
    GS[:, 0] = base[:, 0] / R[0, 0]

    for i in range(1, w):
        R[ : i , i] = base[:, i].dot(GS[:, : i])

        if i < min(h, w):
            GS[:, i] = base[:, i] - (R[ : i, i] * GS[:, : i]).sum(axis=1)
            R[i, i] = np.linalg.norm(GS[:, i])
            GS[:, i] /= R[i, i]

    return GS, R

def qr(base):
    h, w = base.shape

    Q = np.zeros((h, h))
    Q[:, : w], R = gram_schmidt(base)

    for i in range(w, h):
        new_v = np.random.rand(h)
        P = new_v.dot(Q[:, : i]) * Q[:, : i]

        Q[:, i] = new_v - P.sum(axis=1)
        Q[:, i] /= np.linalg.norm(Q[:, i])

    return Q, R

def leastsq_qr(A : np.array, b : np.array) -> np.array:
    q, r = qr(A.T.dot(A))
    y = q.T.dot(A.T.dot(b))
    return solve_triangular(r, y, lower = False)


if __name__ == '__main__':
    tests = 1000
    for i in range(tests):

        A = np.random.randint(-10, 10, (np.random.randint(50, 200), np.random.randint(2, 51)))
        b = np.random.randint(-10, 10, (A.shape[0], 1))

        worked = np.allclose(lstsq(A, b)[0], leastsq_qr(A, b))

        if not worked:
            print('Failed')
            exit()

    print('Worked')


