import numpy as np
from Algorithms.Solve_Triangular import solve_triangular

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

def QR(base):
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
    q, r = QR(A)
    y = q.T.dot(b)
    return solve_triangular(r, y, lower = False)


if __name__ == '__main__':
    A = np.random.randint(-10, 10, (120, 130))
    Q, R = QR(A)

    worked_a = np.allclose(Q.dot(R), A)
    worked_q = np.allclose(Q.dot(Q.T), np.eye(*Q.shape))
    worked_r = np.allclose(np.tril(R, -1), np.zeros(R.shape))

    print(f'Worked: {worked_a and worked_q and worked_r}')

