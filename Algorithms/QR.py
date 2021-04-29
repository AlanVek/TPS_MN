import numpy as np
from Solve_Triangular import solve_triangular
from scipy.linalg import lstsq

def qr(base, safe = False, det = False):
    h, w = base.shape

    Q, R = np.zeros((h, h)), np.zeros((h, w))
    R[0, 0] = np.linalg.norm(base[:, 0])
    Q[:, 0] = base[:, 0] / R[0, 0]

    hr = 1

    for i in range(1, max(h, w)):
        must_rand = False
        if i < w:
            R[:i, i] = base[:, i].dot(Q[:, :i])
            if det and not np.allclose(base[i:, i], 0): hr += 1
            subs = base[:, i] - R[:i, i].dot(Q[:, :i].T)

            if safe and np.allclose(subs, 0):
                must_rand = True

        if i >= w or must_rand:
            v = np.random.rand(h)
            subs = v - v.dot(Q[:, : i]).dot(Q[:, : i].T)

        if i < h:
            Q[:, i] = subs
            n = np.linalg.norm(Q[:, i])
            Q[:, i] /= n
            if i < w and not must_rand: R[i, i] = n

    if not det: return Q, R
    return Q, R, hr

def leastsq_qr(A : np.array, b : np.array, safe = False) -> np.array:
    if A.shape[1] > A.shape[0]: return np.zeros(b.shape)

    q, r = qr(A.T.dot(A), safe = safe)
    y = q.T.dot(A.T.dot(b))
    return solve_triangular(r, y, lower = False)

def det_qr(A : np.array, safe = False) -> float:
    q, r, hr = qr(A, safe = safe, det = True)
    print(hr)
    return np.prod(np.diagonal(r)) * (-1) ** (hr)

def inverse_qr(A : np.array, safe = False) -> np.array:
    if A.shape[0] != A.shape[1]: raise Exception('Only square matrixes have an inverse')
    q, r = qr(A, safe = safe)
    if np.count_nonzero(np.diagonal(r) == 0) > 0: raise Exception ('Singular matrix')

    return solve_triangular(r, np.eye(*A.shape), lower = False).dot(q.T)


from time import time
if __name__ == '__main__':
    # tests = 1000
    # start = time()
    # for i in range(tests):
    #
    #     A = np.random.randint(-10, 10, (np.random.randint(50, 200), np.random.randint(2, 51)))
    #     b = np.random.randint(-10, 10, (A.shape[0], 1))
    #
    #     worked = np.allclose(lstsq(A, b)[0], leastsq_qr(A, b))
    #     if not worked:
    #         print('Failed')
    #         print(lstsq(A, b)[0].ravel())
    #         print(leastsq_qr(A, b).ravel())
    #         exit()
    #
    # print(time() - start)
    # print('Worked')

    A = np.random.randint(-10, 10, (5, 5))

    print(np.linalg.det(A))
    print(det_qr(A))
