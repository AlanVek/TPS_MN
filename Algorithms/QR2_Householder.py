import numpy as np
from Useful.Solve_Triangular import solve_triangular

# Version con Householder reflections.
def qr(A : np.array, det = False):
    h, w = A.shape
    qt = np.eye(h)
    hr, x = 1, A[:, [0]].astype(float)

    for i in range(min(h - 1, w)):
        n = np.linalg.norm(x)
        x[0], n = x[0] - n, n * (n - x[0])

        if n:
            hr = -hr
            qt[i:] -= x.dot(x.T.dot(qt[i:])) / n
        if i < w-1:
            x = qt[i+1:].dot(A[:, [i+1]])

    qt = qt[:w]
    if not det: return qt.T, qt.dot(A)
    return qt.T, qt.dot(A), hr


def det_qr(A : np.array) -> float:
    if A.shape[0] != A.shape[1]: raise Exception('Matrix must be square')
    q, r, hr = qr(A, det = True)
    return np.prod(np.diagonal(r)) * hr

def inverse_qr(A : np.array) -> np.array:
    if A.shape[0] != A.shape[1]: raise Exception('Matrix must be square')
    q, r = qr(A)
    if np.count_nonzero(np.diagonal(r) == 0) > 0: raise Exception ('Singular matrix')

    return solve_triangular(r, np.eye(*A.shape), lower = False).dot(q.T)


if __name__ == '__main__':

    tests = 100
    for i in range(tests):
        A = np.random.randint(-10, 10, tuple(np.random.randint(2, 200, 2)))
        q, r = qr(A)

        worked = np.allclose(q.dot(r), A)

        if not worked:
            print(A)
            exit()

    print('Worked')

