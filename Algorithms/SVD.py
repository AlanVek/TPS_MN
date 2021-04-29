import numpy as np
from scipy.linalg import lstsq

def svd(A : np.array) -> np.array:

    (h, w), B, mat_range = A.shape, A.T.dot(A), min(A.shape)
    eigval, V = np.linalg.eig(B)

    idx = eigval.argsort()[::-1]
    eigval, V = np.sqrt(np.abs(eigval[idx])), np.real(V[:, idx])

    U = np.zeros((h, h))
    U[:h, : mat_range] = np.dot(A, V[:, :mat_range]) / eigval[:mat_range]

    for i in range(w, h):
        new_v = np.random.rand(h)
        P = new_v.dot(U[:, : i]).dot(U[:, : i].T)

        U[:, i] = new_v - P
        U[:, i] /= np.linalg.norm(U[:, i])

    sigma = np.zeros(A.shape)
    np.fill_diagonal(sigma, eigval[:mat_range])

    return U, sigma, V

def leastsq_svd(A : np.array, b : np.array) -> np.array:
    u, sigma, v = svd(A)
    np.fill_diagonal(sigma, 1 / np.diagonal(sigma))
    return v.dot(sigma.T).dot(u.T).dot(b)

if __name__ == '__main__':

    tests = 1000

    for i in range(tests):
        A = np.random.randint(-10, 10, tuple(np.random.randint(10, 100, 2)))
        b = np.random.randint(-1, 10, (A.shape[0], 1))

        worked = np.allclose(lstsq(A, b)[0], leastsq_svd(A,b))

        if not worked:
            print('Failed')
            exit()

    print('Worked')
