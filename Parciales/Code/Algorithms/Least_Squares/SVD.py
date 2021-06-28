import numpy as np

def svd(A : np.array) -> np.array:
    (h, w), B = A.shape, A.T.dot(A)
    eigval, V = np.linalg.eig(B)

    idx = eigval.argsort()[::-1]
    singval, V = np.sqrt(np.abs(eigval[idx])), np.real(V[:, idx])
    mat_range = np.count_nonzero(~np.isclose(eigval, 0))

    U = A.dot(V[:, : mat_range]) / singval[ : mat_range]

    V = V[:, :mat_range]

    sigma = singval[:mat_range]

    return U, sigma, V

def pinv(A : np.array) -> np.array:
    u, sigma, v = svd(A)
    return (v/sigma).dot(u.T)

def cuadmin(A : np.array, b : np.array) -> np.array:
    res = pinv(A).dot(b)
    return res, np.linalg.norm(A.dot(res) - b)

##############################################################################

if __name__ == '__main__':

    from scipy.linalg import lstsq

    tests = 1000

    for i in range(tests):
        A = np.random.randint(-10, 10, tuple(np.random.randint(10, 100, 2)))
        b = np.random.randint(-1, 10, (A.shape[0], 1))

        worked = np.allclose(lstsq(A, b)[0], cuadmin(A,b)[0])

        if not worked:
            print('Failed')
            exit()

    print('Worked')