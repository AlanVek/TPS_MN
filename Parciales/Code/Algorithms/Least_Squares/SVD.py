import numpy as np

def svd(A : np.array, full_matrices = True, full_sigma = False) -> np.array:
    """

    Returns U, sigma, V such that A = U * S * V.T, where S depends on the parameter full_sigma

    if full_matrices is True, U has shape (h, h) and V has shape (w, w)
    Otherwise, U has shape (h, K) and V has shape (w, K)

    if full_sigma is True, sigma is a diagonal matrix with correct dimensions to do A = U * sigma * V.T
    Otherwise, it's a vector with the K real and positive singular values of A

    It is recommended to always use full_matrices = False, since it results in much smaller numerical errors.
    This is especially true when working with non-inversible matrices, where numerical errors can grow
    exponentially depending on the matrix's rank-deficiency (which increases its condition number).

    """

    (h, w), B = A.shape, A.T.dot(A)
    eigval, V = np.linalg.eig(B)

    idx = eigval.argsort()[::-1]
    singval, V = np.sqrt(np.abs(eigval[idx])), np.real(V[:, idx])
    mat_range = np.count_nonzero(~np.isclose(eigval, 0))

    U = A.dot(V[:, : mat_range]) / singval[ : mat_range]

    if full_matrices:
        U = np.append(U, np.random.rand(h, h - mat_range), axis = 1)
        for j in range(mat_range, h):
            U[:, j] -= U[:, j].dot(U[:, :j]).dot(U[:, :j].T)
            norm = np.linalg.norm(U[:, j])
            if norm: U[:, j] /= norm

    else: V = V[:, :mat_range]

    if full_sigma:
        sigma = np.zeros((U.shape[1], V.shape[1]))
        np.fill_diagonal(sigma, singval[:mat_range])
    else:
        sigma = singval[:mat_range]

    return U, sigma, V

def inverse_svd(A : np.array) -> np.array:
    u, sigma, v = svd(A, full_matrices = False, full_sigma = False)
    return (v/sigma).dot(u.T)

def pinv(A : np.array) -> np.array:
    return inverse_svd(A)

def leastsq_svd(A : np.array, b : np.array) -> np.array:
    return pinv(A).dot(b)


if __name__ == '__main__':

    from scipy.linalg import lstsq

    tests = 1000

    for i in range(tests):
        A = np.random.randint(-10, 10, tuple(np.random.randint(10, 100, 2)))
        b = np.random.randint(-1, 10, (A.shape[0], 1))

        worked = np.allclose(lstsq(A, b)[0], leastsq_svd(A,b))

        if not worked:
            print('Failed')
            exit()

    print('Worked')