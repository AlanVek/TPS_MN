from QR import qr
import numpy as np

def eigenvalues(A, method = 'qr'):
    if A.shape == (1, 1): return np.array(A[0])
    T = A.copy()
    ITER = 5500
    for i in range(ITER):
        q, r = qr(T)
        T = r.dot(q)
    res, tot = np.zeros(T.shape[0], dtype = complex), 0
    if abs(T[1, 0]) < 1e-12: res[0], tot = T[0, 0], 1
    for i in range(T.shape[0]-1):
        nosecond = False
        if i < T.shape[0] - 2:
            nosecond = abs(T[i+1, i]) < 1e-12 and not abs(T[i+2, i+1]) < 1e-12
        newvals = solve_eig_2by2(T[i:i+2, i:i+2], nosecond=nosecond)
        res[tot : newvals.size + tot] = newvals
        tot += newvals.size
        if tot >= T.shape[0]: break
    return res

def solve_eig_2by2(T, nosecond):
    if not T.shape == (2, 2): raise Exception('Pifiaste')
    if abs(T[1, 0]) < 1e-6:
        return np.array([T[1, 1]])[int(nosecond):]
    return np.roots([1, -T[0,0] - T[1,1], T[1,1]*T[0,0] - T[1, 0] * T[0, 1]])


def roots(poly):
    poly = poly[np.argmax(poly != 0) : ]
    if poly.size <= 1: return np.array([])

    zeros_first = np.argmax(poly[::-1] != 0)

    res = [0] * zeros_first
    poly = poly[ : poly.size - zeros_first]
    if poly.size <= 1: return np.array([])

    A = np.zeros((poly.size - 1, poly.size - 1))

    A[:, -1] = -poly[1:][::-1] / poly[0]
    A += np.eye(*A.shape, -1)

    return np.append(res, eigenvalues(A))


if __name__ == '__main__':
    #
    # print('Working...')
    # for i in range(10):
    #
    #     A = np.random.randint(-10, 10, (13, 13))
    #
    #     eig_numpy = np.linalg.eig(A)[0]
    #     eig = eigenvalues(A)
    #
    #     worked = np.allclose(np.sort(eig), np.sort(eig_numpy))
    #
    #     if not worked:
    #         print('Failed')
    #         exit()
    #
    # print('Worked')

    for i in range(50):
        poly = np.random.randint(-10, 10, np.random.randint(2, 10))
        print(poly)
        my_roots = roots(poly)
        print(np.allclose(np.sort(np.roots(poly)), np.sort(my_roots)))
