import numpy as np
from QR import qr as qr_old
from QR2_Householder import qr as qr2
from Useful.Timer import timer
from scipy.sparse import eye as sparse_eye


def givens(a, b):
    res, m = np.zeros(2), 1 - int(abs(a) > abs(b))
    if m: tau = a / b
    else: tau = b / a
    res[m] = 1 / np.sqrt(1 + tau**2)
    res[1 - m] = res[m] * tau
    return res

def givens2(vals):
    a, b = vals
    res, m = np.zeros(2), 1 - int(abs(a) > abs(b))
    if m: tau = a / b
    else: tau = b / a
    res[m] = 1 / np.sqrt(1 + tau**2)
    res[1 - m] = res[m] * tau
    return res


# Funciona bien para matrices no cuadradas.
# Preferentemente matrices más altas que anchas.
# También funciona bien para matrices con muchos ceros.
def qr3(A : np.array):

    (h, w), G, R = A.shape, np.zeros((2, 2)), A.astype(float)
    q = np.eye(h)

    for j in range(min(w, h)):
        if np.count_nonzero(R[:, j]) > 1:
            for i in range(h - 1, j, -1):
                if R[i, j]:
                    c, s = givens(R[i-1, j], R[i, j])

                    G[1, 1] = G[0, 0] = c
                    G[0, 1] = s
                    G[1, 0] = -s

                    q[i-1:i+1] = G.dot(q[i-1:i+1])
                    R[i-1:i+1] = G.dot(R[i-1:i+1])

    return q[:w].T, R[:w]

def qr4(A : np.array):

    (h, w), R = A.shape, A.astype(float)
    q = np.eye(h)

    for j in range(min(w, h)):
        cant_nozeros = R[:, j] != 0
        cant_nozeros[:j] = False
        cant_nozeros[j] = True
        cant = cant_nozeros[cant_nozeros].size
        cnz = cant_nozeros.copy()  # [ : cant_nozeros.size - areodd]

        while cant > 1:
            if cant % 2: cnz[cnz] = np.append(np.full(cant - 1, True), False)
            pairs = np.apply_along_axis(givens2, 1, R[cnz, j].reshape(-1, 2))

            Gx = sparse_eye(h).tolil()
            ke = cnz.copy()
            even_true = np.tile([True, False], cant // 2)
            ke[ke] = even_true
            Gx[cnz, cnz] = np.repeat(pairs[:, 0], 2)
            Gx[ke, cnz & ~ke] = -pairs[:, 1]
            Gx[cnz & ~ke,ke] = pairs[:, 1]

            cant_nozeros[cnz] = cnz[cnz] = even_true
            cnz[cant_nozeros] = True
            cant = cant//2 + cant%2
            Gx = Gx.tocsr().T
            q = Gx.dot(q)
            R = Gx.dot(R)

    return q[:w].T, R[:w]

A = np.random.randint(-10, 10, (2, 2))
# A = np.zeros((1000, 1000))
# A[:175, :175] = 0

# q, r = qr4(A)

# print(np.allclose(A, q.dot(r).round(2)))

print('Old Version:', timer(qr_old, A, safe = True))
print('Givens para:', timer(qr4, A))
print('Givens sequ:', timer(qr3, A))
print('Householder:', timer(qr2, A))
