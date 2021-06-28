import numpy as np

legajo = 59378

# Ejercicio 3
###########################################

def cuadmin(A : np.array, b : np.array) -> np.array:
    h, w = A.shape
    A_T_A = A.T.dot(A)
    G = np.zeros((w, w))
    G[0, 0] = np.sqrt(A_T_A[0, 0])
    G[1, 0] = A_T_A[0, 1] / G[0, 0]
    G[1, 1] = np.sqrt(A_T_A[1, 1] - G[1, 0]**2)

    B = A.T.dot(b)
    res1 = np.zeros(B.shape)

    res1[0] = B[0] / G[0, 0]
    res1[1] = (B[1] - G[1, 0] * res1[0]) / G[1, 1]

    res2 = np.zeros(res1.shape)
    G2 = G.T
    res2[1] = res1[1] / G2[1, 1]
    res2[0] = (res1[0] - G2[0, 1] * res2[1]) / G2[0, 0]

    return res2, np.linalg.norm(A.dot(res2) - b)

rng = np.random.default_rng(legajo)
M = 1000
A = rng.random((M,2))
b = legajo*A[:,0]+legajo/2*A[:,1]+rng.random((M))

res, err = cuadmin(A, b)

print('Resultado:', res.reshape(-1).round(4))
print('Norma del error:', err.round(4))
