import numpy as np
import matplotlib.pyplot as plt

# Ejercicio 1
##############################

def jacobi(coef: np.array, y: np.array, tol : [int, float]):

    x_k = np.zeros(coef.shape[0])
    diag = coef.diagonal()

    tot = 0
    done = False
    while not done:
        tot += 1
        sums_ = coef.dot(x_k.reshape(-1, 1)).reshape(-1) - diag * x_k
        x_temp = (y - sums_) / diag
        if np.all(np.abs(x_k - x_temp) <= tol): done = True
        x_k = x_temp

    return x_k, tot

legajo = 59378

rng = np.random.default_rng(legajo)
M = 5
A = rng.random((M,M))
b = rng.random((M))

for k in range(M):
    A[k,k] = (-1)**k * (A[k].sum()+k)

x1,it1 = jacobi(A,b,legajo*1e-14)

print(x1, it1)

B = np.diag(1 / np.diagonal(A))
mat = np.eye(*A.shape) - B.dot(A)
eigvals = np.linalg.eig(mat)[0]

print('Autovalores de I - B.A:', np.abs(eigvals))
print('Converge con seguridad:', np.all(np.abs(eigvals) < 1))

#################################