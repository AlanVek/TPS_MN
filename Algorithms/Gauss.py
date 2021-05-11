import numpy as np
from LU import lu

def gauss(A : np.array, b : np.array, pivot_nonzero = False) -> np.array:
    mat = np.append(A, b, axis = 1)
    return lu(mat, pivot_nonzero = pivot_nonzero)[2]

if __name__ == '__main__':

    m = np.random.randint(-10, 10, (7, 8))
    b = m[:, [-1]]

    print(gauss, m[:, :-1],b)
