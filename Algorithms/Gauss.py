import numpy as np
from LU import lu


def gauss(A : np.array, b : np.array) -> np.array:

    if A.shape[1] != A.shape[0] or A.shape[0] != b.shape[0]: raise Exception('Pifiaste con las dimensiones, bro.')

    matrix = np.append(A, b, axis = 1).astype(float)

    if len(matrix) <= 1: return matrix

    if not matrix[0, 0]:
        f_nonzero = np.argmax(matrix[:, 0] != 0)
        matrix[[0, f_nonzero]] = matrix[[f_nonzero, 0]]

    for i in range(1, len(matrix)):
        matrix[i] = matrix[i] - matrix[i, 0] / matrix[0, 0] * matrix[0]

    f0, c0 = matrix[0, 1:], matrix[:, [0]]

    newmatrix = gauss(matrix[1:, 1:-1], matrix[1:, [-1]])
    res = np.append([f0], newmatrix, axis = 0)

    return np.append(c0, res, axis = 1)

def gauss2(A : np.array, b : np.array, pivot_nonzero = False) -> np.array:
    mat = np.append(A, b, axis = 1)
    return lu(mat, pivot_nonzero = pivot_nonzero)[2]

if __name__ == '__main__':

    m = np.array([
        [0, 1, 6, 2],
        [0, 0, 3, 1],
        [7, 1, 4, 0],
        [0, 0, 11,2]
    ])

    b = np.array([
        [4],
        [9],
        [17],
        [33]
    ])

    res = gauss(m, b)
    res2 = gauss2(m, b)
    print(res)
    print(np.linalg.solve(m, b))

    print(np.linalg.solve(res[:, :-1], res[:, [-1]]))
    print(np.linalg.solve(res2[:, :-1], res2[:, [-1]]))
