import numpy as np

def bisec(f, a : float, b : float, tol : float = 1e-15, maxiter : int = 100, table = False):

    """ Solves f(x) = 0 """

    fa, fb = f(a), f(b)
    if not fa: return np.array([a]) if table else a
    if not fb: return np.array([b]) if table else b
    if np.sign(fa) == np.sign(fb) or maxiter <= 0: return np.array([None]) if table else None

    c = []
    for k in range(maxiter):
        c = np.append(c, (a + b) / 2)
        fc = f(c[-1])
        if not fc: break

        if abs(b - a) <= 2 * tol: break

        if np.sign(fa) != np.sign(fc): b, fb = c[-1], fc
        else: a, fa = c[-1], fc

    return (c, b - a) if table else c[-1]


if __name__ == '__main__':
	f = lambda x: x**4 + x**3 + x**2 + x - 40
	tol = 1e-16
	maxiter = 10000
	a, b = 2, 3
	c, err = bisec(f, a, b, tol, maxiter, table = True)
	print(c[-1], err)
	input()
