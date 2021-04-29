import numpy as np
from Powerint import powerint, reduce

def bisec(f, a, b, tol : float, maxiter : int):

    fa, fb = f(a), f(b)
    if not fa: return a
    if not fb: return b
    if np.sign(fa) == np.sign(fb) or maxiter <= 0: return None

    for k in range(maxiter):
        c = (a + b) / 2
        fc = f(c)
        if not fc: return c

        if np.sign(fa) != np.sign(fc): b, fb = c, fc
        else: a, fa = c, fc

        if abs(a - b) <= 2*tol: return c

    return c

def powerrat(x : float, p : int, q : int) -> float:
    if not p: return 1
    if not q: return np.inf if p > 0 else 0

    p, q = reduce(p, q)
    div_int, resto = p//q, p%q/q

    k = powerint(x, abs(p))
    if abs(q) == 1: return powerint(k, np.sign(p * q))

    func = lambda r: powerint(r, q * np.sign(p)) - k
    return bisec(func, 0, k, 1e-15, 100)

if __name__ == '__main__':
    for i in range(10000):
       x = np.random.randint(1, 10)
       p = np.random.randint(-20, 20)
       q = np.random.randint(1, 50) * (-1) ** np.random.choice([0, 1])

       real = x**(p / q)
       worked = abs(powerrat(x, p, q) - real) / real <= 1e-3 # 0.1%

       if not worked:
           print('Pifiaste:', x, p, q)
           exit()

    print('Worked')

    r1 = 2**(3/5)
    r2 = powerrat(2, 3, 5)
