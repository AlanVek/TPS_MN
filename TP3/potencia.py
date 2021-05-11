import numpy as np

# Exponentiation by squaring
def powerint(x : float, p : int) -> float:
    if not x: return 0 if p > 0 else np.inf if p < 0 else 1

    res, k, p = 1, x if p > 0 else 1/x, abs(p)
    while p > 0:
        if p%2: res *= k
        k, p = k * k, p//2

    return res

def newt_raph_power(q, k, maxiter, x_ini, tol = 1e-15):
    """ Solves x**q - k = 0 """

    for i in range(maxiter):
        x_k1 = x_ini / q * (q - 1 + k * powerint(x_ini, -q))
        if abs(x_k1 - x_ini) <= tol: return x_k1
        x_ini = x_k1

    return x_ini

# Para no usar np.sign
def sign(x): return 0 if not x else -1 if x < 0 else 1

def powerrat(x : float, p : int, q : int) -> float:
    if x == 1 or not p: return 1
    if not q: return np.inf if p > 0 else 0

    newp, newq = abs(p), abs(q)

    if newq == 1: return powerint(x, p * sign(q))
    if newp == newq: return powerint(x, sign(p * q))

    k = powerint(x, newp%newq)
    k2 = powerint(x, newp//newq)

    res = newt_raph_power(newq, k, 200, x) * k2
    return powerint(res, sign(p * q))

def test():
    tot = 500
    for i in range(tot):
       x = np.random.randint(1, 50)
       p = np.random.randint(-100, 100)
       q = np.random.randint(1, 50) * (-1)**np.random.randint(0, 2)

       real = x**(p/q)
       calc = powerrat(x, p, q)
       worked = abs(calc - real) / real <= 1e-3

       if not worked:
           print('Failed')
           return

    print('Worked')