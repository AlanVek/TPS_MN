from numpy import inf

# Exponentiation by squaring
def powerint(x : float, p : int) -> float:
    if not x: return 0 if p > 0 else inf if p < 0 else 1
    res, k, p = 1, x if p > 0 else 1/x, abs(p)
    while p > 0:
        if p%2: res *= k
        k, p = k * k, p//2
    return res

def reduce(p, q):
    for i in range(2, min(abs(p), abs(q))+1):
        while p and q and not p%i and not q%i: p, q = p//i, q//i
    return p, q