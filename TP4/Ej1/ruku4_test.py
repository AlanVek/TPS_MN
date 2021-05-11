from Ej1.ruku4 import *

# Testing RK4
####################################################################################
def real_f1(t_0, x_0):
    f = lambda t, x: (x + 1) * np.sin(t)
    K = (x_0 + 1) * np.exp(np.cos(t_0))
    real = lambda t: K * np.exp(-np.cos(t)) - 1

    return f, real

def real_f2(t_0, x_0):
    f = lambda x, t: 3 * (x + t)
    K = (x_0 + (3 * t_0 + 1) / 3) * np.exp(-3 * t_0)
    real = lambda t: K * np.exp(3 * t) - (3 * t + 1) / 3

    return f, real

def test():

    tests = [real_f1, real_f2]

    x_0 = -10 + 20 * np.random.rand()
    t_0 = np.random.rand() * 2
    tf = t_0 + np.random.rand() * 10

    N = 1e3
    delta_t = (tf - t_0) / N

    f, real_f = tests[np.random.randint(0, len(tests))](t_0, x_0)

    t, x = ruku4(f, t_0, tf, delta_t, [x_0])
    real = real_f(t)

    if np.allclose(real, x.reshape(-1), atol = 1e-10): print('Worked')
    else: print('Failed')

####################################################################################

