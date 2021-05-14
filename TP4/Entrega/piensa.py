import numpy as np
import matplotlib.pyplot as plt

def ruku4(f, t_0, t_f, delta_t, x_0):
    """ Solves dx/dt = f(t, x)

    t_0 -> Number
    t_f -> Number
    x_0 -> List or array

    """

    N = int((t_f - t_0) / delta_t) + 1

    x_k = np.zeros((N, len(x_0)))
    x_k[0] = x_0
    t_k = t_0 + delta_t * np.arange(N)

    for i in range(N-1):
        k1 = f(t_k[i], x_k[i])
        k2 = f(t_k[i] + delta_t/2, x_k[i] + k1 * delta_t/2)
        k3 = f(t_k[i] + delta_t/2, x_k[i] + k2 * delta_t/2)
        k4 = f(t_k[i] + delta_t, x_k[i] + k3 * delta_t)

        x_k[i+1] = x_k[i] + (k1 + 2 * k2 + 2 * k3 + k4) * delta_t / 6

    return t_k, x_k

# Test
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
    
    # Test RK4
    ##################################################
    
    tests = [real_f1, real_f2]

    x_0 = -10 + 20 * np.random.rand()
    t_0 = np.random.rand() * 2
    tf = t_0 + np.random.rand() * 10

    N = 1e3
    delta_t = (tf - t_0) / N

    f, real_f = tests[np.random.randint(0, len(tests))](t_0, x_0)

    t, x = ruku4(f, t_0, tf, delta_t, [x_0])
    real = real_f(t)

    if np.allclose(real, x.reshape(-1), atol = 1e-10): print('RK Worked')
    else: print('RK Failed')
        
    ##################################################
    
    # Plot de HH
    ##################################################
        
    i = lambda t: i_0
    x_0 = [v_0, n_0, m_0, h_0]

    t, res = ruku4(hodgkinhuxley, t_0 = 0, t_f = 200, delta_t = .1, x_0 = x_0)

    v, n, m, h = res.T

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(t, v, label = 'V')
    ax2.plot(t, n, label = 'N')
    ax2.plot(t, m, label = 'M')
    ax2.plot(t, h, label = 'H')
    ax1.legend(); ax2.legend()
    ax1.grid(); ax2.grid()

    ax1.set_ylabel('[mV]')
    ax2.set_ylabel('[]')
    ax2.set_xlabel('Time [ms]')

    ax1.set_title(fr'$I_0$ = {i_0}$\mu$A')

    fig.tight_layout()
    plt.show()
    ##################################################

####################################################################################

# CONSTANTS
####################################################################################
a_n = lambda v: .01 * (v+55)/(1 - np.exp(-(v+55)/10))
b_n = lambda v: .125 * np.exp(-(v+65)/80)
a_m = lambda v: .1 * (v + 40)/(1 - np.exp(-(v+40)/10))
b_m = lambda v: 4 * np.exp(-(v+65)/18)
a_h = lambda v: .07 * np.exp(-(v+65)/20)
b_h = lambda v: 1 / (1 + np.exp(-(v+35)/10))


g_Na, g_K, g_L = 120, 36, .3
v_Na, v_K, v_L = 50, -77, -54.4
C = 1
v_0 = -65
n_0 = m_0 = h_0 = 0
####################################################################################

# VARIABLES
####################################################################################
i_0 = 8.9
####################################################################################


# Hodgkin-Huxley
####################################################################################
def hodgkinhuxley(t, vars):
    v, n, m, h = vars
    out_v = (i_0 - g_Na * m**3 * h * (v - v_Na) - g_K * n**4 * (v - v_K) - g_L * (v - v_L)) / C
    out_n = a_n(v) * (1 - n) - n * b_n(v)
    out_m = a_m(v) * (1 - m) - m * b_m(v)
    out_h = a_h(v) * (1 - h) - h * b_h(v)

    return np.array([out_v, out_n, out_m, out_h])
####################################################################################


if __name__ == '__main__':

    test()