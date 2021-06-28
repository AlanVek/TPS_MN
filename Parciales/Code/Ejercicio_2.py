import numpy as np
import matplotlib.pyplot as plt

legajo = 59378

# Ejercicio 2
###########################

from scipy.special import factorial

def taylor3(dx, ddx, dddx, x_0, t_0, t_f, N):
    delta_t = (t_f - t_0) / (N - 1)

    x_k = np.zeros((N, len(x_0)))
    x_k[0] = x_0
    t_k = t_0 + delta_t * np.arange(N)

    for i in range(N - 1):
        derivs = np.asarray([[dx(t_k[i], x_k[i]), ddx(t_k[i], x_k[i]), dddx(t_k[i], x_k[i])]]).T.reshape(-1, 3)
        expos = np.arange(1, 4)
        x_k[i + 1] = x_k[i] + derivs.dot(delta_t ** expos / factorial(expos))

    return t_k, x_k

# Test
########################################################
def dx(t, x): return x[0] + np.sin(t)
def ddx(t, x): return dx(t, x) + np.cos(t)
def dddx(t, x): return ddx(t, x) - np.sin(t)

x_0 = [-1/2]
t_0 = 0
t_f = 2
N = 100

t, x = taylor3(dx, ddx, dddx, x_0, t_0, t_f, N)

K = (x_0[0] + np.cos(t_0)/2 + np.sin(t_0)/2) * np.exp(-t_0)
real = K * np.exp(t) - np.sin(t) / 2 - np.cos(t) / 2

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(t, x.reshape(-1), label = 'Taylor 3')
ax1.plot(t, real, label = 'Real')
ax2.plot(t, real - x.reshape(-1), label = 'Error')

ax1.grid(); ax2.grid()
ax1.legend(); ax2.legend()
ax2.set_xlabel('Time [s]')
ax1.set_title('Resultados Test')
fig.tight_layout()
plt.show()
###############################################################

n = 3
maxerr = 1e-6

def dx(t, _x):
    k, x = _x
    return [-x + np.exp(-t), k]

def ddx(t, _x):
    k, x = _x
    return [-k - np.exp(-t), -x + np.exp(-t)]

def dddx(t, _x):
    k, x = _x
    return [x, -k - np.exp(-t)]

t_0, t_f = -5, 5
x_0 = [1, legajo / 1000]

N_from_delta = lambda delta, t_0, t_f: int(np.round((t_f - t_0)/delta + 1))

delta_t1 = .1

_, x1 = taylor3(dx, ddx, dddx, x_0, t_0, t_f, N_from_delta(delta_t1, t_0, t_f))
_, x2 = taylor3(dx, ddx, dddx, x_0, t_0, t_f, N_from_delta(delta_t1 / 2, t_0, t_f))

c = np.max(np.abs(x1[:, 1] - x2[::2, 1]) / (delta_t1**n * (1 - 1/2**n)))

delta_t = .85 * (maxerr / c)**(1/n)
print('Delta_t:', delta_t.round(5))

t, _x = taylor3(dx, ddx, dddx, x_0, t_0, t_f, N_from_delta(delta_t, t_0, t_f))
k, x = _x.T

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(t, x.reshape(-1), label = 'y')
ax2.plot(t, k.reshape(-1), label = r'$\frac{dy}{dt}$')

ax1.grid(); ax2.grid()
ax1.legend(); ax2.legend()
ax2.set_xlabel('Time [s]')
fig.tight_layout()
plt.show()

_, _x2 = taylor3(dx, ddx, dddx, x_0, t_0, t_f, N_from_delta(delta_t/2, t_0, t_f))

k2, x2 = _x2.T

real_err = np.max(np.abs(x2[:x.size*2:2] - x) / (1 - 1/2**n))
derv_err = np.max(np.abs(k2[:x.size*2:2] - k) / (1 - 1/2**n))

print('x error:', real_err)
print('Deriv error:', derv_err)
