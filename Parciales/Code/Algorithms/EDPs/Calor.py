import numpy as np
import matplotlib.pyplot as plt

def ec_calor(f, g1, g2, t_0, t_f, x_0, x_f, n, m, c):

    delta_t = (t_f - t_0)/(m-1)
    delta_x = (x_f - x_0)/(n-1)

    r = (c**2 * delta_t) / delta_x**2
    u = np.zeros((n, m))

    if not r <= .5:
        print('No va a converger :(')
        return np.zeros(n), np.zeros(m), u

    t = t_0 + delta_t * np.arange(m)
    x = x_0 + delta_x * np.arange(n)

    u[[0, -1]] = [g1(t), g2(t)]
    u[1:-1, 0] = f(x[1:-1])

    for j in range(m-1):
        u[1:-1, j + 1] = (1 - 2 * r) * u[1:-1, j] + r * (u[2:, j] + u[:-2, j])

    return x, t, u

if __name__ == '__main__':
    t_0, t_f, x_0, x_f = 0, 1, 0, 1
    n, m = 128, 512
    c = .1

    f = lambda x: np.sin(2 * np.pi * x * 2)
    g1 = lambda t: np.sin(2 * np.pi * t)
    g2 = lambda t: -g1(t)

    x, t, u = ec_calor(f, g1, g2, t_0, t_f, x_0, x_f, n, m, c)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    X, T = np.meshgrid(x, t)
    ax.plot_surface(X, T, u.T, cmap = 'plasma', linewidth = 2)
    ax.set_xlabel('X')
    ax.set_ylabel('T')

    fig.tight_layout()
    plt.show()