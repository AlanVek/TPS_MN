import numpy as np
import matplotlib.pyplot as plt

def ec_onda(f, g, t_0, t_f, x_0, x_f, n, m, c):

    delta_t = (t_f - t_0)/(m-1)
    delta_x = (x_f - x_0)/(n-1)

    u = np.zeros((n, m))
    r = c * delta_t / delta_x

    if r > 1:
        print('No va a converger :(')
        return np.zeros(n), np.zeros(m), u

    t = t_0 + delta_t * np.arange(m)
    x = x_0 + delta_x * np.arange(n)

    u[1:-1, 0] = f(x[1:-1])
    aprox_f2 = f(x[2:])-2*f(x[1:-1])+f(x[:-2])
    u[1:-1, 1] = f(x[1:-1]) + g(x[1:-1]) * delta_t + aprox_f2 * r**2/2

    for j in range(2, m):
        u[1:-1, j] = 2 * (1 - r**2) * u[1:-1, j-1] + r**2 * (u[2:, j-1] + u[:-2, j-1]) - u[1:-1, j-2]

    return x, t, u

def plot_helper(X, T, Z, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, T, Z, cmap='Spectral')
    ax.set_xlabel('X')
    ax.set_ylabel('T')
    ax.set_title(title)
    ax.grid()
    fig.tight_layout()

    return fig, ax

def test():
    f = lambda x: np.sin(np.pi*x) + np.sin(2*np.pi*x)
    g = lambda x: np.zeros(x.size)

    x_0, x_f = 0, 1
    t_0, t_f = 0, 1
    c, n, m = 2, 109, 221

    x, t, u = ec_onda(f, g, t_0, t_f, x_0, x_f, n, m, c)

    newx = x.reshape(-1, 1)
    newt = t.reshape(1, -1)
    ua = np.sin(np.pi * newx).dot(np.cos(2 * np.pi * newt)) + np.sin(2 * np.pi * newx).dot(np.cos(4 * np.pi * newt))

    X, T = np.meshgrid(x, t)

    print('Error absoluto máximo:', np.abs(ua - u).max())
    fig1, ax1 = plot_helper(X, T, ua.T, title = 'Solución')
    fig2, ax2 = plot_helper(X, T, u.T, title = 'Aproximación numérica')
    fig3, ax3 = plot_helper(X, T, ua.T - u.T, title = 'Errores')

    plt.show()

if __name__ == '__main__':
    test()
