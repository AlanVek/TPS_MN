from Ej1.ruku4 import *
from Ej2.Hodgkin_Huxley import gen_hh
import matplotlib.pyplot as plt


def plot_hh(v_0, n_0 = 0, m_0 = 0, h_0 = 0, i_0 = 0, delta_t = .1, t_f = 200):

    i = lambda t: i_0
    t_0 = 0
    x_0 = [v_0, n_0, m_0, h_0]

    HH = gen_hh(i)

    t, res = ruku4(HH, t_0, t_f, delta_t, x_0)

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