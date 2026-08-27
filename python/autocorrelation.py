import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Sinal triangular retângulo
# =====================================================

t = np.linspace(-2, 3, 5000)

x = np.zeros_like(t)
idx = (t >= 0) & (t <= 1)
x[idx] = 1 - t[idx]

# =====================================================
# Dois atrasos para ilustrar a autocorrelação
# =====================================================

tau1 = 0.3
tau2 = 0.8

x_tau1 = np.interp(t - tau1, t, x, left=0, right=0)
x_tau2 = np.interp(t - tau2, t, x, left=0, right=0)

# =====================================================
# Autocorrelação numérica
# =====================================================

dt = t[1] - t[0]

R = np.correlate(x, x, mode='full') * dt

lags = np.arange(-len(x)+1, len(x)) * dt

# normalização apenas para visualização
R /= np.max(R)

# =====================================================
# Gráficos
# =====================================================

fig, ax = plt.subplots(3,1, figsize=(10,10))

# -----------------------------------------------------
# atraso tau1
# -----------------------------------------------------

ax[0].plot(t, x, lw=2, label=r'$x(t)$')
ax[0].plot(t, x_tau1, lw=2,
           label=rf'$x(t-{tau1})$')

ax[0].fill_between(
    t,
    x*x_tau1,
    alpha=0.4,
    color='orange',
    label='produto ponto a ponto'
)

ax[0].set_title(
    rf'Autocorrelação para $\tau={tau1}$'
)
ax[0].legend()
ax[0].grid(True)

# -----------------------------------------------------
# atraso tau2
# -----------------------------------------------------

ax[1].plot(t, x, lw=2, label=r'$x(t)$')
ax[1].plot(t, x_tau2, lw=2,
           label=rf'$x(t-{tau2})$')

ax[1].fill_between(
    t,
    x*x_tau2,
    alpha=0.4,
    color='orange',
    label='produto ponto a ponto'
)

ax[1].set_title(
    rf'Autocorrelação para $\tau={tau2}$'
)

ax[1].legend()
ax[1].grid(True)

# -----------------------------------------------------
# autocorrelação completa
# -----------------------------------------------------

ax[2].plot(lags, R, lw=2)

ax[2].axvline(tau1,
              color='r',
              ls='--',
              label=rf'$\tau={tau1}$')

ax[2].axvline(tau2,
              color='g',
              ls='--',
              label=rf'$\tau={tau2}$')

ax[2].set_title('Função de autocorrelação normalizada')

ax[2].set_xlabel(r'$\tau$')
ax[2].grid(True)
ax[2].legend()

plt.tight_layout()
plt.show()