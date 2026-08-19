import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PARÂMETROS DO SINAL
# =====================================================

T0 = 2.0      # período
T = 1.0       # largura do pulso
tau = 0.0     # centro do pulso

# =====================================================
# HARMÔNICOS A INCLUIR
# =====================================================

n_ini = 3
n_fim = 100

# =====================================================
# FUNDAMENTAL
# =====================================================

f0 = 1 / T0
w0 = 2 * np.pi * f0

# =====================================================
# EIXO TEMPORAL
# =====================================================

t = np.linspace(-2*T0, 2*T0, 5000)

# =====================================================
# SINAL ORIGINAL
# =====================================================

x = np.zeros_like(t)

for k in range(-20, 21):

    center = tau + k*T0

    mask = (
        (t >= center - T/2)
        &
        (t <= center + T/2)
    )

    x[mask] = 1

# =====================================================
# RECONSTRUÇÃO PARCIAL
# =====================================================

x_rec = np.zeros_like(t)

# componente DC
a0 = T / T0

if n_ini == 0:
    x_rec += a0

# demais harmônicos

for n in range(max(1, n_ini), n_fim + 1):

    an = (
        (2/(n*np.pi))
        * np.sin(n*np.pi*T/T0)
        * np.cos(2*np.pi*n*tau/T0)
    )

    bn = (
        (2/(n*np.pi))
        * np.sin(n*np.pi*T/T0)
        * np.sin(2*np.pi*n*tau/T0)
    )

    x_rec += (
        an*np.cos(n*w0*t)
        +
        bn*np.sin(n*w0*t)
    )

# =====================================================
# POTÊNCIA RECONSTRUÍDA
# =====================================================

P = np.mean(x_rec**2)

# =====================================================
# GRÁFICOS
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(
    t,
    x,
    'k',
    linewidth=2,
    label='Sinal original'
)

plt.plot(
    t,
    x_rec,
    'r',
    linewidth=2,
    label=f'Harmônicos {n_ini} a {n_fim}'
)

plt.grid(True)

plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')

plt.title(
    f'Reconstrução usando harmônicos {n_ini}-{n_fim}'
)

plt.legend()

plt.tight_layout()

plt.show()

print(f"Potência da reconstrução = {P:.6f}")