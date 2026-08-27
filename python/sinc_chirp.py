import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parâmetros
# ============================================================

Fs = 2000
N = 16384

B = 100          # largura de banda
alpha = 0.02     # controla a curvatura da fase

# ============================================================
# Eixo de frequência
# ============================================================

f = np.fft.fftshift(
    np.fft.fftfreq(N, 1/Fs)
)

# ============================================================
# Espectro 1:
# Magnitude retangular + fase nula
# ============================================================

X1 = np.zeros(N, dtype=complex)

mask = np.abs(f) <= B/2

X1[mask] = 1.0

# ============================================================
# Espectro 2:
# Magnitude retangular + fase quadrática
# ============================================================

X2 = np.zeros(N, dtype=complex)

X2[mask] = np.exp(
    -1j * alpha * f[mask]**2 * np.sign( f[mask])
)

# ============================================================
# Sinais no tempo
# ============================================================

x1 = np.fft.ifft(
    np.fft.ifftshift(X1)
)
x1 = np.fft.ifftshift(x1)

x2 = np.fft.ifft(
    np.fft.ifftshift(X2)
)
x2 = np.fft.ifftshift(x2)

t = np.arange(N)/Fs
t = t - np.mean(t)

# ============================================================
# Gráficos
# ============================================================

fig, ax = plt.subplots(
    3,
    1,
    figsize=(10,8)
)

# ------------------------------------------------------------
# Magnitude espectral
# ------------------------------------------------------------

ax[0].plot(
    f,
    np.abs(X1),
    label='Magnitude'
)

ax[0].set_xlim(-100,100)
ax[0].set_title(
    'Magnitude espectral (idêntica nos dois casos)'
)
ax[0].grid(True)

# ------------------------------------------------------------
# Fase
# ------------------------------------------------------------

ax[1].plot(
    f,
    np.angle(X1),
    label='Fase nula'
)

ax[1].plot(
    f,
    np.angle(X2),
    label='Fase quadrática'
)

ax[1].set_xlim(-100,100)

ax[1].set_title(
    'Fase espectral'
)

ax[1].grid(True)
ax[1].legend()

# ------------------------------------------------------------
# Tempo
# ------------------------------------------------------------

ax[2].plot(
    t,
    np.real(x1),
    label='sinc'
)

ax[2].plot(
    t,
    np.real(x2),
    label='chirp'
)

ax[2].set_xlim(-0.5,0.5)

ax[2].set_title(
    'Sinais no tempo'
)

ax[2].grid(True)
ax[2].legend()

plt.tight_layout()
plt.show()