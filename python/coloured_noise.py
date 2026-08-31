import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# =====================================================
# Parâmetros
# =====================================================

Fs = 1000
T = 20
N = Fs * T

np.random.seed(0)

# =====================================================
# Geração de ruído colorido
# =====================================================

def colored_noise(alpha, N, Fs):

    f = np.fft.rfftfreq(N, 1/Fs)

    X = (
        np.random.randn(len(f))
        + 1j*np.random.randn(len(f))
    )

    H = np.ones_like(f)

    H[1:] = 1 / (f[1:] ** (alpha/2))

    Y = X * H

    x = np.fft.irfft(Y, n=N)

    return x

# =====================================================
# Normalização de potência
# =====================================================

def normalize_power(x, power=1):

    return x * np.sqrt(power / np.var(x))

# =====================================================
# Ruídos
# =====================================================

white = normalize_power(colored_noise(0, N, Fs))
pink  = normalize_power(colored_noise(1, N, Fs))
brown = normalize_power(colored_noise(2, N, Fs))

# =====================================================
# PSD estimadas
# =====================================================

f, Pwhite = welch(
    white,
    fs=Fs,
    nperseg=8192,
    scaling='density'
)

_, Ppink = welch(
    pink,
    fs=Fs,
    nperseg=8192,
    scaling='density'
)

_, Pbrown = welch(
    brown,
    fs=Fs,
    nperseg=8192,
    scaling='density'
)

# Evita divisão por zero
f2 = f.copy()
f2[0] = f2[1]

# =====================================================
# PSDs ideais
# =====================================================

# Ajuste visual usando o valor médio em ~10 Hz

idx = np.argmin(np.abs(f - 10))

Kwhite = Pwhite[idx]
Kpink  = Ppink[idx]  * 10
Kbrown = Pbrown[idx] * 10**2

Sideal_white = Kwhite * np.ones_like(f2)
Sideal_pink  = Kpink / f2
Sideal_brown = Kbrown / (f2**2)

# =====================================================
# Figura
# =====================================================

fig, ax = plt.subplots(
    2,
    1,
    figsize=(12, 8)
)

# -----------------------------------------------------
# Tempo
# -----------------------------------------------------

samples = 3000

ax[0].plot(
    np.arange(samples)/Fs,
    white[:samples],
    label='Branco'
)

ax[0].plot(
    np.arange(samples)/Fs,
    pink[:samples],
    label='Rosa'
)

ax[0].plot(
    np.arange(samples)/Fs,
    brown[:samples],
    label='Browniano'
)

ax[0].set_title(
    'Ruídos com a mesma potência total'
)

ax[0].set_xlabel('Tempo (s)')
ax[0].set_ylabel('Amplitude')

ax[0].grid(True)
ax[0].legend()

# -----------------------------------------------------
# Frequência
# -----------------------------------------------------

ax[1].loglog(
    f[1:],
    Pwhite[1:],
    linewidth=2,
    label='Branco (estimada)'
)

ax[1].loglog(
    f[1:],
    Sideal_white[1:],
    '--',
    linewidth=2,
    label='Branco ideal'
)

ax[1].loglog(
    f[1:],
    Ppink[1:],
    linewidth=2,
    label='Rosa (estimada)'
)

ax[1].loglog(
    f[1:],
    Sideal_pink[1:],
    '--',
    linewidth=2,
    label='Rosa ideal (1/f)'
)

ax[1].loglog(
    f[1:],
    Pbrown[1:],
    linewidth=2,
    label='Browniano (estimada)'
)

ax[1].loglog(
    f[1:],
    Sideal_brown[1:],
    '--',
    linewidth=2,
    label='Browniano ideal (1/f²)'
)

ax[1].set_title(
    'PSD estimada versus PSD ideal'
)

ax[1].set_xlabel('Frequência (Hz)')
ax[1].set_ylabel('PSD')

ax[1].grid(True, which='both')
ax[1].legend()

plt.tight_layout()
plt.show()

# =====================================================
# Potências
# =====================================================

print("Potências:")
print(f"Branco     : {np.var(white):.4f}")
print(f"Rosa       : {np.var(pink):.4f}")
print(f"Browniano  : {np.var(brown):.4f}")