import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# PARÂMETROS
# =====================================================

T0 = 2.0          # período
tau = 1.0         # largura do pulso
tc = 0.5         # centro do pulso (fase temporal)

N = 20            # número de harmônicos

# opção:
#   "cosine"
#   "exponential"

spectrum_type = "exponential"


# =====================================================
# COEFICIENTES EXPONENCIAIS
# =====================================================

n = np.arange(-N, N + 1)

cn = (
    (tau / T0)
    * np.sinc(n * tau / T0)
    * np.exp(-1j * 2 * np.pi * n * tc / T0)
)


# =====================================================
# SINAL NO TEMPO
# =====================================================

t = np.linspace(-4, 4, 5000)

x = np.zeros_like(t)

for k in range(-10, 11):

    center = tc + k * T0

    mask = (
        (t >= center - tau / 2)
        &
        (t <= center + tau / 2)
    )

    x[mask] = 1


# =====================================================
# GRÁFICOS
# =====================================================

fig, ax = plt.subplots(3, 1, figsize=(10, 8))

# -----------------------------------------------------
# sinal
# -----------------------------------------------------

ax[0].plot(t, x, lw=2)

ax[0].set_title("Trem de pulsos")
ax[0].set_xlabel("t")
ax[0].set_ylabel("x(t)")
ax[0].grid(True)


# =====================================================
# ESPECTRO TRIGONOMÉTRICO
# =====================================================

if spectrum_type.lower() == "cosine":

    n_pos = np.arange(1, N + 1)
    f0 = 1 / T0
    freq_pos = n_pos * f0

    an = 2 * np.real(cn[N + 1:])
    bn = -2 * np.imag(cn[N + 1:])

    A = np.sqrt(an**2 + bn**2)

    phi = np.arctan2(-bn, an)

    ax[1].stem(freq_pos, A)

    ax[1].set_title(
        "Espectro Trigonométrico - Amplitude"
    )

    ax[1].set_xlabel("freq (Hz)")
    ax[1].set_ylabel("A_n")
    ax[1].grid(True)

    ax[2].stem(freq_pos, phi)

    ax[2].set_title(
        "Espectro Trigonométrico - Fase"
    )

    ax[2].set_xlabel("freq (Hz)")
    ax[2].set_ylabel("rad")
    ax[2].grid(True)


# =====================================================
# ESPECTRO EXPONENCIAL
# =====================================================

else:
    freq_pos = n / T0

    ax[1].stem(freq_pos, np.abs(cn))

    ax[1].set_title(
        "Espectro Exponencial - |A_n|"
    )

    ax[1].set_xlabel("freq (Hz)")
    ax[1].set_ylabel("|A_n|")
    ax[1].grid(True)

    ax[2].stem(freq_pos, np.angle(cn))

    ax[2].set_title(
        "Espectro Exponencial - Fase"
    )

    ax[2].set_xlabel("frequência (Hz)")
    ax[2].set_ylabel("rad")
    ax[2].grid(True)


plt.tight_layout()
plt.show()