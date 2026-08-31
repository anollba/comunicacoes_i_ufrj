import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Parâmetros
# ==========================

A = 1.0                 # amplitude (+A ou -A)
Ts = 0.1                # duração de cada pulso (s)
Npulsos = 50            # número de pulsos

fs = 1000               # frequência de amostragem
B = 2 / Ts              # largura de banda do filtro

# ==========================
# Trem de pulsos aleatório
# ==========================

Ns = int(Ts * fs)                       # amostras por pulso
simbolos = np.random.choice([-A, A], size=Npulsos)

x = np.repeat(simbolos, Ns)

t = np.arange(len(x)) / fs

# ==========================
# Filtro passa-baixas ideal
# ==========================

N = len(x)

f = np.fft.fftfreq(N, d=1/fs)

X = np.fft.fft(x)

H = np.abs(f) <= B

Y = X * H

y = np.real(np.fft.ifft(Y))

# ==========================
# Potência do sinal filtrado
# ==========================

Ps = np.mean(y**2)

snr_db_list = [-3, 0, 3, 10]

# ==========================
# Gráficos
# ==========================

fig, axs = plt.subplots(
    len(snr_db_list) + 2,
    1,
    figsize=(12, 10),
    sharex=True
)

axs[0].plot(t, x, linewidth=1)
axs[0].set_title("Trem de pulsos aleatórios ±A")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

axs[1].plot(t, y, linewidth=1)
axs[1].set_title(f"Sinal filtrado (B = 2/Ts = {B:.1f} Hz)")
axs[1].set_ylabel("Amplitude")
axs[1].grid(True)

for k, snr_db in enumerate(snr_db_list):

    snr_linear = 10**(snr_db/10)

    Pn = Ps / snr_linear

    ruido = np.sqrt(Pn) * np.random.randn(len(y))

    r = y + ruido

    axs[k+2].plot(t, r, linewidth=0.8)
    axs[k+2].set_title(f"Sinal + ruído (SNR = {snr_db} dB)")
    axs[k+2].set_ylabel("Amplitude")
    axs[k+2].grid(True)

axs[-1].set_xlabel("Tempo (s)")

plt.tight_layout()
plt.show()