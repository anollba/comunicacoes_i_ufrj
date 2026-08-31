import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# =====================================================
# CONFIGURAÇÃO
# =====================================================

include_signal = True      # True = adiciona senoide
                            # False = apenas ruído

signal_freq = 15           # Hz
signal_amplitude = 0.01

Fs = 4000                  # Hz
T = 4                      # s
N = int(Fs * T)

t = np.arange(N) / Fs

# PSD bilateral do ruído
N0_over_2 = 1e-3

# larguras de banda do ruído
bandwidths = [20, 100, 400]

# =====================================================
# SINAL
# =====================================================

s = signal_amplitude * np.sin(
    2 * np.pi * signal_freq * t
)

# =====================================================
# GERAÇÃO DOS RUÍDOS
# =====================================================

noises = []
received_signals = []
powers = []

for B in bandwidths:

    # Frequências da FFT
    f = np.fft.fftfreq(N, d=1/Fs)

    # PSD retangular
    S = np.zeros(N)
    S[np.abs(f) <= B] = N0_over_2

    # Ruído complexo gaussiano
    X = (
        np.random.randn(N)
        + 1j * np.random.randn(N)
    ) * np.sqrt(S * Fs)

    # Simetrização para gerar sinal real
    X = (X + np.conj(X[::-1])) / 2

    # Ruído no tempo
    w = np.real(np.fft.ifft(X))

    noises.append(w)

    if include_signal:
        r = s + w
    else:
        r = w

    received_signals.append(r)

    powers.append(np.mean(w**2))

# =====================================================
# ESCALA DOS GRÁFICOS
# =====================================================

if include_signal:
    ymax = 1.05 * np.max(
        np.abs(np.concatenate(received_signals))
    )
else:
    ymax = 1.05 * np.max(
        np.abs(np.concatenate(noises))
    )

# =====================================================
# PLOTS
# =====================================================

fig, axs = plt.subplots(
    3,
    2,
    figsize=(14, 10)
)

for i, (B, w, r, P) in enumerate(
        zip(bandwidths,
            noises,
            received_signals,
            powers)):

    # ------------------------------------------
    # Domínio do tempo
    # ------------------------------------------

    t_plot = t[:1200]

    if include_signal:

        axs[i,0].plot(
            t_plot,
            s[:1200],
            'k',
            linewidth=2,
            label='Sinal 15 Hz'
        )

        axs[i,0].plot(
            t_plot,
            r[:1200],
            'tab:blue',
            alpha=0.8,
            label='Sinal + ruído'
        )

    else:

        axs[i,0].plot(
            t_plot,
            w[:1200],
            label='Ruído'
        )

    axs[i,0].set_ylim([-ymax, ymax])

    axs[i,0].set_title(
        f'B = {B} Hz'
    )

    axs[i,0].set_xlabel('Tempo (s)')
    axs[i,0].set_ylabel('Amplitude')
    axs[i,0].grid(True)
    axs[i,0].legend()

    # ------------------------------------------
    # PSD
    # ------------------------------------------

    f_welch, Pxx = welch(
        r,
        fs=Fs,
        nperseg=4096,
        scaling='density'
    )

    axs[i,1].plot(
        f_welch,
        Pxx,
        linewidth=2,
        label='PSD estimada'
    )

    # PSD teórica do ruído
    axs[i,1].hlines(
        N0_over_2,
        0,
        B,
        colors='red',
        linewidth=3,
        label=r'PSD ruído ($N_0/2$)'
    )

    axs[i,1].vlines(
        B,
        0,
        N0_over_2,
        colors='red',
        linestyles='--'
    )

    # marca a frequência da senoide
    if include_signal:

        axs[i,1].axvline(
            signal_freq,
            color='green',
            linestyle=':',
            linewidth=2,
            label='Senoide 15 Hz'
        )

    axs[i,1].set_xlim(0, 500)
    axs[i,1].set_ylim(0, None)

    axs[i,1].set_title(
        f'PSD - B = {B} Hz'
    )

    axs[i,1].set_xlabel('Frequência (Hz)')
    axs[i,1].set_ylabel('PSD')
    axs[i,1].grid(True)
    axs[i,1].legend()

plt.tight_layout()
plt.show()

# =====================================================
# POTÊNCIAS
# =====================================================

print("\nPotência do ruído")

for B, P in zip(bandwidths, powers):

    P_theoretical = 2 * B * N0_over_2

    print(
        f"B = {B:4d} Hz | "
        f"Medida = {P:.6f} | "
        f"Teórica = {P_theoretical:.6f}"
    )