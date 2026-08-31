import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parâmetros
# ============================================================

Fs = 5000
f0 = 50
A = 1.0

T_list = [0.1, 0.5, 2.0]

Nfft = 32768

# Janela fixa para visualização
t_plot = np.arange(-1.2, 1.2, 1/Fs)

fig, ax = plt.subplots(
    len(T_list),
    2,
    figsize=(12,8)
)

# ============================================================
# Loop
# ============================================================

for k, T in enumerate(T_list):

    # --------------------------------------------------------
    # Senoide truncada
    # --------------------------------------------------------

    x = (
        A*np.cos(2*np.pi*f0*t_plot)
        * (np.abs(t_plot) <= T/2)
    )

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    X = np.fft.fftshift(
        np.fft.fft(x, Nfft)
    )

    f = np.fft.fftshift(
        np.fft.fftfreq(Nfft, 1/Fs)
    )

    # aproximação da PSD
    PSD_approx = np.abs(X)**2 / T

    # --------------------------------------------------------
    # Tempo
    # --------------------------------------------------------

    ax[k,0].plot(
        t_plot,
        x,
        linewidth=2
    )

    ax[k,0].set_xlim(-1.2,1.2)
    ax[k,0].set_ylim(-1.1,1.1)

    ax[k,0].set_title(
        f'T = {T} s'
    )

    ax[k,0].set_xlabel('Tempo (s)')
    ax[k,0].grid(True)

    # --------------------------------------------------------
    # Espectro
    # --------------------------------------------------------

    ax[k,1].plot(
        f,
        PSD_approx,
        linewidth=2
    )

    ax[k,1].axvline(
        f0,
        color='red',
        ls='--'
    )

    ax[k,1].axvline(
        -f0,
        color='red',
        ls='--'
    )

    ax[k,1].set_xlim(-100,100)
    ax[k,1].set_ylim(0,1e7)


    ax[k,1].set_title(
        r'$|X_T(f)|^2/T$'
    )

    ax[k,1].set_xlabel(
        'Frequência (Hz)'
    )

    ax[k,1].grid(True)

plt.tight_layout()
plt.show()