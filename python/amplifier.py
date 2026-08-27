import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Modelo de Rapp
# ============================================================

def rapp_amplifier(x, A_sat=1.0, p=3.0):

    r = np.abs(x)

    gain = 1.0 / (
        (1 + (r / A_sat)**(2*p))**(1/(2*p))
    )

    return x * gain


# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Escolher:
# 'tone'      -> uma senoide
# 'two_tone'  -> duas senoides
# 'tri'       -> pulso triangular

signal_type = 'tri'

# Escala do espectro:
# 'linear'
# 'dB'

spectrum_scale = 'dB'

# ============================================================
# Parâmetros gerais
# ============================================================

Fs = 10000
N = 16384

A_sat = 1.0

# Parâmetro de suavidade do modelo de Rapp
p = 3.0

# Input Back-Off (dB)
IBO_dB = -9

t = np.arange(N) / Fs

Ain = A_sat * 10**(-IBO_dB/20)

# ============================================================
# Geração do sinal
# ============================================================

if signal_type == 'tone':

    f0 = 500

    x = Ain * np.cos(
        2*np.pi*f0*t
    )

    signal_name = f'Senoide ({f0} Hz)'

elif signal_type == 'two_tone':

    f1 = 60
    f2 = 100

    x = Ain * (
        np.cos(2*np.pi*f1*t)
        +
        np.cos(2*np.pi*f2*t)
    ) / 2

    signal_name = (
        f'Dois tons ({f1} Hz e {f2} Hz)'
    )

elif signal_type == 'tri':

    pulse_width = 0.10

    u = (
        t - pulse_width/2
    ) / (pulse_width/2)

    x = Ain * np.maximum(
        1 - np.abs(u),
        0
    )

    signal_name = (
        f'Pulso triangular '
        f'(T={pulse_width*1000:.0f} ms)'
    )

else:

    raise ValueError(
        "signal_type deve ser "
        "'tone', 'two_tone' ou 'tri'"
    )

# ============================================================
# Amplificador
# ============================================================

y = rapp_amplifier(
    x,
    A_sat=A_sat,
    p=p
)

# ============================================================
# Curva AM/AM
# ============================================================

Ain_curve = np.linspace(
    0,
    2*A_sat,
    2000
)

Aout_curve = Ain_curve / (
    (1 + (Ain_curve/A_sat)**(2*p))**(1/(2*p))
)

# ============================================================
# FFT
# ============================================================

f = np.fft.fftshift(
    np.fft.fftfreq(
        N,
        d=1/Fs
    )
)

X = np.fft.fftshift(
    np.fft.fft(x)
)

Y = np.fft.fftshift(
    np.fft.fft(y)
)

# Magnitude linear normalizada

Xlin = np.abs(X)
Ylin = np.abs(Y)

Xlin = Xlin / np.max(Xlin)
Ylin = Ylin / np.max(Ylin)

# Magnitude em dB

XdB = 20*np.log10(
    Xlin + 1e-12
)

YdB = 20*np.log10(
    Ylin + 1e-12
)

# ============================================================
# Figura
# ============================================================

fig = plt.figure(
    figsize=(12,10)
)

# ------------------------------------------------------------
# Curva AM/AM
# ------------------------------------------------------------

ax1 = plt.subplot(311)

ax1.plot(
    Ain_curve,
    Aout_curve,
    linewidth=2,
    label='Modelo de Rapp'
)

ax1.plot(
    [0, 2*A_sat],
    [0, 2*A_sat],
    '--',
    color='gray',
    label='Amplificador linear'
)

ax1.set_title(
    f'Curva AM/AM (A_sat={A_sat}, p={p})'
)

ax1.set_xlabel(
    'Amplitude de entrada'
)

ax1.set_ylabel(
    'Amplitude de saída'
)

ax1.grid(True)
ax1.legend()

# ------------------------------------------------------------
# Domínio do tempo
# ------------------------------------------------------------

ax2 = plt.subplot(312)

if signal_type == 'tone':

    samples = 250

elif signal_type == 'two_tone':

    samples = 1000

else:

    samples = int(0.15*Fs)

ax2.plot(
    t[:samples],
    x[:samples],
    linewidth=2,
    label='Entrada'
)

ax2.plot(
    t[:samples],
    y[:samples],
    linewidth=2,
    label='Saída'
)

ax2.set_title(
    f'{signal_name}   |   IBO={IBO_dB} dB'
)

ax2.set_xlabel(
    'Tempo (s)'
)

ax2.set_ylabel(
    'Amplitude'
)

ax2.grid(True)
ax2.legend()

# ------------------------------------------------------------
# Espectro
# ------------------------------------------------------------

ax3 = plt.subplot(313)

if spectrum_scale == 'linear':

    ax3.plot(
        f,
        Xlin,
        linewidth=2,
        label='Entrada'
    )

    ax3.plot(
        f,
        Ylin,
        linewidth=2,
        label='Saída'
    )

    ax3.set_ylabel(
        'Magnitude normalizada'
    )

else:

    ax3.plot(
        f,
        XdB,
        linewidth=2,
        label='Entrada'
    )

    ax3.plot(
        f,
        YdB,
        linewidth=2,
        label='Saída'
    )

    ax3.set_ylabel(
        'Magnitude (dB)'
    )

    ax3.set_ylim(
        -100,
        5
    )

if signal_type == 'two_tone':

    ax3.set_xlim(
        -400,
        400
    )

elif signal_type == 'tone':

    ax3.set_xlim(
        -3000,
        3000
    )

else:

    ax3.set_xlim(
        -200,
        200
    )

ax3.set_title(
    'Espectro'
)

ax3.set_xlabel(
    'Frequência (Hz)'
)

ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()