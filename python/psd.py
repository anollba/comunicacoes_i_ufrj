import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PSD de exemplo
# ============================================================

f = np.linspace(-100, 100, 5000)

# PSD gaussiana apenas para ilustração
S = np.exp(-(f/30)**2)

# ============================================================
# Banda de interesse
# ============================================================

B=30

mask = np.abs(f) <= B

# potência na banda
P_band = np.trapezoid(S[mask], f[mask])

# ============================================================
# Gráfico
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(
    f,
    S,
    linewidth=2,
    color='navy',
    label=r'$S_x(f)$'
)

plt.fill_between(
    f[mask],
    S[mask],
    color='orange',
    alpha=0.6,
    label='Potência na banda'
)

plt.axvline(
    -B,
    linestyle='--',
    color='red'
)

plt.axvline(
    B,
    linestyle='--',
    color='red'
)

plt.text(
    5,
    0.5,
    r'$P_B=\int_{-B}^{B}S_x(f)\,df$',
    fontsize=14
)

plt.xlabel('Frequência (Hz)')
plt.ylabel(r'$S_x(f)$ (W/Hz)')

plt.title(
    'Densidade Espectral de Potência'
)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Potência na banda = {P_band:.2f}")