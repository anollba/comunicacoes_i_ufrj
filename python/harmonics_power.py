import numpy as np
import matplotlib.pyplot as plt

Nmax = 20

n = np.arange(1, Nmax + 1)

a0 = 0.5
an = (2/(n*np.pi))*np.sin(n*np.pi/2)

P = []

for N in range(0, Nmax + 1):
    PN = a0**2 + 0.5*np.sum(an[:N]**2)
    P.append(PN)

plt.figure(figsize=(8,4))

plt.plot(range(0, Nmax+1), P, lw=2,
         label='Potência acumulada')

plt.axhline(
    y=0.5,
    color='r',
    linestyle='--',
    label='Potência total = 0.5'
)

plt.xlabel('Número de harmônicos')
plt.ylabel('Potência')
plt.title('Convergência da potência (T = T0/2)')
plt.grid(True)
plt.legend()
plt.ylim(bottom=0)
plt.xlim(0, Nmax)