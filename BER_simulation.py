import matplotlib.pyplot as plt
import numpy as np
import math


snr = np.linspace(0, 10, 30000)

# ber = 0.5 * math.erfc(0.5 * snr)

ber = [0.5 * math.erfc(0.5 * i) for i in snr]

a = plt.figure()
plt.plot(snr, ber)
# plt.plot(snr,10e-6)
plt.plot(snr, [10e-3 for i in len(snr)])
plt.yscale('log')

plt.show()

