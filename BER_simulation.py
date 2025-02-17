
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

snr_db = 16.55 
snr = 10**(snr_db/10)
ber = 0.5 * sc.special.erfc(np.sqrt(snr)/2)
print(ber)

# BER 0.001 -> SNR 12.81 dB
# BEr 0.000001 -> SNR 16.55 dB