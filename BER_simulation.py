## BER vs SNR for Optical OOK Modulation
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

# snr_db = 9.80
# snr = 10**(snr_db/10)
# ber = 0.5 * sc.special.erfc(np.sqrt(snr/2))
# print(ber)

# BER 0.001 -> SNR 
# BEr 0.000001 -> SNR 

ber_values = [0.001, 0.000001]

# Compute the required SNR values for the given BERs
snr_values_db = [10 * np.log10(2 * (sc.special.erfcinv(2 * ber))**2) for ber in ber_values]

print(snr_values_db)