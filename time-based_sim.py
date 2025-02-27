import numpy as np
import matplotlib.pyplot as plt



#####=- Functions -=#####
# Convert dB to linear scale
def db_2_lin(val):    
    lin_val = 10**(val / 10)
    return lin_val

# Generate PRBS signal
def gen_prbs(n_bits):
    seed_size = 5
    taps = [5, 2]  # Feedback taps positions (e.g., [5, 2] for x^5 + x^2 + 1)
    seed = np.random.choice([0, 1], size=(seed_size))

    state = np.array(seed, dtype=int)
    taps = np.array(taps) - 1  # Convert to zero-based indexing
    signal = []

    for _ in range(n_bits):
        new_bit = np.bitwise_xor.reduce(state[taps])  # XOR the tapped bits
        signal.append(state[-1])  # Output the last bit
        state = np.roll(state, -1)  # Shift left
        state[-1] = new_bit  # Insert new bit
    
    return signal
        
# Calculate time-variant loss: jitter-induced scintillation
def calc_jit_loss(p_0, sigma_pj, theta_div, size):
    loss_jit = 10 * np.log(p_0 ** ((4 * sigma_pj ** 2)/(theta_div ** 2)))
    losses_jit = np.random.normal(loss_jit, sigma_pj, size)
    
    return losses_jit

# Calculate time-variant loss: scintillation (turbulence)
def calc_scint_loss(p_0, sigma_i_sq, size):
    loss_scint = (3.3 - 5.77 * np.sqrt(-np.log(p_0))) * sigma_i_sq ** (2/5)
    losses_scint = np.random.normal(loss_scint, np.sqrt(sigma_i_sq), size)
    
    return losses_scint

# Generate AWGN noise for given SNR
def gen_awgn(signal, snr_db):
    signal_power = np.mean(signal**2)  # Compute signal power
    snr_linear = db_2_lin(snr_db)  # Convert SNR from dB to linear scale
    noise_power = signal_power / snr_linear  # Compute noise power
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)  # Generate Gaussian noise
    
    return noise



#####=- Inputs -=#####
random = False  # Switch: if False, use seed 0
R_f = 10  # Frequency ratio: ratio between transmitter frequency and random frequency (visual) 

# PRBS
bitrate = 50  # Transmitted bits per second [-]
t_end = 1  # Signal duration [s]

# Transmitter
P_l = 0.08  # Transmitter laser power [W]
lam = 1550 * 10**(-9)  # Laser wavelength [m]
theta_div = 10 * 10**(-6)  # Laser divergence [rad]
sigma_pj = 2 * 10**(-6)  # Jitter RMS [rad]

# Environment 
z = 50  # Optical path length [m]
n = 1   # Air refractive index [-]
sigma_i_sq = 0.2  # Scintillation index [-]

# Losses
L_c = -5  # Constant loss: all link budget losses except for (jitter-induced) scintillation [dB]

# Receiver 
p_0 = 0.1  # Outage probability [-]
snr = 5  # Signal-to-noise ratio [dB]



#####=- Calculations -=#####
if random == True:
    np.random.seed(0)  

# Generate PRBS transmitter signal  
n_bits = bitrate * t_end
tx_bits = gen_prbs(n_bits)  # PRBS generator
tx_signal = np.multiply(np.repeat(tx_bits, R_f), P_l)  # Transmitted signal
t = np.linspace(0, t_end, len(tx_signal))  # Time steps

# Attenuate signal: include losses
L_pj = calc_jit_loss(p_0, sigma_pj, theta_div, len(tx_signal))  # Pointing jitter loss [dB]
L_sc = calc_scint_loss(p_0, sigma_i_sq, len(tx_signal))  # Scintillation loss [dB]
L_tot = db_2_lin(L_c + L_pj + L_sc)  # Total loss [-]
tx_signal_loss =  L_tot * tx_signal

# Add Gaussian noise (AWGN)
awgn = gen_awgn(tx_signal_loss, snr)
rx_signal = (tx_signal_loss + awgn)

# Apply on-off keying
threshold = np.mean(rx_signal[::R_f])
rx_bits = (rx_signal[::R_f] > threshold).astype(int)
bit_errors = np.sum(tx_bits != rx_bits)
BER = bit_errors / n_bits
print("BER: " + str(BER))



#####=- Plotter -=#####
# Create figure for plots
plt.figure(figsize=(12, 9))

# Plot 1: Received signals
plt.subplot(3, 1, 1)
plt.step(t, tx_signal_loss, where='post', label="Attenuated signal", linewidth=2, alpha=0.7)
plt.step(t, rx_signal, where='post', label="Noisy signal: SNR = "+str(snr)+" dB", linewidth=1, alpha=0.7)
#plt.plot(t, rx_signal, label="Noisy signal", linewidth=1, alpha=0.7)
plt.scatter(t[::R_f], rx_signal[::R_f], label="Receiver sampling", s=15)
plt.step(t, np.repeat(rx_signal[::R_f], R_f), where='post', label="Received signal", linewidth=2, alpha=0.7)
plt.axhline(threshold, color='r', linestyle='dashed', label="Decision Threshold = "+str(round(threshold,4)))
plt.xlabel("Time [s]")
plt.ylabel("Power [W]")
plt.title("Attenuated, noisy and received signals")
plt.grid(True)
plt.legend()

# Plot 2: Transmitted and received binary signals
plt.subplot(3, 1, 2)
plt.step(t, np.repeat(tx_bits, R_f), where='post', label="Transmitted binary signal", linewidth=3, alpha=0.7)
plt.step(t, np.repeat(rx_bits, R_f), where='post', label="Received binary signal", linewidth=3, alpha=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Transmitted and received binary signals: bitrate = "+str(bitrate)+str(" bps")+", BER = "+str(BER))
plt.grid(True)
plt.legend()

# Plot 3: Histogram of received signal
plt.subplot(3, 1, 3)
plt.hist(rx_signal[::R_f], bins=10, density=True, alpha=0.6, color='b', edgecolor='black')
plt.axvline(threshold, color='r', linestyle='dashed', label="Decision Threshold = "+str(round(threshold,4)))
plt.xlabel("Power [W]")
plt.ylabel("Probability density [-]")
plt.title("Histogram of received power")
plt.legend()
plt.grid(True)

# Show all plots
plt.tight_layout()
plt.show()
