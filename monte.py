import matplotlib.pyplot as plt
import numpy as np
import pylab as pyl
from scipy import special
plt.close("all")

def modulate_BSPK(bits):
    return np.reshape(bits * 2 - 1, (-1, 1))  # reshape bits in [0,1] to [-1,1]

def detecting_BPSK(received_signal):
    return (np.real(received_signal) > 0).astype(int)  # Proper slicing for array

if __name__ == '__main__':
    # Set simulation parameters
    PSK_order = 2  # BPSK
    number_of_bits = int(np.log2(PSK_order))  # Number of bits per symbol
    number_of_realizations = 10000  # Number of packets for simulation
    points = 31  
    SNR = np.linspace(-10, 20, points)  # SNR in dB

    # Initialize BER storage
    BER = np.zeros((number_of_realizations, len(SNR)))

    print("Simulation started....")
    
    # Simulation loop over SNR and random realizations
    for SNR_index, current_SNR in enumerate(SNR):
        for realization in range(number_of_realizations):
            # Generate random bits
            b = np.random.randint(0, 2, int(number_of_bits))
            x = modulate_BSPK(b)  # Map bits to symbols

            # Add noise
            noisePower = 10**(-current_SNR / 20)  # Calculate noise power
            noise = noisePower * (1/np.sqrt(2)) * (np.random.randn(len(x)) + 1j * np.random.randn(len(x)))  # Generate noise

            y_AWGN = x + noise  # Add noise to the signal

            # Detect received bits
            b_received = detecting_BPSK(y_AWGN)

            # Calculate bit errors
            BER[realization, SNR_index] = np.sum(np.abs(b - b_received)) / number_of_bits

        print(f"{100*SNR_index/len(SNR):.2f}% completed")  

    print("Simulation finished")

    # Calculate mean BER over realizations
    mean_BER = np.mean(BER, axis=0)

    # Calculate theoretical BER for BPSK
    SNR_lin = 10**(SNR / 10)
    mean_BER_theoretical = 0.5 * special.erfc(np.sqrt(SNR_lin))

    # Plot BER results
    plt.figure()
    plt.semilogy(SNR, mean_BER, marker='.', label='Simulated')
    plt.semilogy(SNR, mean_BER_theoretical, marker='*', label='Theoretical')
    plt.grid(True)
    plt.axis([-10, 20, 1e-3, 1])
    plt.ylabel('BER')
    plt.xlabel('SNR (dB)')
    plt.title('BER vs SNR for BPSK with AWGN')
    plt.legend()
    plt.show()
