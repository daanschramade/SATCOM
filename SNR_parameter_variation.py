# Re-import necessary libraries after execution state reset
import numpy as np
import matplotlib.pyplot as plt

# Reimplement the updated OpticalLinkBudget class with compute_link_budget method using user's formulas
class OpticalLinkBudgetUpdated:
    def __init__(self, Tx_power, T_atmos, theta_div, sigma_pj, optics_array):
        self.Tx_power = Tx_power  # Laser power in W
        self.T_atmos = T_atmos  # Atmospheric transmission factor
        self.theta_div = theta_div  # Divergence angle
        self.sigma_pj = sigma_pj  # Pointing jitter
        self.optics_array = optics_array  # Optical efficiency

    def free_space_loss(self, lambd, L):
        """Compute free-space loss using user's formula"""
        return 20 * np.log10((4 * np.pi * L) / lambd)

    def tx_gain(self):
        """Compute transmitter gain"""
        return 10 * np.log10(8 / (self.theta_div ** 2))

    def rx_gain(self, Dr, wave):
        """Compute receiver gain"""
        return 10 * np.log10(((np.pi * Dr) / wave) ** 2)

    def total_optics_loss(self):
        """Compute total optical loss"""
        optics_loss = np.prod(self.optics_array)
        return 10 * np.log10(optics_loss)

    def compute_link_budget(self, lambd, L):
        """Compute full optical link budget using user's approach"""

        # Calculate individual gains and losses
        Gtx = self.tx_gain()
        optics_loss = self.total_optics_loss()
        Lfs = self.free_space_loss(lambd, L)
        atmos_loss = 10 * np.log10(self.T_atmos)
        Grx = self.rx_gain(1, lambd)  # Assuming Dr=1 for receiver gain

        # Compute total losses
        total_losses = Gtx + optics_loss + Lfs + atmos_loss + Grx

        # Compute received power (in dB and Watts)
        P_rx_db = 10 * np.log10(self.Tx_power) - total_losses
        P_rx = 10 ** (P_rx_db / 10)

        return {
            "L": L,
            "Wavelength": lambd,
            "Lfs": Lfs,
            "Total Losses": total_losses,
            "P_rx (dB)": P_rx_db,
            "P_rx (W)": P_rx
        }

# Reimplement the SNR Calculator class
class SNRCalculatorUpdated:
    def __init__(self, optical_link):
        self.optical_link = optical_link
        self.k_B = 1.38e-23  # Boltzmann constant (J/K)
        self.T = 300  # Temperature in Kelvin
        self.R_L = 50  # Load resistance in ohms
        self.q = 1.6e-19  # Electron charge (C)
        self.B = 10e6  # Bandwidth (10 MHz)

    def compute_snr(self, wavelength_values, L_values):
        snr_results = {}
        
        for L in L_values:
            snr_values = []
            
            for lambd in wavelength_values:
                link_budget = self.optical_link.compute_link_budget(lambd, L)
                P_rx = link_budget["P_rx (W)"]
                
                # Compute noise power (σ²)
                sigma2_thermal = self.k_B * self.T / self.R_L  # Thermal noise
                I_d = P_rx / (self.q * 0.6)  # Assume quantum efficiency of 0.6
                sigma2_shot = 2 * self.q * I_d * self.B  # Shot noise
                sigma2 = sigma2_thermal + sigma2_shot  # Total noise power
                
                # Compute SNR
                snr = np.maximum(P_rx / sigma2, 1e-12)  # Prevent divide by zero errors
                snr_values.append(snr)
            
            snr_results[L] = snr_values
        return snr_results

    def plot_snr(self, wavelength_values, snr_results):
        plt.figure(figsize=(8, 6))
        for L, snr_values in snr_results.items():
            plt.plot(wavelength_values * 1e6, snr_values, marker='o', linestyle='-', label=f'L = {L} m')
        plt.xlabel("Wavelength (µm)")
        plt.ylabel("SNR")
        plt.title("SNR vs. Wavelength for Different Distances")
        plt.legend()
        plt.grid(True)
        plt.show()

# Define parameter variation
wavelength_values = np.linspace(0.8e-6, 1.6e-6, 50)  # Wavelength range from 0.8µm to 1.6µm
L_values = np.linspace(10, 50, 5)  # Distance from 10m to 50m

# Initialize OpticalLinkBudget object with user's parameters
optical_link_updated = OpticalLinkBudgetUpdated(
    Tx_power=2.5e-3,  # Laser power in W
    T_atmos=0.9,  # Atmospheric transmission factor
    theta_div=10e-6,  # Max divergence angle in radians
    sigma_pj=0.5e-6,  # Pointing jitter in radians
    optics_array=[0.999] * 12  # Optical efficiency
)

# Initialize and compute SNR using updated method
snr_calculator_updated = SNRCalculatorUpdated(optical_link_updated)
snr_results_updated = snr_calculator_updated.compute_snr(wavelength_values, L_values)

# Plot the updated SNR results
snr_calculator_updated.plot_snr(wavelength_values, snr_results_updated)
