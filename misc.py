import numpy as np
import matplotlib.pyplot as plt

class OpticalLinkBudget:
    def __init__(self, 
                 Tx_power, 
                 T_atmos, 
                 theta_div, 
                 sigma_pj, 
                 optics_array, 
                 Dr, 
                 wave, 
                 L, 
                 temp, 
                 r, 
                 p0, 
                 p_out, 
                 sigma_i, 
                 r0, 
                 eta_rx):
        self.Tx_power = Tx_power  # Laser power in W
        self.T_atmos = T_atmos  # Atmospheric transmission factor
        self.theta_div = theta_div  # Beam divergence angle (radians)
        self.sigma_pj = sigma_pj  # Pointing jitter (radians)
        self.optics_array = optics_array  # Optical efficiency
        self.Dr = Dr  # Receiver diameter (m)   
        self.wave = wave  # Wavelength (m)
        self.L = L  # Distance Tx to Rx (m)
        self.temp = temp  # Temperature in Celsius
        self.r = r  # Static pointing error radius (m)
        self.p0 = p0  # Initial pointing probability
        self.p_out = p_out  # Scintillation outage probability
        self.sigma_i = sigma_i  # Scintillation index
        self.r0 = r0  # Fried parameter (coherence length)
        self.eta_rx = eta_rx  # Receiver efficiency

    def tx_gain(self):
        """Transmitter Gain"""
        G_tx = 8 / (self.theta_div ** 2)
        return 10 * np.log10(G_tx)

    def rx_gain(self):
        """Receiver Gain"""
        return 10 * np.log10(self.eta_rx * ((np.pi * self.Dr) / self.wave) ** 2)

    def free_space_loss(self):
        """Free space loss using Friis equation"""
        L_fs = (4 * np.pi * self.L / self.wave) ** 2
        return -np.abs(10 * np.log10(L_fs))

    def total_optics_loss(self):
        """Optical Loss"""
        optics_loss = np.prod(self.optics_array)
        return -np.abs(10 * np.log10(optics_loss))

    def static_pointing_loss(self):
        """Static Pointing Loss"""
        theta_pe = self.r / self.L
        T_pe = np.exp((-2 * theta_pe ** 2) / self.theta_div**2) 
        return -np.abs(10 * np.log10(max(T_pe, 1e-6)))

    def jitter_loss(self):
        """Jitter Loss"""
        return -np.abs(10 * np.log10(self.theta_div**2 / (self.theta_div**2 + 4 * self.sigma_pj**2)))

    def beam_spread_loss(self):
        """Beam Spread Loss"""
        D_spot = self.L * self.theta_div  # Ensure this updates dynamically
        return -np.abs(10 * np.log10((1 + (D_spot / self.r0) ** (5/3)) ** (3/5)))

    def wavefront_loss(self):
        """Wavefront Loss"""
        D_spot = self.L * self.theta_div  # Ensure this updates dynamically
        return -np.abs(10 * np.log10((1 + (D_spot / self.r0) ** (5/3)) ** (-5/6)))

    def scintillation_loss(self):
        """Scintillation Loss"""
        p_out = max(self.p_out, 1e-6)  # Prevent log(0) errors
        return -np.abs((3.3 - 5.77 * np.sqrt(-np.log(p_out))) * self.sigma_i ** (4/5))

    def atmos_loss(self):
        """Atmospheric Loss"""
        return -np.abs(10 * np.log10(self.T_atmos))
    
    def snr(self, P_rx):
        """Signal to Noise ratio"""
        sigma2_thermal = 1.38e-23 * (273.15 + self.temp) / 50  # Thermal noise
        I_d = P_rx / (1.6e-19 * 0.99)  # Assume quantum efficiency of 0.99
        sigma2_shot = 2 * 1.6e-19 * I_d  # Shot noise
        sigma2 = sigma2_thermal + sigma2_shot  # Total noise power
        snr = P_rx / sigma2
        return snr

    def compute_link_budget(self):
        """Computes the full optical link budget"""


        Gtx = self.tx_gain()
        Grx = self.rx_gain()
        optics_loss = self.total_optics_loss()
        Lfs = self.free_space_loss()
        atmos_loss = self.atmos_loss()
        L_static = self.static_pointing_loss()
        L_jitter = self.jitter_loss()
        L_scint = self.scintillation_loss()
        L_spread = self.beam_spread_loss()
        L_wave = self.wavefront_loss()

        total_losses = optics_loss + Lfs + atmos_loss + L_static + L_jitter + L_scint + L_spread + L_wave
        total_gain = Gtx + Grx

        P_tx_db = 10 * np.log10(self.Tx_power)
        P_rx_db = P_tx_db + total_gain + total_losses
        P_rx = 10 ** (P_rx_db / 10)

        snr = self.snr(P_rx)
        snr_db = 10 * np.log10(snr)

        # Debugging: Print key loss/gain values to verify updates
        print(f"L: {self.L:.1f} m, Wavelength: {self.wave*1e6:.2f} µm, Free Space Loss: {Lfs:.2f} dB, Rx Gain: {Grx:.2f} dB, SNR: {snr_db:.2f} dB")

        return {"SNR (dB)": snr_db}

# --- Simulation ---
wavelength_values = np.linspace(0.8e-6, 1.6e-6, 50)
L_values = np.linspace(10, 50000, 5)

optical_link = OpticalLinkBudget(
    Tx_power=2.5e-3, T_atmos=1, theta_div=10e-6, sigma_pj=0.5e-6, optics_array=[0.999]*12,
    Dr=0.03, wave=1.55e-6, L=1, temp=20, r=0.20, p0=0.001, p_out=0.01,
    sigma_i=1, r0=1, eta_rx=0.7
)

snr_results = {}

for L in L_values:
    optical_link.L = L
    snr_vals = []
    for wave in wavelength_values:
        optical_link.wave = wave
        snr_vals.append(optical_link.compute_link_budget()["SNR (dB)"])
    snr_results[L] = snr_vals

plt.figure(figsize=(8, 6))
for L, snr_values in snr_results.items():
    plt.plot(wavelength_values * 1e6, snr_values, label=f'L = {L} m')
plt.xlabel("Wavelength (µm)")
plt.ylabel("SNR (dB)")
plt.title("SNR vs. Wavelength for Different Distances")
plt.legend()
plt.grid(True)
plt.show()
