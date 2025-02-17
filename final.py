import numpy as np

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
        self.theta_div = theta_div  # Divergence angle (radians)
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
        self.D_spot = self.L * self.theta_div # Beam spot size (m)
        self.eta_rx = eta_rx # reciever efficiency

    def tx_gain(self):
        """Transmitter Gain"""
        G_tx = 8 / (self.theta_div ** 2)
        return 10 * np.log10(G_tx)

    def rx_gain(self):
        """Receiver Gain"""
        return 10 * np.log10(self.eta_rx*((np.pi * self.Dr) / self.wave) ** 2)

    def free_space_loss(self):
        """Free space loss using the correct Friis equation"""
        L_fs = (4 * np.pi * self.L / self.wave) ** 2
        return 10 * np.log10(L_fs)

    def total_optics_loss(self):
        """Optical Loss"""
        optics_loss = np.prod(self.optics_array)
        return 10 * np.log10(optics_loss)

    def static_pointing_loss(self):
        """Static Pointing Loss"""
        theta_pe = self.r / self.L
        T_pe = np.exp((-2 * theta_pe ** 2) / self.theta_div**2) 
        T_pe = max(T_pe, 1e-6) # it was throwing inf errors without this
        return 10 * np.log10(T_pe)

    def jitter_loss(self):
        """Jitter Loss"""
        return 10 * np.log10(self.theta_div**2 / (self.theta_div**2 + 4 * self.sigma_pj**2))

    def beam_spread_loss(self):
        """Beam Spread Loss"""
        return 10 * np.log10((1 + (self.D_spot / self.r0) ** (5/3)) ** (3/5))

    def wavefront_loss(self):
        """Wavefront Loss"""
        return 10 * np.log10((1 + (self.D_spot / self.r0) ** (5/3)) ** (-5/6))

    def scintillation_loss(self):
        """Scintillation Loss"""
        p_out = max(self.p_out, 1e-6)  # Prevent log(0) errors
        return (3.3 - 5.77 * np.sqrt(-np.log(p_out))) * self.sigma_i ** (4/5)
    
    def atmos_loss(self):
        """Atmospheric Loss"""
        return 10 * np.log10(self.T_atmos)

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
        P_rx_db = P_tx_db + total_gain - total_losses
        P_rx = 10 ** (P_rx_db / 10)

        sigma2_thermal = 1.38e-23 * (273.15 + self.temp) / 50  # Thermal noise
        I_d = P_rx / (1.6e-19 * 0.99)  # Assume quantum efficiency of 0.99
        sigma2_shot = 2 * 1.6e-19 * I_d * 10e6  # Shot noise
        sigma2 = sigma2_thermal + sigma2_shot  # Total noise power
        snr_db = 10 * np.log10(P_rx / sigma2)

        print("Losses and Gains:")
        print(f"  Transmitter Gain: {Gtx:.2f} dB")
        print(f"  Receiver Gain: {Grx:.2f} dB")
        print(f"  Optical Loss: {optics_loss:.2f} dB")
        print(f"  Free Space Loss: {Lfs:.2f} dB")
        print(f"  Atmospheric Loss: {atmos_loss:.2f} dB")
        print(f"  Static Pointing Loss: {L_static:.2f} dB")
        print(f"  Jitter Loss: {L_jitter:.2f} dB")
        print(f"  Scintillation Loss: {L_scint:.2f} dB")
        print(f"  Beam Spread Loss: {L_spread:.2f} dB")
        print(f"  Wavefront Loss: {L_wave:.2f} dB")
        print(f"  Total Losses: {total_losses:.2f} dB")
        print(f"  Link Margin: {total_gain - total_losses:.2f} dB")
        print(f"  Computed SNR: {snr_db:.2f} dB")

        return {
            "L": self.L,
            "Wavelength": self.wave,
            "Free Space Loss": Lfs,
            "Total Losses": total_losses,
            "P_rx (dB)": P_rx_db,
            "P_rx (W)": P_rx,
            "SNR (dB)": snr_db
        }

# # From lecture slides (works):
optical_link = OpticalLinkBudget(
    Tx_power=2.5e-3,  # 2.5 mW
    T_atmos=0.9,
    theta_div=10e-6,  # 10 mrad
    sigma_pj=0.5e-6,
    optics_array=[0.99] * 12,
    Dr=0.01,  # 3 cm receiver aperture
    wave=1.55e-6,  # 1.55 μm wavelength (infrared)
    L=1,  # Distance in meters
    temp=20,  # Celsius
    r=0.20,  # Static pointing error radius
    p0=0.1,  # Initial pointing probability
    p_out=0.01,  # Scintillation outage probability
    sigma_i=1,  # Scintillation index
    r0=1,  # Fried parameter
    eta_rx = 0.7 # Reciever efficiency
)

# Compute the link budget with no extra arguments
link_budget = optical_link.compute_link_budget()
print(link_budget)

# # From lecture slides (works):
# optical_link = OpticalLinkBudget(
#     Tx_power=1,  # 2.5 mW
#     T_atmos=0.9,
#     theta_div=20e-6,  # 10 mrad
#     sigma_pj=0.5e-6,
#     optics_array=[0.99] * 12,
#     Dr=0.5,  # 3 cm receiver aperture
#     wave=1.55e-6,  # 1.55 μm wavelength (infrared)
#     L=1000e3,  # Distance in meters
#     temp=20,  # Celsius
#     r=0.20,  # Static pointing error radius
#     p0=0.1,  # Initial pointing probability
#     p_out=0.01,  # Scintillation outage probability
#     sigma_i=1,  # Scintillation index
#     r0=1,  # Fried parameter
#     eta_rx = 0.6# Reciever efficiency
# )

# # Compute the link budget with no extra arguments
# link_budget = optical_link.compute_link_budget()
# print(link_budget)
