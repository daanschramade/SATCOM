import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

class OpticalLinkBudget:
    def __init__(self, Tx_power, T_atmos, theta_div, sigma_pj, optics_array, Dr, wave, L, temp):
        self.Tx_power = Tx_power  # Laser power in W
        self.T_atmos = T_atmos  # Atmospheric transmission factor
        self.theta_div = theta_div  # Divergence angle (radians)
        self.sigma_pj = sigma_pj  # Pointing jitter (radians)
        self.optics_array = optics_array  # Optical efficiency
        self.Dr = Dr  # Receiver diameter (m)   
        self.wave = wave  # Wavelength (m)
        self.L = L  # Distance Tx to Rx (m)
        self.temp = temp
        self.direct_rx = ( np.pi * self.Dr / self.wave)**2 # directivity of reciever ##TODO: source this in document
        self.direct_tx = 4 * np.pi / self.theta_div**2 # directivity of transmitter ##TODO: source this in document
    
    def tx_gain(self):
        """Transmitter Gain"""
        G_tx = 8 / (self.theta_div ** 2)
        return 10 * np.log10(G_tx)
    
    def rx_gain(self):
        """Receiver Gain"""
        return 10 * np.log10(((np.pi * self.Dr) / self.wave) ** 2)
    
    def free_space_loss(self):
        """Free space loss"""
        L_fs = self.direct_rx * self.direct_tx * (self.wave / (4 * np.pi * self.L)) ** 2
        return -10 * np.log10(L_fs)  # Changed sign for correct loss representation
    
    def total_optics_loss(self):
        """Optical Loss"""
        optics_loss = np.prod(self.optics_array)
        return -10 * np.log10(optics_loss)  # Changed sign for correct loss representation
    
    def static_pointing_loss(self, r):
        """Static Pointing Loss"""
        theta_pe = r / self.L
        T_pe = np.exp((-2 * theta_pe ** 2) / self.theta_div)
        return -10 * np.log10(T_pe)  # Changed sign for correct loss representation
    
    def jitter_loss(self, p0):
        """Jitter Loss"""
        L_pa = -10 * np.log10((self.theta_div ** 2) / (self.theta_div ** 2 + 4 * self.sigma_pj ** 2))
        L_ps = -10 * np.log10(p0 ** ((4 * self.sigma_pj ** 2) / (self.theta_div ** 2)))
        return L_pa + L_ps
    
    def beam_spread_loss(self, D_spot, r0):
        """Beam Spread Loss"""
        return -10 * np.log10((1 + (D_spot / r0) ** (5/3)) ** (3/5))
    
    def wavefront_loss(self, D_spot, r0):
        """Wavefront Loss"""
        return -10 * np.log10((1 + (D_spot / r0) ** (5/3)) ** (-5/6))
    
    def scintillation_loss(self, p_out, sigma_i):
        """Scinitallation Loss"""
        return -(3.3 - 5.77 * np.sqrt(-np.log(p_out))) * sigma_i ** (4/5)
    
    def compute_link_budget(self, r, p0, p_out, sigma_i, D_spot, r0):
        """This funcion uses the above functions and
           sums them up to find the total link budget"""
        Gtx = self.tx_gain()
        optics_loss = self.total_optics_loss()
        Lfs = self.free_space_loss()
        atmos_loss = -10 * np.log10(self.T_atmos)  # Corrected sign for loss
        L_static = self.static_pointing_loss(r)
        L_jitter = self.jitter_loss(p0)
        L_scint = self.scintillation_loss(p_out, sigma_i)
        L_spread = self.beam_spread_loss(D_spot, r0)
        L_wave = self.wavefront_loss(D_spot, r0)
        Grx = self.rx_gain()
        
        total_losses = optics_loss + Lfs + atmos_loss + L_static + L_jitter + L_scint + L_spread + L_wave
        total_gain = Gtx + Grx
        link_margin = total_gain - total_losses
        
        P_tx_db = 10 * np.log10(self.Tx_power)
        P_rx_db = P_tx_db + link_margin
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
        print(f"  Link Margin: {link_margin:.2f} dB")
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

optical_link = OpticalLinkBudget(
    Tx_power=2.5e-3, # 2.5 mW
    T_atmos=0.9,
    theta_div= 10e-3, #10 mrad
    sigma_pj=0.5e-6,
    optics_array=[0.8] * 12,
    Dr=0.03, # 3 cm (TBR)
    wave=1.55e-6, # 
    L=1,
    temp = 20 # Celsius
)

link_budget = optical_link.compute_link_budget(r=0.01, p0=0.1, p_out=0.1, sigma_i=1, D_spot=1, r0=1)
print(link_budget)