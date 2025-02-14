from cProfile import label
from matplotlib import legend
import numpy as np
import math
import matplotlib.pyplot as plt

#TODO: For power loss due to transmission a function of the form Tx_loss can be used, either we have a direct value for transmission or we need to model it somehow

# Simple first order, general functions for calculating the gain and losses
#------------------------------------------------------------------------------------------------------------
# Transmitter part
#---------------------
def Tx_gain(theta_div):
    """
    theta_div: the divergence angle
    """
    Gtx = 8/(theta_div**2)
    return Gtx
#---------------------

# Atmospheric losses and free space
#---------------------
def free_space(wave, L):
    """
    wave: wavelength of the laser used \\
    L: Length from Tx to Rx
    """
    L_fs = ((wave)/(4 * np.pi * L)) ** 2
    return L_fs
#---------------------

def atmos_loss(T):
    T_db = 10 * np.log10(T)
    return T_db

# Static pointing error loss
#---------------------
def theta_pe (r, L):
    """
    r: the radius of the laser beam at FWHM \\
    L: Length from Tx to Rx
    """
    theta_pe = r/L      
    return theta_pe

def static_loss(theta_pe, theta_div):
    """
    theta_pe: the static pointing error angle \\
    theta_div: the divergence angle
    """
    T_pe = np.exp((-2 * theta_pe ** 2)/(theta_div))
    L_pe = 10 * np.log(T_pe)
    return L_pe
#---------------------

# Jitter pointing error loss
#---------------------
def avg_jitter_loss(theta_div):
    """
    theta_div: the divergence angle
    """
    sigma_pj = 0.2 * theta_div          #TODO: Do we have to implement the bessel function to determine sigma_pj?
    T_pa = (theta_div ** 2)/(theta_div ** 2 + 4 * sigma_pj ** 2)
    L_pa = 10 * np.log(T_pa)
    return L_pa

def jitter_scint_loss (p0, sigma_pj, theta_div):
    """
    p0: outage probability \\
    sigma_pj: pointing jitter variance \\
    theta_div: the divergence angle
    """
    L_ps = 10 * np.log(p0 ** ((4 * sigma_pj ** 2)/(theta_div ** 2)))
    return L_ps

def total_jitter_loss(L_pa, L_ps):
    """
    L_pa: average jitter pointing loss \\
    L_ps: Pointing jitter induced scintillation
    """
    L_pj = L_pa + L_ps
    return L_pj
#---------------------

# Beam wander & Beam Spread & Wavefront distortion
#---------------------
def beam_wander (wave, D, r0):
    """
    Beam wander is neglible for downlink applications \\
    wave: wavelength of the laser used \\
    D: Diameter of the spot on the detector \\
    r0: the waist radius of the gaussian representation of the beam
    """
    sigma_bw = np.sqrt(0.54 * (wave/D) ** 2 * (D/r0) ** (5/3))
    return sigma_bw

def aoa (wave, D, ro):
    """
    Mainly occurs when turbulence is close to the receiver, non-neglible for downlink \\
    wave: wavelength of the laser used \\
    D: Diameter of the spot on the detector \\
    r0: the waist radius of the gaussian representation of the beam
    """
    sigma_a = np.sqrt(0.18 * (D/ro) ** (5/3) * (wave/D) ** 2)
    return

def beam_spread_loss(D, r0): #TODO: As mentioned in the slides, beam spread and wavefront share an equation?
    """
    D: Diameter of the spot on the detector \\
    r0: the waist radius of the gaussian representation of the beam
    """
    L_bs = 10 * np.log10((1 + (D/r0) ** (5/3)) ** (3/5))
    return L_bs

def wavefront_loss (D, r0):
    """
    D: Diameter of the spot on the detector \\
    r0: the waist radius of the gaussian representation of the beam
    """
    L_wv = 10 * np.log10((1 + (D/r0) ** (5/3)) ** (-5/6))
    return L_wv
#---------------------

# Scintillation losses
#---------------------
def scint_loss (p_out, sigma_i):
    """ 
    p_out: outage proability \\
    sigma_i: scintillation index
    """
    L_sc = (3.3 - 5.77 * np.sqrt(-np.log(p_out))) * sigma_i ** (4/5)
    return L_sc
#---------------------

# Receiver
#---------------------
def Rx_gain(Dr, wave):
    """
    Dr: diameter of the receiver antenna \\
    wave: wavelength of the laser used
    """
    G_rx = ((np.pi * Dr)/wave) ** 2
    return G_rx
#---------------------

# Optical elements
#---------------------
def optical_fibre(sigma, w0):
    """
    sigma: the jitter std \\
    w0: mode field radius of the gaussian field distribution.
    """
    eff = 0.8145 * 1/(((2 * sigma ** 2)/(w0 ** 2)) + 1)
    return eff


def Total_optics_loss(optics):
    """
    optics: 1D-array containing the transmission, reflection or efficiency factors of each optical element with \\
    which the transmitting laser interacts. Think of mirrors, lenses, optical fibres etc.
    """
    optics_loss = np.sum(optics)
    optics_loss_db = 10 * np.log10(optics_loss)
    return optics_loss_db
#---------------------
#------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Run link budget
    #----------------
    """
    Dependend on the setup of the laser communication bread board,
    the losses and gains etc, per element have to be determined. For example, a
    beam splitter will split the beam for obtaining a baseline, meaning that the power
    changes due to this splitting, and lenses will also have a specific transmission factor.
    The laser signal will go trough the following elements, (for the link budget the path from laser module to detector is needed):
    Laser module --> Modulator --> Lens 1 --> Mirror 1 --> Beam splitter 1 --> ND filter --> Mirror 3 --> ND filter --> Beam splitter 1 --> Mirror 2 --> Beam spliiter 2
    --> Mirror 4 --> Lens 3 --> Detector
    """
    # Define constants
    # Constant parameters and assumed values
    #---------------------
    Tx_power = 1                            # Laser Power [W]
    wave_length = 1                         # Optical Laser wavelength [m]
    L = 1                                   # Distance Tx to Rx [m]
    theta_div = 1                           # Divergence angle [microrad]
    Rx_treshold = 1
    T_atmos = 1
    r = 1 
    p0 = 0.1
    sigma_pj = 0.6
    scint_ind = 1
    D_spot = 1
    r0 = 1
    Drx = 1

    # Optical fibre, L1, M1, BS1, ND, M3, ND, BS1, M2, BS2, M4, L3 
    optics_array = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] #FIXME: Still dummy
    #---------------------

    # Losses and gains in link budget #FIXME: empty function parameters
    laser_power = ...
    Gtx = Tx_gain(theta_div)
    optics_loss = Total_optics_loss(optics_array) #Including elements from both Tx and Rx
    #--
    Lfs = free_space(wave_length, L)
    atmos_loss = atmos_loss(T_atmos)
    #--
    sys_pl = static_loss(theta_pe(r, L), theta_div)
    total_jit = total_jitter_loss(avg_jitter_loss(theta_div), jitter_scint_loss(p0, sigma_pj, theta_div))
    scint_loss = scint_loss(p0, scint_ind)
    beam_spread = beam_spread_loss(D_spot, r0)
    wave_front = wavefront_loss(D_spot, r0)
    #--
    Grx = Rx_gain(Drx, wave_length)
    #--
    total_losses = Gtx + optics_loss + Lfs + atmos_loss + sys_pl + total_jit + scint_loss + beam_spread + wave_front + Grx
    link_margin = total_losses + Tx_power - Rx_treshold
    print(f'Total losses: {total_losses} [dB]')
    print(f'Link margin: {link_margin} [dB]')
    #----------------

# Trade off pointing accuracy and beam divergence - See slide 61 of lecture 16 Laser instrumentation
#---------------------------------------------------------------------------------------------------
def geometric_loss(Lfs, Gtx):
    """
    Lfs: free space loss
    Gtx: Tx gain
    """
    geo_loss = Lfs + Gtx
    return geo_loss

def system_tradeoff(geo_loss, L_pj, theta_div):
    """
    theta_div: list of divergence angles [0 to 100] or so.
    geo_loss: geometric loss - should be a list containing loss values that are dependend on divergence angle
    L_pj: total jitter loss - should be a list containing loss values that are dependend on divergence angle
    """
    comb = geo_loss + L_pj
    
    plt.plot(theta_div, geo_loss, label='Geometric loss')
    plt.plot(theta_div, L_pj, label='Pointing Jitter loss')
    plt.plot(theta_div, comb, label='Combined performance')
    plt.grid()
    plt.xlabel(f'Divergence Angle [$\mu rad$]')
    plt.ylabel(f'Link Losses [dB]')
    plt.legend()
    plt.show()
    return  
#---------------------------------------------------------------------------------------------------
