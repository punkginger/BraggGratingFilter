import numpy as np

def bragg_grating_tmm(f_array, Lm, Le, nmm, nee, am, ae, N, Lc, pishift, Lpi, delta):
    """
    f_array: array of frequencies to evaluate (Hz)
    Lm: length of the metal section (m)
    Le: length of the etch section (m)
    nmm: real part of the refractive index of the metal
    nee: real part of the refractive index of the etch
    am: absorption coefficient of the metal (m^-1)
    ae: absorption coefficient of the etch (m^-1)
    N: number of Bragg periods
    Lc: chip length (m)
    pishift: boolean, whether to include a central pi-shift
    Lpi: length of the pi-shift section (m)
    delta: deviation from the exact Bragg condition (m)

    Returns: reflected (array), transmitted (array), stopband_freq (float)
    """
    # Ensure input frequency is a numpy array
    f = np.asarray(f_array)
    reflected = np.zeros(len(f))
    transmitted = np.zeros(len(f))
    
    c = 299792458.0
    wavelength = c / f
    PI= np.pi
    
    
    """ 
    refractive index: n = n_{real} - i * k.
    extinction coefficient(imaginary part): k = wavelength / (4 * pi) * absorption coefficient
    """
    km = wavelength / (4 * PI) * am
    ke = wavelength / (4 * PI) * ae
    nm = nmm - km * 1j
    ne = nee - ke * 1j
    
    
    """
    Propagation constants Beta = 2 * pi / wavelength * n
    phi_+ = Beta_m * Lm + Beta_e * Le
    phi_- = Beta_m * Lm - Beta_e * Le
    """
    beta_m = 2 * PI / wavelength * nm
    beta_e = 2 * PI / wavelength * ne
    
    LAMBDA = Lm + Le
    L_laser = Lc - N * LAMBDA
    
    # Reflection and transmission coefficients at the interface
    r = (nee - nmm) / (nee + nmm)
    t = np.sqrt(1 - r**2)
    
    # Pre-calculate phase terms for the standard Bragg period
    phi_plus = beta_m * Lm + beta_e * Le
    phi_minus = beta_m * Lm - beta_e * Le
    
    
    # Loop through each frequency point
    for i in range(len(f)):
        
        # 1. Calculate precise phase lengths based on pishift AND delta
        if pishift:
            
            phi_pluspi = beta_e[i] * Lpi
            phi_minuspi = beta_e[i] * Lpi
            
            
            phase_left = beta_m[i] * ((L_laser - Lpi) / 2.0 - delta)
            phase_right = beta_m[i] * ((L_laser - Lpi) / 2.0 + delta)
            
            # Build the central pi-shift matrix
            T11pi = (1/t**2) * (np.exp(1j*phi_pluspi) - (r**2) * np.exp(-1j*phi_minuspi))
            T12pi = (r/t**2) * (np.exp(-1j*phi_pluspi) - np.exp(1j*phi_minuspi))
            T21pi = (r/t**2) * (np.exp(1j*phi_pluspi) - np.exp(-1j*phi_minuspi))
            T22pi = (1/t**2) * (np.exp(-1j*phi_pluspi) - (r**2) * np.exp(1j*phi_minuspi))
            Tpi = np.array([[T11pi, T12pi], [T21pi, T22pi]])
            
        else:
            phase_left = beta_m[i] * (L_laser / 2.0 - delta)
            phase_right = beta_m[i] * (L_laser / 2.0 + delta)

        # 2. Build the standard matrices for this frequency
        T11 = (1/t**2) * (np.exp(1j*phi_plus[i]) - (r**2) * np.exp(-1j*phi_minus[i]))
        T12 = (r/t**2) * (np.exp(-1j*phi_plus[i]) - np.exp(1j*phi_minus[i]))
        T21 = (r/t**2) * (np.exp(1j*phi_plus[i]) - np.exp(-1j*phi_minus[i]))
        T22 = (1/t**2) * (np.exp(-1j*phi_plus[i]) - (r**2) * np.exp(1j*phi_minus[i]))
        Tbragg = np.array([[T11, T12], [T21, T22]])
        
        Tlaserleft = np.array([[np.exp(1j*phase_left), 0], [0, np.exp(-1j*phase_left)]])
        Tlaserright = np.array([[np.exp(1j*phase_right), 0], [0, np.exp(-1j*phase_right)]])

        # 3. Assemble the total Transfer Matrix (using @ for matrix multiplication)
        if pishift:
            half_N = int(N // 2)
            # The @ operator in Python 3.5+ does exact matrix multiplication
            T = Tlaserleft @ np.linalg.matrix_power(Tbragg, half_N) @ Tpi @ np.linalg.matrix_power(Tbragg, half_N) @ Tlaserright
        else:
            T = Tlaserleft @ np.linalg.matrix_power(Tbragg, int(N)) @ Tlaserright
            
        # 4. Extract Scattering matrix (S-matrix) parameters
        # S21 is transmission, S11 is reflection. 
        S11 = T[1, 0] / T[0, 0]
        S12 = np.linalg.det(T) / T[0, 0] 
        
        reflected[i] = np.abs(S11)**2
        transmitted[i] = np.abs(S12)**2

    # Find the deepest part of the stopband
    min_loc = np.argmin(transmitted)
    stopband = f[min_loc]
    
    return reflected, transmitted, stopband