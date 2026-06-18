import numpy as np

def bragg_grating_tmm(f_array, Lm, Le, nmm, nee, am, ae, N, Lc, pishift, Lpi, delta,
                      N_sub=10, d_trans=0.0, sigma_Le=0.0, seed=None):
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
    N_sub: number of sub-layers for the interface transition
    d_trans: total thickness of the graded transition layer (m)
    sigma_Le: standard deviation of etched length variation across periods (m)
    seed: optional random seed for disorder

    Returns: reflected (array), transmitted (array), stopband_freq (float)
    """
    # Ensure input frequency is a numpy array
    f = np.asarray(f_array)
    reflected = np.zeros(len(f))
    transmitted = np.zeros(len(f))
    
    c = 299792458.0
    wavelength = c / f
    PI = np.pi

    if seed is not None:
        np.random.seed(int(seed))

    le_noise = np.zeros(int(N))
    if sigma_Le > 0.0:
        le_noise = np.random.normal(0.0, sigma_Le, int(N))

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
        how many radians the wave's phase changed per meter it travels
    phi_+ = Beta_m * Lm + Beta_e * Le
    phi_- = Beta_m * Lm - Beta_e * Le
    """
    beta_m = 2 * PI / wavelength * nm
    beta_e = 2 * PI / wavelength * ne
    
    LAMBDA = Lm + Le #one period
    L_laser = Lc - N * LAMBDA #total len of grating 
    
    # Reflection and transmission coefficients at the interface
    r = (nee - nmm) / (nee + nmm)
    t = np.sqrt(1 - r**2)
    
    # phase terms for the standard Bragg period
    phi_plus = beta_m * Lm + beta_e * Le
    phi_minus = beta_m * Lm - beta_e * Le
    
    
    for i in range(len(f)):
        
        # Calculate precise phase lengths based on pishift and delta
        if pishift:
            
            phi_pluspi = beta_e[i] * Lpi # no metal part
            phi_minuspi = beta_e[i] * Lpi
            
            
            phi_left = beta_m[i] * ((L_laser - Lpi) / 2.0 - delta)
            phi_right = beta_m[i] * ((L_laser - Lpi) / 2.0 + delta)
            
            # Build the central pi-shift matrix
            T11pi = (1/t**2) * (np.exp(1j*phi_pluspi) - (r**2) * np.exp(-1j*phi_minuspi))
            T12pi = (r/t**2) * (np.exp(-1j*phi_pluspi) - np.exp(1j*phi_minuspi))
            T21pi = (r/t**2) * (np.exp(1j*phi_pluspi) - np.exp(-1j*phi_minuspi))
            T22pi = (1/t**2) * (np.exp(-1j*phi_pluspi) - (r**2) * np.exp(1j*phi_minuspi))
            Tpi = np.array([[T11pi, T12pi], [T21pi, T22pi]])
            
        else:
            phi_left = beta_m[i] * (L_laser / 2.0 - delta)
            phi_right = beta_m[i] * (L_laser / 2.0 + delta)

        # Build the standard matrices without pi-shift
        T11 = (1/t**2) * (np.exp(1j*phi_plus[i]) - (r**2) * np.exp(-1j*phi_minus[i]))
        T12 = (r/t**2) * (np.exp(-1j*phi_plus[i]) - np.exp(1j*phi_minus[i]))
        T21 = (r/t**2) * (np.exp(1j*phi_plus[i]) - np.exp(-1j*phi_minus[i]))
        T22 = (1/t**2) * (np.exp(-1j*phi_plus[i]) - (r**2) * np.exp(1j*phi_minus[i]))
        T_bragg = np.array([[T11, T12], [T21, T22]])

        wl = wavelength[i]
        nmi = nm[i]
        nei = ne[i]
        Pm = np.array([[np.exp(1j * beta_m[i] * Lm), 0], [0, np.exp(-1j * beta_m[i] * Lm)]])
        Pe = np.array([[np.exp(1j * beta_e[i] * Le), 0], [0, np.exp(-1j * beta_e[i] * Le)]])

        # calculate the transfer matrices for transition layers
        if d_trans > 0.0 and N_sub > 0:
            dz = d_trans / float(N_sub) # thickness of each sub-layer
            T_m2e = np.eye(2, dtype=complex) # initialize the transfer matrix from metal to etch
            curr_n = nmi
            for s in range(1, int(N_sub) + 1):
                next_n = nmi + (nei - nmi) * s / float(N_sub + 1) # linear interpolation of refractive index
                rs = (curr_n - next_n) / (curr_n + next_n)
                ts = np.sqrt(1 - rs**2)
                beta_s = 2.0 * PI / wl * next_n
                P_s = np.array([[np.exp(1j * beta_s * dz), 0], [0, np.exp(-1j * beta_s * dz)]])
                T_m2e = np.array([[1.0 / ts, rs / ts], [rs / ts, 1.0 / ts]]) @ P_s @ T_m2e
                curr_n = next_n
            rs = (curr_n - nei) / (curr_n + nei)
            ts = np.sqrt(1 - rs**2)
            T_m2e = np.array([[1.0 / ts, rs / ts], [rs / ts, 1.0 / ts]]) @ T_m2e

            T_e2m = np.eye(2, dtype=complex) # initialize the transfer matrix from etch to metal
            curr_n = nei
            for s in range(1, int(N_sub) + 1):
                next_n = nei + (nmi - nei) * s / float(N_sub + 1)
                rs = (curr_n - next_n) / (curr_n + next_n)
                ts = np.sqrt(1 - rs**2)
                beta_s = 2.0 * PI / wl * next_n
                P_s = np.array([[np.exp(1j * beta_s * dz), 0], [0, np.exp(-1j * beta_s * dz)]])
                T_e2m = np.array([[1.0 / ts, rs / ts], [rs / ts, 1.0 / ts]]) @ P_s @ T_e2m
                curr_n = next_n
            rs = (curr_n - nmi) / (curr_n + nmi)
            ts = np.sqrt(1 - rs**2)
            T_e2m = np.array([[1.0 / ts, rs / ts], [rs / ts, 1.0 / ts]]) @ T_e2m
        else:
            T_m2e = np.array([[1.0 / t, r / t], [r / t, 1.0 / t]], dtype=complex)
            T_e2m = T_m2e

        if sigma_Le > 0.0:
            T_grating_total = np.eye(2, dtype=complex)
            for p in range(int(N)):
                # change the etched length for this period
                Le_p = max(0.0, Le + le_noise[p])
                # Build the transfer matrix for this period with the tweaked Le
                Pe_p = np.array([[np.exp(1j * beta_e[i] * Le_p), 0], [0, np.exp(-1j * beta_e[i] * Le_p)]])
                T_period = T_e2m @ Pe_p @ T_m2e @ Pm
                T_grating_total = T_period @ T_grating_total
        else:
            if d_trans > 0.0 and N_sub > 0:
                T_period = T_e2m @ Pe @ T_m2e @ Pm
            else:
                T_period = T_bragg
            T_grating_total = np.linalg.matrix_power(T_period, int(N))

        T_left = np.array([[np.exp(1j*phi_left), 0], [0, np.exp(-1j*phi_left)]])
        T_right = np.array([[np.exp(1j*phi_right), 0], [0, np.exp(-1j*phi_right)]])

        # Assemble the total Transfer Matrix 
        if pishift:
            half_N = int(N // 2)
            if sigma_Le > 0.0:
                T_left_half = np.eye(2, dtype=complex)
                for p in range(half_N):
                    Le_p = max(0.0, Le + le_noise[p])
                    Pe_p = np.array([[np.exp(1j * beta_e[i] * Le_p), 0], [0, np.exp(-1j * beta_e[i] * Le_p)]])
                    T_left_half = (T_e2m @ Pe_p @ T_m2e @ Pm) @ T_left_half

                T_right_half = np.eye(2, dtype=complex)
                for p in range(half_N, int(N)):
                    Le_p = max(0.0, Le + le_noise[p])
                    Pe_p = np.array([[np.exp(1j * beta_e[i] * Le_p), 0], [0, np.exp(-1j * beta_e[i] * Le_p)]])
                    T_right_half = (T_e2m @ Pe_p @ T_m2e @ Pm) @ T_right_half
            else:
                T_left_half = np.linalg.matrix_power(T_period, half_N)
                T_right_half = T_left_half

            Ppi = np.array([[np.exp(1j * beta_e[i] * Lpi), 0], [0, np.exp(-1j * beta_e[i] * Lpi)]])
            Tpi = T_e2m @ Ppi @ T_m2e
            T = T_left @ T_left_half @ Tpi @ T_right_half @ T_right
        else:
            T = T_left @ T_grating_total @ T_right
            
        """
        Extract Scattering matrix (S-matrix) parameters
        S12/S21 transmission
            S21 = 1 / T11
            S12 = det(T) / T11
            for reciprocal optical structures, det(T) = 1
        S11 reflection
            S11 = T21 / T11
        """
        S11 = T[1, 0] / T[0, 0]
        S12 = np.linalg.det(T) / T[0, 0] 
        
        # get the ref and trans power array, power is proportional to amplitude
        reflected[i] = np.abs(S11)**2
        transmitted[i] = np.abs(S12)**2

    # Find the deepest part of the stopband
    min_location = np.argmin(transmitted)
    stopband = f[min_location] 
    
    return reflected, transmitted, stopband