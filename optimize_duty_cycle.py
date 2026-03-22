import numpy as np
from scipy.signal import find_peaks, peak_widths
from bragg_grating_tmm import bragg_grating_tmm

def optimize_duty_cycle(f_target, nmm, nee, am, ae, N, Lc, duty_cycle_range, delta, weight_trans, weight_q, weight_rej):
    """
    duty_cycle_range: a tuple or list, e.g., (0.40, 0.85)
    """
    duty_cycles = np.linspace(duty_cycle_range[0], duty_cycle_range[1], 100)
    Lm_array = np.zeros(len(duty_cycles))
    Le_array = np.zeros(len(duty_cycles))
    peak_transmissions = np.zeros(len(duty_cycles))
    q_factors = np.zeros(len(duty_cycles))
    rejections = np.zeros(len(duty_cycles))
    sb_widths = np.zeros(len(duty_cycles))
    
    c = 299792458.0
    f_range = np.linspace(f_target - 1e12, f_target + 0.5e12, 5000)


    def evaluate_filter_analytical(transmitted_power, theoretical_sb_width):
        peaks, _ = find_peaks(transmitted_power, prominence=0.001) # positions of peaks
        
        if len(peaks) == 0:
            return 0.0, 0.0, 0.0
            
        # Lock onto the peak closest to target frequency
        closest_peak_idx = peaks[np.argmin(np.abs(f_range[peaks] - f_target))] # peak[index of closest peak]
        peak_trans = transmitted_power[closest_peak_idx]
        peak_freq = f_range[closest_peak_idx]
        

        if closest_peak_idx == 0 or closest_peak_idx == len(transmitted_power) - 1:
            return 0.0, 0.0, 0.0
            
        # Calculate the Q Factor 
        """
        def peak_widths(x, peaks, rel_height=0.5, prominence_data=None, wlen=None)
        x : sequence
            A signal with peaks.
        peaks : sequence
            Indices of peaks in `x`.
        
        Q = f_peak / FWHM, where FWHM is the Full Width at Half Maximum of the peak.
        """
        widths, _, _, _ = peak_widths(transmitted_power, [closest_peak_idx], rel_height=0.5) # half-height for FWHM
        step_freq = f_range[1] - f_range[0] 
        fwhm_hz = widths[0] * step_freq
        q_factor = peak_freq / fwhm_hz if fwhm_hz > 0 else 0
        
        # Calculate Stopband Rejection Ratio using Theoretical Edges
        left_edge_freq = f_target - (theoretical_sb_width / 2.0)
        right_edge_freq = f_target + (theoretical_sb_width / 2.0)
        
        left_edge_idx = np.argmin(np.abs(f_range - left_edge_freq))
        right_edge_idx = np.argmin(np.abs(f_range - right_edge_freq))
        
        """
        edge_power = (transmitted_power[left_edge_idx] + transmitted_power[right_edge_idx]) / 2.0
        rejection_ratio = peak_trans / edge_power if edge_power > 0 else 0
        """
        stopband_region = transmitted_power[left_edge_idx:right_edge_idx]
        if len(stopband_region) > 0:
            true_noise_floor = np.min(stopband_region)
            rejection_ratio = peak_trans / true_noise_floor if true_noise_floor > 0 else 0
        else:
            rejection_ratio = 0

        return peak_trans, q_factor, rejection_ratio


    for i, D in enumerate(duty_cycles):
        n_eff = D * nmm + (1.0 - D) * nee
        
        delta_n = abs(nmm - nee)
        sb_width_hz = f_target * (2.0 * delta_n / (np.pi * n_eff)) * np.sin(np.pi * D)
        sb_widths[i] = sb_width_hz
        
        Lambda = c / (2.0 * n_eff * f_target)
        Lm = D * Lambda
        Le = (1.0 - D) * Lambda
        Lpi = 2 * Le
        Lm_array[i] = Lm
        Le_array[i] = Le
                
        ref, trans, sb = bragg_grating_tmm(f_range, Lm, Le, nmm, nee, am, ae, N, Lc, True, Lpi, delta)
        p_trans, q_fac, rej_ratio = evaluate_filter_analytical(trans, sb_width_hz)
        
        peak_transmissions[i] = p_trans
        q_factors[i] = q_fac
        rejections[i] = rej_ratio

    """
    x_{norm} = x - x_{min} \ x_{max} - x_{min}
    """
    def normalize_array(arr):
        ptp = np.ptp(arr) # max - min
        if ptp == 0:      # Prevent dividing by zero
            return np.zeros_like(arr)
        return (arr - np.min(arr)) / ptp

    norm_trans = normalize_array(peak_transmissions)
    norm_q = normalize_array(q_factors)
    norm_rej = normalize_array(rejections)

    """
    weight_trans = 0.50  # 50% importance to Peak Power 
    weight_q = 0.25      # 25% importance to Sharpness
    weight_rej = 0.25    # 25% importance to Noise Blocking
    """

    fitness_scores = (weight_trans * norm_trans) + (weight_q * norm_q) + (weight_rej * norm_rej)
    
    best_idx = np.argmax(fitness_scores)
    optimal_D = duty_cycles[best_idx]
    optimal_Lm = Lm_array[best_idx]
    optimal_Le = Le_array[best_idx]
    
    return duty_cycles, Lm_array, Le_array, peak_transmissions, q_factors, rejections, sb_widths, optimal_D, optimal_Lm, optimal_Le, best_idx