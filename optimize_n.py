import numpy as np
from scipy.signal import find_peaks, peak_widths
from bragg_grating_tmm import bragg_grating_tmm

def optimize_n(f_target, nmm, nee, am, ae, fixed_duty, n_min, n_max, Lc, delta,  weight_trans, weight_q, weight_rej):        
    n_array = np.arange(n_min, n_max + 2, 2)
    peak_transmissions = np.zeros(len(n_array))
    q_factors = np.zeros(len(n_array))
    rejections = np.zeros(len(n_array))
    c = 299792458.0
    f_range = np.linspace(f_target - 1e12, f_target + 0.5e12, 5000)


    n_eff = fixed_duty * nmm + (1.0 - fixed_duty) * nee
    delta_n = abs(nmm - nee)
    sb_width_hz = f_target * (2.0 * delta_n / (np.pi * n_eff)) * np.sin(np.pi * fixed_duty)
    Lambda = c / (2.0 * n_eff * f_target)
    Lm = fixed_duty * Lambda
    Le = (1.0 - fixed_duty) * Lambda
    Lpi = 2 * Le

    def evaluate_filter_analytical(transmitted_power):
        peaks, _ = find_peaks(transmitted_power, prominence=0.001)
        if len(peaks) == 0:
            return 0.0, 0.0, 0.0
            
        closest_peak_idx = peaks[np.argmin(np.abs(f_range[peaks] - f_target))]
        peak_trans = transmitted_power[closest_peak_idx]
        peak_freq = f_range[closest_peak_idx]
        
        if closest_peak_idx == 0 or closest_peak_idx == len(transmitted_power) - 1:
            return 0.0, 0.0, 0.0
            
        widths, _, _, _ = peak_widths(transmitted_power, [closest_peak_idx], rel_height=0.5)
        df = f_range[1] - f_range[0] 
        fwhm_hz = widths[0] * df
        q_factor = peak_freq / fwhm_hz if fwhm_hz > 0 else 0
        
        left_edge_freq = f_target - (sb_width_hz / 2.0)
        right_edge_freq = f_target + (sb_width_hz / 2.0)
        
        left_fence_idx = np.argmin(np.abs(f_range - left_edge_freq))
        right_fence_idx = np.argmin(np.abs(f_range - right_edge_freq))
        
        stopband_region = transmitted_power[left_fence_idx:right_fence_idx]
        
        if len(stopband_region) > 0:
            true_noise_floor = np.min(stopband_region)
            rejection_ratio = peak_trans / true_noise_floor if true_noise_floor > 0 else 0
        else:
            rejection_ratio = 0
            
        return peak_trans, q_factor, rejection_ratio


    for i, N in enumerate(n_array):
        ref, trans, sb = bragg_grating_tmm(f_range, Lm, Le, nmm, nee, am, ae, N, Lc, True, Lpi, delta)
        p_trans, q_fac, rej_ratio = evaluate_filter_analytical(trans)
        
        peak_transmissions[i] = p_trans
        q_factors[i] = q_fac
        rejections[i] = rej_ratio


    def normalize_array(arr):
        ptp = np.ptp(arr)
        if ptp == 0: return np.zeros_like(arr)
        return (arr - np.min(arr)) / ptp

    norm_trans = normalize_array(peak_transmissions)
    norm_q = normalize_array(q_factors)
    norm_rej = normalize_array(rejections)

    fitness_scores = (weight_trans * norm_trans) + (weight_q * norm_q) + (weight_rej * norm_rej)
    best_idx = np.argmax(fitness_scores)
    optimal_N = n_array[best_idx]
    
    return n_array, peak_transmissions, q_factors, rejections, sb_width_hz, optimal_N, best_idx