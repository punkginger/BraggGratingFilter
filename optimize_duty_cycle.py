import numpy as np
from scipy.signal import find_peaks
from bragg_grating_tmm import bragg_grating_tmm

def optimize_duty_cycle(f_target, nmm, nee, am, ae, N, Lc, duty_cycle_range, delta):
    """
    duty_cycle_range: a tuple or list, e.g., (0.40, 0.85)
    """
    duty_cycles = np.linspace(duty_cycle_range[0], duty_cycle_range[1], 100)
    Lm_array = np.zeros(len(duty_cycles))
    Le_array = np.zeros(len(duty_cycles))
    peak_transmissions = np.zeros(len(duty_cycles))
    
    c = 299792458.0
    f_range = np.linspace(f_target - 1e12, f_target + 0.5e12, 5000)

    def evaluate_filter_quality(transmitted_power):
        valleys, properties = find_peaks(-transmitted_power, prominence=0.01, distance=50)
        if len(valleys) < 2: 
            return 0.0 
        
        valley_depths = properties['prominences']
        sorted_indices = np.argsort(valley_depths)[::-1] 
        
        idx_start = min(valleys[sorted_indices[0]], valleys[sorted_indices[1]])
        idx_end = max(valleys[sorted_indices[0]], valleys[sorted_indices[1]])
        
        trans_between = transmitted_power[idx_start:idx_end]
        if len(trans_between) == 0: 
            return 0.0
        
        return np.max(trans_between)

    for i, D in enumerate(duty_cycles):
        
        # Calculate the effective index for this duty cycle  
        n_eff = D * nmm + (1.0 - D) * nee
        # Calculate the total period required to hit f_target exactly
        Lambda = c / (2.0 * n_eff * f_target)
        # Split the period into precise Lm and Le
        Lm = D * Lambda
        Le = (1.0 - D) * Lambda
        Lpi = 2 * Le
        Lm_array[i] = Lm
        Le_array[i] = Le
               
        ref, trans, sb = bragg_grating_tmm(f_range, Lm, Le, nmm, nee, am, ae, N, Lc, True, Lpi, delta)
        
        peak_transmissions[i] = evaluate_filter_quality(trans)

    
    best_idx = np.argmax(peak_transmissions)
    optimal_D = duty_cycles[best_idx]
    optimal_Lm = Lm_array[best_idx]
    optimal_Le = Le_array[best_idx]
    
    return Lm_array, Le_array, peak_transmissions, optimal_D, optimal_Lm, optimal_Le