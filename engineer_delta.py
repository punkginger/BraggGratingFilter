import numpy as np
from scipy.signal import find_peaks
from bragg_grating_tmm import bragg_grating_tmm

def optimize_delta(f_target, delta_array, Lm, Le, nmm, nee, am, ae, N, Lc, Lpi):
    f_range = np.linspace(f_target - 0.25e12, f_target + 0.25e12, 5000)
    delta_f = []
    optimal_delta = 0

    def calculate_error(transmitted_power):
        valleys, _ = find_peaks(-transmitted_power)
        if len(valleys) < 2: return np.inf # the wave should be like a "W"
        
        valley_depths = -transmitted_power[valleys]
        sorted_indices = np.argsort(valley_depths)[::-1] # this make sure that the deepest two are picked
        idx_start = min(valleys[sorted_indices[0]], valleys[sorted_indices[1]])
        idx_end = max(valleys[sorted_indices[0]], valleys[sorted_indices[1]])
        
        trans_between = transmitted_power[idx_start:idx_end]
        if len(trans_between) == 0: return np.inf
        
        local_peak_idx = np.argmax(trans_between)
        fsuspect = f_range[idx_start + local_peak_idx]
        return np.abs(fsuspect - f_target)

    for i in range(len(delta_array)):
        ref, trans, sb = bragg_grating_tmm(f_range, Lm, Le, nmm, nee, am, ae, N, Lc, True, Lpi, delta_array[i])
        delta_f.append(calculate_error(trans))
        min_location = np.argmin(delta_f)
        optimal_delta = delta_array[min_location]
    
    return np.array(delta_f), optimal_delta
    