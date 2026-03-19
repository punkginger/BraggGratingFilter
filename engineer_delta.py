import numpy as np
from scipy.signal import find_peaks
from bragg_grating_tmm import bragg_grating_tmm

def optimize_delta(f_target, delta_array, Lm, Le, nmm, nee, am, ae, N, Lc, Lpi):
    f_test = np.linspace(f_target - 0.25e12, f_target + 0.25e12, 2000)
    
    error_array = []
    
    for current_delta in delta_array:
        # Run the TMM engine (pishift is strictly True here!)
        ref, trans, _ = bragg_grating_tmm(
            f_test, Lm, Le, nmm, nee, am, ae, int(N), Lc, True, Lpi, current_delta
        )
        
        # --- THE BRILLIANT STOPBAND ISOLATION LOGIC ---
        # 1. Find the valleys (the edges of the stopband)
        valleys, _ = find_peaks(-trans)
        
        if len(valleys) < 2:
            # If there's no stopband at all, the error is infinite
            error_array.append(float('inf'))
            continue
            
        # 2. Find the two deepest valleys
        valley_depths = -trans[valleys]
        sorted_indices = np.argsort(valley_depths)[::-1]
        idx_start = min(valleys[sorted_indices[0]], valleys[sorted_indices[1]])
        idx_end = max(valleys[sorted_indices[0]], valleys[sorted_indices[1]])
        
        # 3. Isolate the spectrum exactly between the two walls
        trans_between = trans[idx_start:idx_end]
        
        if len(trans_between) == 0:
            error_array.append(float('inf'))
            continue
            
        # 4. Find the local peak *inside* the stopband cavity
        local_peak_idx = np.argmax(trans_between)
        
        # 5. Map it back to the actual frequency array
        peak_freq = f_test[idx_start + local_peak_idx]
        
        # Calculate how far off we are from your dream target!
        error = abs(peak_freq - f_target)
        error_array.append(error)
        
    error_array = np.array(error_array)
    
    # Find the absolute best delta value from our valid errors
    minloc = np.argmin(error_array)
    optimal_delta = delta_array[minloc]
        
    return error_array, optimal_delta