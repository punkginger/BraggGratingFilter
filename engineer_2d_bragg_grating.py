import numpy as np
from bragg_grating_tmm import bragg_grating_tmm

def run_2d_sweep(f_target, Lpi_array, delta_array, Lm, Le, nmm, nee, am, ae, N, Lc):
    """
    Executes a comprehensive 2D parameter sweep to find the global optimal 
    pi-shift dimensions for a single-mode QCL.
    """
    print(f"Brewing the physics! Sweeping a {len(Lpi_array)}x{len(delta_array)} grid...")
    
    # We look tightly around the target frequency to find the defect spike
    # 2000 points gives us a brilliant high-resolution look!
    f_test = np.linspace(f_target - 0.2e12, f_target + 0.2e12, 2000)
    
    # Initialize our beautiful 2D results matrix with zeros
    rows = len(Lpi_array)
    cols = len(delta_array)
    error_matrix = np.zeros((rows, cols))
    
    min_error = float('inf')
    best_Lpi = 0.0
    best_delta = 0.0
    
    # The Double Loop: Testing every single combination of Lpi and delta
    for i in range(rows):
        for j in range(cols):
            current_Lpi = Lpi_array[i]
            current_delta = delta_array[j]
            
            # Run the TMM physics engine (Notice pishift is permanently True here)
            ref, trans, _ = bragg_grating_tmm(
                f_test, Lm, Le, nmm, nee, am, ae, int(N), Lc, True, current_Lpi, current_delta
            )
            
            # Find the exact frequency where the magic door opens (peak transmission)
            peak_idx = np.argmax(trans)
            peak_freq = f_test[peak_idx]
            
            # Calculate how far off we are from the user's dream target
            error = abs(peak_freq - f_target)
            
            # Save this error to our 2D heatmap matrix
            error_matrix[i, j] = error
            
            # Track the absolute best "Global Minimum" combination
            if error < min_error:
                min_error = error
                best_Lpi = current_Lpi
                best_delta = current_delta
                
    return error_matrix, best_Lpi, best_delta