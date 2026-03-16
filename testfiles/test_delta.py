import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engineer_delta import optimize_delta

def run_delta_test():
    print("Finding the best delta parameter for target frequency...")
    
    # 1. PERFECT BRAGG PARAMETERS FOR 2.0 THz!
    f_target = 2.0e12  
    Lm = 11.12e-6      # Corrected to center the stopband at 2.0 THz
    Le = 11.12e-6      # Corrected to center the stopband at 2.0 THz
    nmm = 3.67                
    nee = 3.07                 
    am = 750                 
    ae = 3812               
    N = 40             # Increased to 40 so the stopband is deep and obvious!
    Lc = 2e-3                  
    Lpi = 11.12e-6     # Quarter-wave phase shift (Lambda / 2)
    
    # 2. The 1D Sweep Array for Delta (-5 um to 5 um)
    delta_sweep = np.linspace(-5e-6, 5e-6, 50)

    # 3. Call the Unit 3 engine
    print("Running the heavy physics engine, please wait...")
    error_results, optimal_delta = optimize_delta(
        f_target, delta_sweep, Lm, Le, nmm, nee, am, ae, N, Lc, Lpi
    )

    print(f"Optimization Complete, darling!")
    print(f"Target Frequency: {f_target / 1e12:.3f} THz")
    print(f"Absolute Best Delta: {optimal_delta * 1e6:.3f} um")

    # 4. Filter out any 'inf' errors so the graph doesn't break
    valid_idx = np.where(error_results != float('inf'))[0]
    
    if len(valid_idx) == 0:
        print("CRITICAL: The engine still couldn't see the stopband! We need to check find_peaks.")
        return

    # 5. Draw the beautiful diagnostic plot
    plt.figure(figsize=(10, 6))
    plt.plot(delta_sweep[valid_idx] * 1e6, error_results[valid_idx] / 1e12, 'b-', linewidth=3, label='Frequency Error')
    
    min_idx = np.argmin(error_results[valid_idx])
    actual_best_delta = delta_sweep[valid_idx][min_idx]
    actual_best_error = error_results[valid_idx][min_idx]
    
    plt.plot(actual_best_delta * 1e6, actual_best_error / 1e12, '*r', markersize=15, label='Optimal Delta')
    
    plt.xlabel('Deviation $\delta$ [$\mu$m]', fontsize=14)
    plt.ylabel('Distance from Target Frequency [THz]', fontsize=14)
    plt.title('Finding the Perfect Delta for 2.0 THz', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

if __name__ == "__main__":
    run_delta_test()