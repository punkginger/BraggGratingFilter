"""
example: 
    f: 2 THz
    Lm: 50 different metal lengths between 8 um and 12 um
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engineer_bragg_grating import engineer_bragg_grating

def run_optimizer_test():
    print("finding the best parameter for target frequency")
    f_target = 2.0e12  
    Le = 3.11e-6                
    nmm = 3.67                
    nee = 3.07                 
    am = 7.5 * 100                 
    ae = 38.12 * 100               
    N = 14                      
    Lc = 2e-3                  
    pishift = True                
    Lpi = 2 * Le      
    delta = 0.0
    Lm_sweep = np.linspace(8e-6, 15e-6, 50)

    delta_f_results, optimal_Lm = engineer_bragg_grating(
        f_target, Lm_sweep, Le, nmm, nee, am, ae, N, Lc, pishift, Lpi, delta
    )

    print(f"Optimization Complete")
    print(f"Target Frequency: {f_target / 1e12} THz")
    print(f"Absolute Best Metal Length (Lm): {optimal_Lm * 1e6:.3f} um")

    
    plt.figure(figsize=(10, 6))
    plt.plot(Lm_sweep * 1e6, delta_f_results / 1e12, 'b-', linewidth=3, label='Frequency Error')
    min_idx = np.argmin(delta_f_results)
    plt.plot(optimal_Lm * 1e6, delta_f_results[min_idx] / 1e12, '*r', markersize=15, label='Optimal Length')
    
    plt.xlabel('Metal Length $L_m$ [$\mu$m]', fontsize=14)
    plt.ylabel('Distance from Target Frequency [THz]', fontsize=14)
    plt.title('Finding the Perfect Metal Length for 2.0 THz', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

if __name__ == "__main__":
    run_optimizer_test()