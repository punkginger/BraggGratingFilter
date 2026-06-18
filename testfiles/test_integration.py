import os
import sys

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bragg_grating_tmm import bragg_grating_tmm
from optimize_duty_cycle import optimize_duty_cycle

def test_integration():
    f_target = 2.0e12
    f_range = np.linspace(f_target - 0.1e12, f_target + 0.1e12, 100)
    
    Lm, Le, nmm, nee, am, ae, N, Lc = 12.5e-6, 12.5e-6, 3.5, 3.2, 100, 50, 50, 2e-3
    
    # Test 1: bragg_grating_tmm ideal vs default non-ideal
    ref1, trans1, sb1 = bragg_grating_tmm(f_range, Lm, Le, nmm, nee, am, ae, N, Lc, True, 25e-6, 0)
    ref2, trans2, sb2 = bragg_grating_tmm(f_range, Lm, Le, nmm, nee, am, ae, N, Lc, True, 25e-6, 0, 
                                          N_sub=0, d_trans=0, sigma_Le=0)
    
    assert np.allclose(trans1, trans2), "Ideal and default non-ideal should be identical"
    print("Check 1: bragg_grating_tmm ideal/non-ideal consistency PASSED")
    
    # Test 2: optimize_duty_cycle basic run
    duty_cycles, Lm_array, Le_array, peak_trans, q_factors, rejs, sb_widths, opt_D, opt_Lm, opt_Le, best_idx = \
        optimize_duty_cycle(f_target, nmm, nee, am, ae, N, Lc, (0.5, 0.7), 0, 0.5, 0.25, 0.25)
    
    assert len(duty_cycles) == 100
    assert 0.5 <= opt_D <= 0.7
    print(f"Check 2: optimize_duty_cycle run PASSED (Optimal D: {opt_D:.4f})")
    
    # Test 3: Robustness logic
    _, _, _, peak_trans_rob, _, _, _, _, _, _, _ = \
        optimize_duty_cycle(f_target, nmm, nee, am, ae, N, Lc, (0.5, 0.7), 0, 0.5, 0.25, 0.25,
                            d_trans=1e-7, sigma_Le=1e-7, n_trials=2)
    
    assert not np.array_equal(peak_trans, peak_trans_rob), "Robust results should differ from ideal"
    print("Check 3: Robustness optimization logic produces different results PASSED")

if __name__ == "__main__":
    test_integration()
