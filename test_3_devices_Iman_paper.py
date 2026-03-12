import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from bragg_grating_tmm import bragg_grating_tmm

def run_tests():
    # example frequency array: 1 THz to 4 THz, 1000 points
    f = np.linspace(1, 4, 1000) * 1e12
    
    print("Running simulations... grab a quick cuppa!")

    # Test 1 - Device 1 
    Lm1 = (12.93 - 3.11) * 1e-6           
    Le1 = 3.11e-6                
    nmm1 = 3.67                
    nee1 = 3.07                 
    am1 = 7.5 * 100                 
    ae1 = 38.12 * 100               
    N1 = 14                      
    Lc1 = 2e-3                  
    pishift1 = True                
    Lpi1 = 2 * Le1      
    delta1 = 230e-6

    ref1, trans1, stopband1 = bragg_grating_tmm(f, Lm1, Le1, nmm1, nee1, am1, ae1, N1, Lc1, pishift1, Lpi1, delta1)

    plt.figure(1, figsize=(10, 6))
    plt.plot(f / 1e12, trans1, linewidth=4)
    plt.xlabel('f [THz]', fontsize=14)
    plt.ylabel('Transmitted power [a.u]', fontsize=14)
    plt.title('Test Device 1 with 2Le pi element ($\delta$ = 230 $\mu$m)', fontsize=16)
    plt.grid(True)
    
    # Find the minimums (valleys) by looking for peaks in the negative data
    valleys1, _ = find_peaks(-trans1)
    plt.plot(f[valleys1] / 1e12, trans1[valleys1], '*r', markersize=12)


    # Test 2 - Device 2 
    Lm2 = (12.52 - 3.44) * 1e-6           
    Le2 = 3.44e-6                
    N2 = 14                      
    Lc2 = 1.76e-3                                 
    Lpi2 = 2 * Le2     
    delta2 = 0.0  # Dead center

    ref2, trans2, stopband2 = bragg_grating_tmm(f, Lm2, Le2, nmm1, nee1, am1, ae1, N2, Lc2, pishift1, Lpi2, delta2)

    plt.figure(2, figsize=(10, 6))
    plt.plot(f / 1e12, trans2, linewidth=4)
    plt.xlabel('f [THz]', fontsize=14)
    plt.ylabel('Transmitted power [a.u]', fontsize=14)
    plt.title('Test Device 2 with 2Le pi element ($\delta$ = 0)', fontsize=16)
    plt.grid(True)
    
    valleys2, _ = find_peaks(-trans2)
    plt.plot(f[valleys2] / 1e12, trans2[valleys2], '*r', markersize=12)


    # Test 3 - Device 3 
    Lm3 = (11.99 - 1.79) * 1e-6           
    Le3 = 1.79e-6                
    N3 = 14                      
    Lc3 = 2.7e-3                               
    Lpi3 = 2 * Le3     
    delta3 = 70e-6

    ref3, trans3, stopband3 = bragg_grating_tmm(f, Lm3, Le3, nmm1, nee1, am1, ae1, N3, Lc3, pishift1, Lpi3, delta3)

    plt.figure(3, figsize=(10, 6))
    plt.plot(f / 1e12, trans3, linewidth=4)
    plt.xlabel('f [THz]', fontsize=14)
    plt.ylabel('Transmitted power [a.u]', fontsize=14)
    plt.title('Test Device 3 with 2Le pi element ($\delta$ = 70 $\mu$m)', fontsize=16)
    plt.grid(True)
    
    valleys3, _ = find_peaks(-trans3)
    plt.plot(f[valleys3] / 1e12, trans3[valleys3], '*r', markersize=12)

    plt.show()

if __name__ == "__main__":
    run_tests()