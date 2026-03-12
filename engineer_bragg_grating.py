import numpy as np
from scipy.signal import find_peaks
from bragg_grating_tmm import bragg_grating_tmm

def engineer_bragg_grating(f_target, Lm, Le, nmm, nee, am, ae, N, Lc, pishift, Lpi, delta):
    
    # 1. Force everything to be a 1D array so we can measure their lengths safely
    Lm = np.atleast_1d(Lm)
    Le = np.atleast_1d(Le)
    nmm = np.atleast_1d(nmm)
    nee = np.atleast_1d(nee)
    am = np.atleast_1d(am)
    ae = np.atleast_1d(ae)
    N = np.atleast_1d(N)
    Lc = np.atleast_1d(Lc)
    Lpi = np.atleast_1d(Lpi)

    # 2. Check exactly which parameters are arrays
    sweepLm = len(Lm) > 1
    sweepLe = len(Le) > 1
    sweepnmm = len(nmm) > 1
    sweepnee = len(nee) > 1
    sweepam = len(am) > 1
    sweepae = len(ae) > 1
    sweepN = len(N) > 1
    sweepLc = len(Lc) > 1
    sweepLpi = len(Lpi) > 1

    total_sweeps = sum([sweepLm, sweepLe, sweepnmm, sweepnee, sweepam, sweepae, sweepN, sweepLc, sweepLpi])
    if total_sweeps > 1:
        raise ValueError("You can only sweep one parameter at one time.")
    if total_sweeps == 0:
        raise ValueError("You must provide at least one parameter as an array to perform the sweep.")

    f = np.linspace(f_target - 1e12, f_target + 0.5e12, 5000)
    delta_f = []
    optimal_x = 0


    def calculate_error(transmitted_power):
        valleys, _ = find_peaks(-transmitted_power)
        if len(valleys) < 2: return np.inf
        
        valley_depths = -transmitted_power[valleys]
        sorted_indices = np.argsort(valley_depths)[::-1]
        idx_start = min(valleys[sorted_indices[0]], valleys[sorted_indices[1]])
        idx_end = max(valleys[sorted_indices[0]], valleys[sorted_indices[1]])
        
        trans_between = transmitted_power[idx_start:idx_end]
        if len(trans_between) == 0: return np.inf
        
        local_peak_idx = np.argmax(trans_between)
        fsuspect = f[idx_start + local_peak_idx]
        return np.abs(fsuspect - f_target)


    if sweepLm:
        for i in range(len(Lm)):
            ref, trans, sb = bragg_grating_tmm(f, Lm[i], Le[0], nmm[0], nee[0], am[0], ae[0], int(N[0]), Lc[0], pishift, Lpi[0], delta)
            delta_f.append(calculate_error(trans))
        minloc = np.argmin(delta_f)
        optimal_x = Lm[minloc]

    elif sweepLe:
        for i in range(len(Le)):
            ref, trans, sb = bragg_grating_tmm(f, Lm[0], Le[i], nmm[0], nee[0], am[0], ae[0], int(N[0]), Lc[0], pishift, Lpi[0], delta)
            delta_f.append(calculate_error(trans))
        minloc = np.argmin(delta_f)
        optimal_x = Le[minloc]

    elif sweepnmm:
        for i in range(len(nmm)):
            ref, trans, sb = bragg_grating_tmm(f, Lm[0], Le[0], nmm[i], nee[0], am[0], ae[0], int(N[0]), Lc[0], pishift, Lpi[0], delta)
            delta_f.append(calculate_error(trans))
        minloc = np.argmin(delta_f)
        optimal_x = nmm[minloc]

    elif sweepnee:
        for i in range(len(nee)):
            ref, trans, sb = bragg_grating_tmm(f, Lm[0], Le[0], nmm[0], nee[i], am[0], ae[0], int(N[0]), Lc[0], pishift, Lpi[0], delta)
            delta_f.append(calculate_error(trans))
        minloc = np.argmin(delta_f)
        optimal_x = nee[minloc]

    elif sweepam:
        for i in range(len(am)):
            ref, trans, sb = bragg_grating_tmm(f, Lm[0], Le[0], nmm[0], nee[0], am[i], ae[0], int(N[0]), Lc[0], pishift, Lpi[0], delta)
            delta_f.append(calculate_error(trans))
        minloc = np.argmin(delta_f)
        optimal_x = am[minloc]

    elif sweepae:
        for i in range(len(ae)):
            ref, trans, sb = bragg_grating_tmm(f, Lm[0], Le[0], nmm[0], nee[0], am[0], ae[i], int(N[0]), Lc[0], pishift, Lpi[0], delta)
            delta_f.append(calculate_error(trans))
        minloc = np.argmin(delta_f)
        optimal_x = ae[minloc]

    elif sweepN:
        for i in range(len(N)):
            ref, trans, sb = bragg_grating_tmm(f, Lm[0], Le[0], nmm[0], nee[0], am[0], ae[0], int(N[i]), Lc[0], pishift, Lpi[0], delta)
            delta_f.append(calculate_error(trans))
        minloc = np.argmin(delta_f)
        optimal_x = N[minloc]

    elif sweepLc:
        for i in range(len(Lc)):
            ref, trans, sb = bragg_grating_tmm(f, Lm[0], Le[0], nmm[0], nee[0], am[0], ae[0], int(N[0]), Lc[i], pishift, Lpi[0], delta)
            delta_f.append(calculate_error(trans))
        minloc = np.argmin(delta_f)
        optimal_x = Lc[minloc]

    elif sweepLpi:
        for i in range(len(Lpi)):
            ref, trans, sb = bragg_grating_tmm(f, Lm[0], Le[0], nmm[0], nee[0], am[0], ae[0], int(N[0]), Lc[0], pishift, Lpi[i], delta)
            delta_f.append(calculate_error(trans))
        minloc = np.argmin(delta_f)
        optimal_x = Lpi[minloc]


    return np.array(delta_f), optimal_x