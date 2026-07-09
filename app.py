from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.signal import find_peaks, peak_widths
from engineer_bragg_grating import engineer_bragg_grating
from bragg_grating_tmm import bragg_grating_tmm
from optimize_duty_cycle import optimize_duty_cycle
from optimize_n import optimize_n

app = Flask(__name__)

C_LIGHT = 299792458.0


def compute_spectrum_metrics(f, T, f_br, sb_width):
    """
    Compute figure-of-merit metrics for a transmission spectrum T(f).

    f: frequency array (Hz)
    T: transmission power array (same length as f)
    f_br: theoretical Bragg/target frequency (Hz), used for detuning + fence
    sb_width: theoretical stopband width (Hz), used for the rejection fence

    Returns a dict of metrics, or {"valid": False} if no usable defect peak
    exists inside the provided frequency window.
    """
    f = np.asarray(f, dtype=float)
    T = np.asarray(T, dtype=float)

    prominence_min = max(0.001, 0.02 * float(np.max(T)))
    peaks, _ = find_peaks(T, prominence=prominence_min)
    if len(peaks) == 0:
        return {"valid": False}

    # Defect peak = peak closest to the theoretical Bragg frequency
    peak_idx = int(peaks[np.argmin(np.abs(f[peaks] - f_br))])
    peak_trans = float(T[peak_idx])
    peak_freq = float(f[peak_idx])

    if peak_idx == 0 or peak_idx == len(T) - 1 or peak_trans <= 0.0:
        return {"valid": False}

    # --- FWHM + Q via scipy peak_widths (half maximum) ---
    widths, _, _, _ = peak_widths(T, [peak_idx], rel_height=0.5)
    step_freq = f[1] - f[0]
    fwhm_hz = float(widths[0] * step_freq)
    q_factor = peak_freq / fwhm_hz if fwhm_hz > 0 else 0.0

    # --- Insertion loss ---
    insertion_loss_db = float(-10.0 * np.log10(peak_trans)) if peak_trans > 0 else None

    # --- Stopband rejection using the theoretical "fence" around f_br ---
    left_edge = f_br - sb_width / 2.0
    right_edge = f_br + sb_width / 2.0
    left_idx = int(np.argmin(np.abs(f - left_edge)))
    right_idx = int(np.argmin(np.abs(f - right_edge)))
    if right_idx < left_idx:
        left_idx, right_idx = right_idx, left_idx
    stopband_region = T[left_idx:right_idx + 1]
    if len(stopband_region) > 0:
        t_floor = float(np.min(stopband_region))
    else:
        t_floor = float(np.min(T))
    rejection_ratio = peak_trans / t_floor if t_floor > 0 else 0.0
    rejection_db = float(10.0 * np.log10(rejection_ratio)) if rejection_ratio > 0 else None

    # --- Side-Mode Suppression Ratio (SMSR) ---
    # Tallest competing mode outside a guard band around the defect peak.
    # The guard excludes the passband lobe so stopband-edge / Fabry-Perot
    # side modes are measured, not ripples on the main spike itself.
    guard_hz = max(2.0 * fwhm_hz, 0.05 * sb_width if sb_width > 0 else step_freq * 20)
    in_guard = np.abs(f - peak_freq) <= guard_hz

    side_trans = None
    side_freq = None
    smsr_db = None
    side_candidates = [p for p in peaks if p != peak_idx and not in_guard[p]]
    if side_candidates:
        side_idx = int(side_candidates[np.argmax(T[side_candidates])])
        side_trans = float(T[side_idx])
        side_freq = float(f[side_idx])
    else:
        outside = ~in_guard
        if np.any(outside):
            side_idx = int(np.argmax(np.where(outside, T, -1.0)))
            side_trans = float(T[side_idx])
            side_freq = float(f[side_idx])

    if side_trans is not None and side_trans > 0:
        smsr_db = float(10.0 * np.log10(peak_trans / side_trans))

    # --- Center-frequency detuning ---
    detuning_ghz = float(abs(peak_freq - f_br) / 1e9)

    return {
        "valid": True,
        "peak_trans": peak_trans,
        "peak_freq_thz": peak_freq / 1e12,
        "fwhm_ghz": fwhm_hz / 1e9,
        "q_factor": q_factor,
        "insertion_loss_db": insertion_loss_db,
        "t_floor": t_floor,
        "rejection_ratio": rejection_ratio,
        "rejection_db": rejection_db,
        "smsr_db": smsr_db,
        "side_freq_thz": (side_freq / 1e12) if side_freq is not None else None,
        "side_trans": side_trans,
        "detuning_ghz": detuning_ghz,
    }

@app.route('/')
def simulate_page():
    return render_template('simulate.html', active_page='simulate')

@app.route('/design')
def design_page():
    return render_template('design.html', active_page='design')

@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    try:
        data = request.json or {}
        f_start = float(data.get('f_start', 0))
        f_stop = float(data.get('f_stop', 0))
        points = int(data.get('points', 1000))
        Lm = float(data.get('Lm', 0))
        Le = float(data.get('Le', 0))
        nmm = float(data.get('nmm', 0))
        nee = float(data.get('nee', 0))
        am = float(data.get('am', 0))
        ae = float(data.get('ae', 0))
        N = int(data.get('N', 0))
        Lc = float(data.get('Lc', 0))
        pishift = bool(data.get('pishift', False))
        Lpi = float(data.get('Lpi', 0))
        delta = float(data.get('delta', 0))
        N_sub = int(data.get('N_sub', 10))
        d_trans = float(data.get('d_trans', 0.0))
        sigma_Le = float(data.get('sigma_Le', 0.0))
        seed = data.get('seed')
        if seed is not None and seed != "":
            seed = int(seed)
        else:
            seed = None
        
        f = np.linspace(f_start, f_stop, points)
        
        ref, trans, stopband = bragg_grating_tmm(
            f, Lm, Le, nmm, nee, am, ae, N, Lc, pishift, Lpi, delta,
            N_sub=N_sub, d_trans=d_trans, sigma_Le=sigma_Le, seed=seed
        )
        ref_ideal, trans_ideal, stopband_ideal = bragg_grating_tmm(
            f, Lm, Le, nmm, nee, am, ae, N, Lc, pishift, Lpi, delta,
            N_sub=0, d_trans=0.0, sigma_Le=0.0, seed=None
        )

        # Derive the theoretical Bragg frequency and stopband width from the
        # entered geometry (README section "parameter optimization").
        Lambda = Lm + Le
        if Lambda > 0:
            sigma = Lm / Lambda
        else:
            sigma = 0.0
        n_eff = sigma * nmm + (1.0 - sigma) * nee
        if n_eff > 0 and Lambda > 0:
            f_br = C_LIGHT / (2.0 * n_eff * Lambda)
            sb_width = f_br * (2.0 * abs(nmm - nee) / (np.pi * n_eff)) * np.sin(np.pi * sigma)
        else:
            f_br = float(np.mean(f))
            sb_width = 0.0

        metrics_nonideal = compute_spectrum_metrics(f, trans, f_br, sb_width)
        metrics_ideal = compute_spectrum_metrics(f, trans_ideal, f_br, sb_width)

        # Percentage degradation between ideal and non-ideal figures of merit.
        def _pct_drop(ideal_val, real_val):
            if ideal_val is None or real_val is None or ideal_val == 0:
                return None
            return float((ideal_val - real_val) / abs(ideal_val) * 100.0)

        degradation = {}
        if metrics_nonideal.get("valid") and metrics_ideal.get("valid"):
            degradation = {
                "peak_pct": _pct_drop(metrics_ideal["peak_trans"], metrics_nonideal["peak_trans"]),
                "q_pct": _pct_drop(metrics_ideal["q_factor"], metrics_nonideal["q_factor"]),
                "smsr_pct": _pct_drop(metrics_ideal["smsr_db"], metrics_nonideal["smsr_db"]),
                "rejection_pct": _pct_drop(metrics_ideal["rejection_db"], metrics_nonideal["rejection_db"]),
            }

        # dB spectra (floored to avoid -inf where transmission underflows).
        DB_FLOOR = -60.0
        trans_db = np.where(trans > 0, 10.0 * np.log10(np.clip(trans, 1e-12, None)), DB_FLOOR)
        trans_db = np.clip(trans_db, DB_FLOOR, None)
        trans_ideal_db = np.where(trans_ideal > 0, 10.0 * np.log10(np.clip(trans_ideal, 1e-12, None)), DB_FLOOR)
        trans_ideal_db = np.clip(trans_ideal_db, DB_FLOOR, None)

        return jsonify({
            "status": "success",
            "data": {
                "frequencies": (f / 1e12).tolist(),
                "transmission": trans.tolist(),
                "transmission_ideal": trans_ideal.tolist(),
                "transmission_db": trans_db.tolist(),
                "transmission_ideal_db": trans_ideal_db.tolist(),
                "metrics": {
                    "nonideal": metrics_nonideal,
                    "ideal": metrics_ideal,
                    "degradation": degradation,
                    "f_br_thz": f_br / 1e12,
                    "sb_width_ghz": sb_width / 1e9,
                    "fence_left_thz": (f_br - sb_width / 2.0) / 1e12,
                    "fence_right_thz": (f_br + sb_width / 2.0) / 1e12,
                }
            }
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/sweep_duty', methods=['POST'])
def run_design():
    try:
        data = request.json or {}
        f_target = float(data.get('freq', 0))
        delta = float(data.get('delta', 0))
        nmm = float(data.get('nmm', 0))
        nee = float(data.get('nee', 0))
        am = float(data.get('am', 0))
        ae = float(data.get('ae', 0))
        N = int(data.get('N', 0))
        Lc = float(data.get('Lc', 0))
        weight_trans = float(data.get('weight_trans', 0))
        weight_q = float(data.get('weight_q', 0))
        weight_rej = float(data.get('weight_rej', 0))
        
        duty_min = float(data.get('duty_min', 0.40))
        duty_max = float(data.get('duty_max', 0.85))
        duty_cycle_range = (duty_min, duty_max)
        
        duty_cycles, Lm_array, Le_array, peak_transmissions, q_factors, rejections, sb_widths, optimal_D, optimal_Lm, optimal_Le, best_idx = optimize_duty_cycle(
            f_target, nmm, nee, am, ae, N, Lc, duty_cycle_range, delta, weight_trans, weight_q, weight_rej
        )
        
        optimal_q = q_factors[best_idx]
        optimal_rej = rejections[best_idx]
        optimal_sb_width = sb_widths[best_idx]

        results = {
            "duty_cycles": duty_cycles.tolist(),
            "Lm_array": Lm_array.tolist(),
            "Le_array": Le_array.tolist(),
            "peak_transmissions": peak_transmissions.tolist(),
            "q_factors": q_factors.tolist(),
            "rejections": rejections.tolist(),
            "sb_widths": sb_widths.tolist(),
                
            "optimal_D": float(optimal_D),
            "optimal_Lm": float(optimal_Lm),
            "optimal_Le": float(optimal_Le),
            "optimal_q": float(optimal_q),
            "optimal_rej": float(optimal_rej),
            "optimal_sb_width": float(optimal_sb_width),
            "best_idx": int(best_idx)
        }
        
        return jsonify({"status": "success", "data": results}), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/sweep_n', methods=['POST'])
def sweep_n_endpoint():
    try:
        data = request.json or {}
        
        f_target = float(data.get('freq', 0))
        delta = float(data.get('delta', 0))
        nmm = float(data.get('nmm', 0))
        nee = float(data.get('nee', 0))
        am = float(data.get('am', 0))
        ae = float(data.get('ae', 0))
        Lc = float(data.get('Lc', 0))
        
        fixed_duty = float(data.get('duty_cycle', 0))
        n_min = int(data.get('n_min', 10))
        n_max = int(data.get('n_max', 40))
        
        weight_trans = float(data.get('weight_trans', 0))
        weight_q = float(data.get('weight_q', 0))
        weight_rej = float(data.get('weight_rej', 0))

        n_array, p_trans, q_facs, rejs, sb_width, opt_N, best_idx = optimize_n(
            f_target, nmm, nee, am, ae, fixed_duty, n_min, n_max, Lc, delta, weight_trans, weight_q, weight_rej
        )
        
        optimal_q = q_facs[best_idx]
        optimal_rej = rejs[best_idx]

        results = {
            "n_array": n_array.tolist(),
            "peak_transmissions": p_trans.tolist(),
            "q_factors": q_facs.tolist(),
            "rejections": rejs.tolist(),
            
            "optimal_N": int(opt_N),
            "optimal_q": float(optimal_q),
            "optimal_rej": float(optimal_rej),
            "optimal_sb_width": float(sb_width), 
            "best_idx": int(best_idx)
        }

        return jsonify({"status": "success", "data": results}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# optimize function directly from original matlab code, abandoned for now.
@app.route('/optimize')
def optimize_page():
    return render_template('optimize.html', active_page='optimize')


@app.route('/api/optimize', methods=['POST'])
def run_optimization():
    data = request.json
    
    try:
        sweep_array = np.linspace(data['sweep_start'], data['sweep_stop'], data['sweep_points'])
        static = data['static_params']
        params = {
            'Lm': static.get('Lm'),
            'Le': static.get('Le'),
            'nmm': static.get('nmm'),
            'nee': static.get('nee'),
            'am': static.get('am'),
            'ae': static.get('ae'),
            'N': static.get('N'),
            'Lc': static.get('Lc'),
            'Lpi': static.get('Lpi')
        }
        

        sweep_param_name = data['sweep_param']
        params[sweep_param_name] = sweep_array
        
        error_array, optimal_value = engineer_bragg_grating(
            data['f_target'],
            params['Lm'], params['Le'], params['nmm'], params['nee'],
            params['am'], params['ae'], params['N'], params['Lc'],
            True, params['Lpi'], data['delta']
        )
        
        return jsonify({
            "status": "success",
            "optimal_value": float(optimal_value),
            "sweep_array": sweep_array.tolist(),
            "error_array": error_array.tolist()
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)