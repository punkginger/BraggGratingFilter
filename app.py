from flask import Flask, render_template, request, jsonify
import numpy as np
from engineer_bragg_grating import engineer_bragg_grating
from bragg_grating_tmm import bragg_grating_tmm
from optimize_duty_cycle import optimize_duty_cycle
from optimize_n import optimize_n

app = Flask(__name__)

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
        

        return jsonify({
            "status": "success",
            "data": {
                "frequencies": (f / 1e12).tolist(), 
                "transmission": trans.tolist()
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