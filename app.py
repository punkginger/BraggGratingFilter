from flask import Flask, render_template, request, jsonify
import numpy as np
from engineer_bragg_grating import engineer_bragg_grating
from bragg_grating_tmm import bragg_grating_tmm
from engineer_delta import optimize_delta

app = Flask(__name__)

@app.route('/')
def simulate_page():
    return render_template('simulate.html', active_page='simulate')


@app.route('/optimize')
def optimize_page():
    return render_template('optimize.html', active_page='optimize')


@app.route('/design')
def design_page():
    return render_template('design.html', active_page='design')


@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    data = request.json
    
    try:
        f = np.linspace(data['f_start'], data['f_stop'], data['points'])
        
        ref, trans, stopband = bragg_grating_tmm(
            f, 
            data['Lm'], data['Le'], data['nmm'], data['nee'], 
            data['am'], data['ae'], int(data['N']), data['Lc'], 
            data['pishift'], data['Lpi'], data['delta']
        )
        
        return jsonify({
            "status": "success",
            "frequencies": (f / 1e12).tolist(), # Convert back to THz 
            "transmission": trans.tolist()
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


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


@app.route('/api/design', methods=['POST'])
def run_design():
    data = request.json
    
    try:
        delta_array = np.linspace(data['delta_start'], data['delta_stop'], data['points'])
        f_target = float(data['f_target'])
        Lm = float(data['Lm'])
        Le = float(data['Le'])
        Lpi = float(data['Lpi'])
        nmm = float(data['nmm'])
        nee = float(data['nee'])
        am = float(data['am'])
        ae = float(data['ae'])
        N = int(data['N'])
        Lc = float(data['Lc'])
        error_array, optimal_delta = optimize_delta(
            f_target, delta_array, Lm, Le, nmm, nee, am, ae, N, Lc, Lpi
        )
        
        return jsonify({
            "status": "success",
            "optimal_delta": float(optimal_delta),
            "delta_array": delta_array.tolist(),
            "error_array": error_array.tolist()
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)