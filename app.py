from flask import Flask, render_template, request, jsonify
import numpy as np
from engineer_bragg_grating import engineer_bragg_grating
from bragg_grating_tmm import bragg_grating_tmm

app = Flask(__name__)

@app.route('/')
def simulate_page():
    return render_template('simulate.html', active_page='simulate')

@app.route('/optimize')
def optimize_page():
    return render_template('optimize.html', active_page='optimize')


@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    data = request.json
    
    try:
        # 1. Build the high-resolution frequency array
        f = np.linspace(data['f_start'], data['f_stop'], data['points'])
        
        # 2. Call your raw physics engine
        ref, trans, stopband = bragg_grating_tmm(
            f, 
            data['Lm'], data['Le'], data['nmm'], data['nee'], 
            data['am'], data['ae'], int(data['N']), data['Lc'], 
            data['pishift'], data['Lpi'], data['delta']
        )
        
        # 3. Package the numpy arrays back into standard lists for JSON
        return jsonify({
            "status": "success",
            "frequencies": (f / 1e12).tolist(), # Convert back to THz for the chart!
            "transmission": trans.tolist()
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/optimize', methods=['POST'])
def run_optimization():
    data = request.json
    
    try:
        # 1. Build the sweep array based on the user's start/stop limits
        sweep_array = np.linspace(data['sweep_start'], data['sweep_stop'], data['sweep_points'])
        
        # 2. Unpack all the static variables
        static = data['static_params']
        
        # 3. We create a clever dictionary to map everything perfectly. 
        # We fill it with static values first...
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
        
        # ...and then we overwrite exactly ONE parameter with our sweep array!
        sweep_param_name = data['sweep_param']
        params[sweep_param_name] = sweep_array
        
        # 4. Call your brilliant service layer
        # (Notice pishift is hardcoded to True here, as requested by the frontend logic)
        error_array, optimal_value = engineer_bragg_grating(
            data['f_target'],
            params['Lm'], params['Le'], params['nmm'], params['nee'],
            params['am'], params['ae'], params['N'], params['Lc'],
            True, params['Lpi'], data['delta']
        )
        
        # 5. Send the glorious results back to the browser
        return jsonify({
            "status": "success",
            "optimal_value": float(optimal_value),
            "sweep_array": sweep_array.tolist(),
            "error_array": error_array.tolist()
        }), 200

    except Exception as e:
        # We catch any ValueError thrown by the engineer_bragg_grating engine!
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)