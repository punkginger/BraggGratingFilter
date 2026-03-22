from flask import Flask, render_template, request, jsonify
import numpy as np
from engineer_bragg_grating import engineer_bragg_grating
from bragg_grating_tmm import bragg_grating_tmm
from optimize_duty_cycle import optimize_duty_cycle

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
        
        f = np.linspace(f_start, f_stop, points)
        
        ref, trans, stopband = bragg_grating_tmm(
            f, Lm, Le, nmm, nee, am, ae, N, Lc, pishift, Lpi, delta
        )
        

        return jsonify({
            "status": "success",
            "data": {
                "frequencies": (f / 1e12).tolist(), 
                "transmission": trans.tolist()
            }
        }), 200

    except Exception as e:
        print(f"Oh eck, a simulation error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/design', methods=['POST'])
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
        
        duty_min = float(data.get('duty_min', 0.40))
        duty_max = float(data.get('duty_max', 0.85))
        duty_cycle_range = (duty_min, duty_max)
        
        Lm_array, Le_array, peak_transmissions, optimal_D, optimal_Lm, optimal_Le = optimize_duty_cycle(
            f_target, nmm, nee, am, ae, N, Lc, duty_cycle_range, delta
        )
                
        results = {
            "Lm_array": Lm_array.tolist(),
            "optimal_Lm": f"{optimal_Lm * 1e6:.3f}",
            "optimal_Le": f"{optimal_Le * 1e6:.3f}",
            "optimal_D": f"{optimal_D:.3f}",
            "peak_transmissions": peak_transmissions.tolist()
        }
        
        return jsonify({"status": "success", "data": results}), 200
        
    except Exception as e:
        print(f"Oh eck, a design error occurred: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)