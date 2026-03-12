from flask import Flask, render_template, request, jsonify
import numpy as np
from engineer_bragg_grating import engineer_bragg_grating

app = Flask(__name__)

@app.route('/')
def simulate_page():
    # We pass 'active_page' so the navigation bar knows which button to highlight blue!
    return render_template('simulate.html', active_page='simulate')

# Route 2: The Optimization tool
@app.route('/optimize')
def optimize_page():
    return render_template('optimize.html', active_page='optimize')


@app.route('/api/optimize', methods=['POST'])
def optimize():
    data = request.json
    
    try:
        f_target = float(data.get('f_target', 2.0e12))
        
        # To keep our first test simple, we will hardcode the static variables here in the backend
        # Later, you can add all these to your HTML form!
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
        
        # The array we want to sweep (Metal Lengths from 8um to 12um)
        Lm_sweep = np.linspace(8e-6, 12e-6, 50)
        
        # Call the engine!
        delta_f, optimal_Lm = engineer_bragg_grating(
            f_target, Lm_sweep, Le, nmm, nee, am, ae, N, Lc, pishift, Lpi, delta
        )
        
        # Return Success Code 200 and the JSON data!
        return jsonify({
            "status": "success",
            "optimal_Lm": float(optimal_Lm),
            "delta_f": delta_f.tolist(),
            "Lm_sweep": Lm_sweep.tolist() # Sending this back so the frontend can plot it!
        }), 200

    except ValueError as e:
        # If the engine crashes, return Error Code 400 and the message
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)