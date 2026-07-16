# AGENTS.md

## Cursor Cloud specific instructions

### What this is
A single self-contained **Flask** web app, "Advanced Bragg Grating Studio", for simulating/designing THz QCL Bragg gratings. Physics core is `bragg_grating_tmm.py` (Transfer Matrix Method) with optimization routines in `optimize_duty_cycle.py` / `optimize_n.py`. There is no database or other external service — the only process to run is the Flask app.

### Run the app (development)
- Start with `python3 app.py`. It serves on `http://127.0.0.1:5000` with `debug=True` (auto-reload on file changes).
- Routes: `/` (Direct Simulation), `/design` (Filter Design), `/optimize` (legacy, "abandoned for now"). API endpoints: `/api/simulate`, `/api/sweep_duty`, `/api/sweep_n`, `/api/optimize` (POST JSON).
- The frontend loads MathJax and Chart.js from a CDN, so equations/charts need outbound internet to render; the backend API works without it.
- The simulation form does not pre-fill defaults — all base physical parameters must be entered before "Run Simulation" succeeds (otherwise a validation banner appears).

### Test
- `python3 -m pytest testfiles/` (or `python3 testfiles/test_integration.py`). Run from repo root; test files `sys.path.append` the parent so root modules import correctly.
- Known pre-existing failure (not an environment issue): `testfiles/test_integration.py::test_integration` "Check 3" calls `optimize_duty_cycle(..., d_trans=, sigma_Le=, n_trials=)`, but the current `optimize_duty_cycle` signature does not accept those kwargs, so it raises `TypeError`. Checks 1 and 2 pass. `test_3_devices_Iman_paper.py` and `test_target_freq_optimization.py` are matplotlib plotting scripts meant to be run directly.

### Lint / build
- No linter and no build step are configured (pure Python, no bundling).
