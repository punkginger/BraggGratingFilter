# Filter Quality Metrics

This document explains the figure-of-merit metrics computed on the **Direct Simulation** page and shown in the scorecard and the accompanying charts. Every metric is derived directly from the simulated transmission spectrum `T(f)` returned by `bragg_grating_tmm`, and each one is computed **twice**: once for the full **non-ideal** device (interface grading + roughness) and once for the **ideal** device (`d_trans = 0`, `sigma_Le = 0`, `N_sub = 0`).

The goal of all these numbers is to answer one question: *how good is the central π-shift defect peak as a single-mode bandpass filter?*

Implementation: `compute_spectrum_metrics()` in [app.py](app.py); rendering in [templates/simulate.html](templates/simulate.html).

---

## Reference quantities derived from geometry

Before any metric is measured, two theoretical references are computed from the entered geometry (no extra input required). These follow the definitions in the project [README.md](README.md).

| Symbol     | Meaning                              | Formula                       |
| ---------- | ------------------------------------ | ----------------------------- |
| `sigma`    | Duty cycle                           | `Lm / (Lm + Le)`              |
| `n_eff`    | Effective refractive index           | `sigma*nmm + (1 - sigma)*nee` |
| `Lambda`   | Grating period                       | `Lm + Le`                     |
| `f_br`     | Theoretical Bragg (target) frequency | `c / (2 * n_eff * Lambda)`    |
| `sb_width` | Theoretical stopband width           | `f_br * (2*                   |

`f_br` is used as the design target for detuning and as the centre of the rejection "fence"; `sb_width` sets the width of that fence.

---

## The defect peak

Everything keys off one point on the spectrum: the **defect peak** (the passband spike inside the stopband).

- All local maxima are found with `scipy.signal.find_peaks(T, prominence=0.001)`.
- The defect peak is the one **closest in frequency to `f_br`**.
- `T_peak` is its transmission value; `f_peak` is its frequency.

If no peak is found, or the "peak" sits on the very edge of the swept range, the metrics are marked invalid and the UI shows a "no defect peak found" message instead of misleading numbers.

---

## 1. Side-Mode Suppression Ratio (SMSR)

**What it is:** the contrast, in decibels, between the main defect peak and the next-strongest competing transmission peak (the Fabry-Perot side modes / stopband-edge lobes). It is *the* standard metric for proving single-mode laser operation, which is the entire purpose of this filter.

**Formula:**

```
SMSR = 10 * log10(T_peak / T_side)
```

where `T_side` is the height of the tallest peak that is **not** the defect peak.

**How to read it:**

- Higher is better.
- A device is generally accepted as **single-mode when SMSR >= 30 dB**. The scorecard shows a green **SINGLE-MODE** badge above this threshold and a red **MULTI-MODE** badge below it.
- If SMSR is near 0 dB (or negative), a side mode is as strong as (or stronger than) the intended peak — the filter is not selecting a single mode.

**Where to see it:** scorecard card + the dB spectrum chart (the vertical gap between the main peak and the next tallest peak *is* the SMSR).

---

## 2. Insertion Loss (IL)

**What it is:** how much optical power is lost passing through the filter at the passband peak — the price paid for material absorption inside the grating.

**Formula:**

```
IL = -10 * log10(T_peak)
```

**How to read it:**

- Lower is better. `IL = 0 dB` would mean perfect (lossless) transmission.
- This is just the peak transmission expressed in the conventional engineering unit; a peak transmission of `0.5` corresponds to about `3 dB` of loss.

---

## 3. Stopband Rejection

**What it is:** the contrast between the bright defect peak and the dark noise floor of the stopband — i.e. how deep the "off" region of the filter is.

**The problem it solves:** the outer edges of a finite, lossy stopband are distorted by Fabry-Perot ripples. Blindly hunting for the minimum transmission can accidentally measure a ripple instead of the true floor. To avoid this, a **theoretical "fence"** is placed around the target frequency:

```
f_left  = f_br - sb_width / 2
f_right = f_br + sb_width / 2
```

The noise floor `T_floor` is the minimum transmission **strictly inside** that fence, guaranteeing we measure the true stopband darkness.

**Formula:**

```
Rejection (dB) = 10 * log10(T_peak / T_floor)
```

**How to read it:** higher is better; it measures how far the peak stands above the surrounding suppressed background.

---

## 4. 3 dB Bandwidth (FWHM) and Quality Factor (Q)

**What they are:** two complementary descriptions of how *sharp* the passband is.

- **FWHM** (Full Width at Half Maximum): the frequency width of the peak measured at half its height, reported in **GHz**. It is the concrete linewidth of the filter. Computed with `scipy.signal.peak_widths(..., rel_height=0.5)`, then multiplied by the frequency step.
- **Quality Factor (Q):** the dimensionless sharpness relative to the operating frequency.

**Formula:**

```
Q = f_peak / FWHM
```

**How to read them:**

- A **narrow FWHM** and a **high Q** both mean a sharper, more frequency-selective filter.
- Q is unitless and good for comparing filters at different frequencies; FWHM (GHz) is the tangible spec number.

**Where to see them:** scorecard cards + the passband close-up chart, where the red dashed line marks the half-maximum level and the green shaded band marks the FWHM width.

---

## 5. Shape Factor (skirt steepness)

**What it is:** how "box-like" the filter is — whether it drops off steeply after the passband or tails off slowly.

**Formula:**

```
Shape Factor = FW_20dB / FW_3dB
```

where `FW_3dB` is the FWHM and `FW_20dB` is the width measured much lower down, at 1% of the peak height (i.e. 20 dB below the peak). Both widths are found by walking outwards from the peak and linearly interpolating where the curve crosses each level (`_interp_crossing` in [app.py](app.py)).

**How to read it:**

- The ideal is **close to 1** — the peak is nearly as narrow far down as it is at the top (steep, rectangular skirts).
- A **large** value means the peak is narrow at the top but flares out at the base (shallow skirts, poorer selectivity).
- Shows `--` when the curve never drops to the 20 dB level within the swept range (widen the range to measure it).

---

## 6. Centre-Frequency Detuning

**What it is:** how far the actual defect peak lands from the intended design frequency — a check on design accuracy.

**Formula:**

```
Detuning (GHz) = |f_peak - f_br|
```

**How to read it:** smaller is better. A large detuning means the fabricated/entered geometry resonates away from the target `f_br`.

---

## 7. Peak Transmission and Peak Frequency

Reported directly for reference:

- **Peak Transmission** `T_peak` in arbitrary units (0 to 1) — the raw height of the defect mode.
- **Peak Frequency** `f_peak` in THz — where the defect mode actually sits.

---

## 8. Non-Ideal Degradation

**What it is:** a direct measurement of how much the non-ideal effects (interface grading and etch-length roughness) cost you, by comparing the non-ideal device against the ideal one.

**Formula (per metric):**

```
drop_% = (ideal_value - nonideal_value) / |ideal_value| * 100
```

computed for Peak Transmission, Q, SMSR, and Rejection.

**Where to see it:** the **Ideal vs Non-Ideal** bar chart. To keep mixed units (a.u., Q, dB) readable on one axis, each non-ideal metric is plotted as a **percentage of its ideal value** (ideal is always the 100% reference bar). The tooltip shows the raw values behind each bar.

---

## Chart summary

| Chart                          | Purpose                                                                                                              |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| Linear transmission spectrum   | Original view: ideal vs non-ideal `T(f)` on a linear axis.                                                           |
| dB (log) transmission spectrum | Reveals stopband depth and side modes invisible on the linear plot; SMSR and rejection are visible as vertical gaps. |
| Figure-of-merit scorecard      | All scalar metrics at a glance, with the single-mode SMSR check highlighted.                                         |
| Passband close-up (FWHM)       | Zoom on the defect peak with half-max line and shaded FWHM band.                                                     |
| Ideal vs Non-Ideal bars        | Degradation of Peak / Q / SMSR / Rejection due to non-ideal effects.                                                 |

---

## Caveats

- **SMSR, rejection, shape factor and detuning depend on the swept frequency range.** If the range is too narrow to contain the side modes or the full peak skirts, those metrics may be absent (`--`). Widening the sweep around `f_br` resolves this.
- The dB spectrum is floored at **-60 dB** to avoid `-inf` where transmission underflows to zero.
- Metrics are only meaningful for a device that actually produces a defect peak — typically with the **central π-shift enabled**.
