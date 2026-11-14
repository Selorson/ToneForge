//! `eq_sketch_fitter`: Freehand `EQ Sketch` Curve-to-Parametric-Bands Fitter for `FabFilter Pro-Q 4`.
//!
//! Converts a user-drawn freehand `(freq_hz, gain_db)` gesture into up to `24` `EqBandParams`
//! (`LowCut`, `LowShelf`, `Bell`, `HighShelf`, `HighCut`) using edge-feature classification,
//! greedy residual peak initialization, and damped Levenberg-Marquardt nonlinear least-squares
//! optimization directly against `FractionalSlopeBiquadBand`'s exact complex transfer function.

use crate::biquad_frac_slope::FractionalSlopeBiquadBand;
use crate::params_601::{EqBandParams, FilterShape, PROQ4_NUM_BANDS};
use num_complex::Complex64;

const SKETCH_GRID_BINS: usize = 128;
const FIT_SAMPLE_RATE: f64 = 48_000.0;
const ALLOWED_CUT_SLOPES: [f32; 7] = [6.0, 12.0, 18.0, 24.0, 36.0, 48.0, 72.0];

/// Evaluates the exact combined magnitude response in dB of a slice of `EqBandParams` at `freq_hz`.
pub fn eval_bands_combined_db(bands: &[EqBandParams], freq_hz: f64, sample_rate: f64) -> f32 {
    let mut biquad = FractionalSlopeBiquadBand::new_unity();
    let mut h_total = Complex64::new(1.0, 0.0);
    for bp in bands {
        if !bp.used || !bp.enabled {
            continue;
        }
        let gq = bp.shape.has_gain_q_interaction();
        biquad.configure_shape(
            bp.shape,
            bp.freq_hz as f64,
            bp.gain_db as f64,
            bp.q_ui as f64,
            bp.slope_db_oct as f64,
            sample_rate,
            gq,
        );
        h_total *= biquad.eval_complex_response(freq_hz, sample_rate);
    }
    (20.0 * h_total.norm().max(1e-9).log10()) as f32
}

#[inline]
fn eval_single_band_db(bp: &EqBandParams, freq_hz: f64) -> f64 {
    let mut biquad = FractionalSlopeBiquadBand::new_unity();
    biquad.configure_shape(
        bp.shape,
        bp.freq_hz as f64,
        bp.gain_db as f64,
        bp.q_ui as f64,
        bp.slope_db_oct as f64,
        FIT_SAMPLE_RATE,
        bp.shape.has_gain_q_interaction(),
    );
    20.0 * biquad
        .eval_complex_response(freq_hz, FIT_SAMPLE_RATE)
        .norm()
        .max(1e-9)
        .log10()
}

/// Fits up to `max_bands` (`1..=24`) parametric `EqBandParams` to a freehand `(freq_hz, gain_db)` curve.
pub fn fit_eq_sketch_curve(points_hz_db: &[(f32, f32)], max_bands: usize) -> Vec<EqBandParams> {
    let cap = max_bands.clamp(1, PROQ4_NUM_BANDS);
    let mut clean: Vec<(f64, f64)> = points_hz_db
        .iter()
        .filter(|&&(f, g)| f.is_finite() && g.is_finite() && f >= 10.0 && f <= 30_000.0)
        .map(|&(f, g)| ((f as f64).clamp(20.0, 20_000.0), (g as f64).clamp(-30.0, 30.0)))
        .collect();

    if clean.is_empty() {
        return Vec::new();
    }
    clean.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    clean.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-3);

    if clean.len() == 1 {
        let (f, g) = clean[0];
        if g.abs() < 0.1 {
            return Vec::new();
        }
        let mut bp = EqBandParams::default();
        bp.used = true;
        bp.enabled = true;
        bp.shape = FilterShape::Bell;
        bp.freq_hz = f as f32;
        bp.gain_db = g as f32;
        bp.q_ui = 1.0;
        bp.slope_db_oct = 12.0;
        return vec![bp];
    }

    let f_min = clean.first().unwrap().0.clamp(20.0, 18_000.0);
    let f_max = clean.last().unwrap().0.clamp((f_min * 1.25).min(20_000.0), 20_000.0);
    let log_min = f_min.log2();
    let log_max = f_max.log2();

    let mut grid_hz = [0.0f64; SKETCH_GRID_BINS];
    let mut target_db = [0.0f64; SKETCH_GRID_BINS];
    let mut src_idx = 0usize;
    for i in 0..SKETCH_GRID_BINS {
        let t = (i as f64) / ((SKETCH_GRID_BINS - 1) as f64);
        let log_f = log_min + t * (log_max - log_min);
        let f = log_f.exp2();
        grid_hz[i] = f;
        while src_idx + 2 < clean.len() && clean[src_idx + 1].0 < f {
            src_idx += 1;
        }
        let (f0, g0) = clean[src_idx];
        let (f1, g1) = clean[(src_idx + 1).min(clean.len() - 1)];
        let l0 = f0.log2();
        let l1 = f1.log2();
        let alpha = if (l1 - l0).abs() > 1e-9 {
            ((log_f - l0) / (l1 - l0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        target_db[i] = g0 + alpha * (g1 - g0);
    }

    let mut residual = target_db;
    let mut weights = [1.0f64; SKETCH_GRID_BINS];
    let mut bands: Vec<EqBandParams> = Vec::with_capacity(cap);
    let mut lowcut_fc_opt: Option<f64> = None;
    let mut highcut_fc_opt: Option<f64> = None;

    // 1. Detect LowCut or LowShelf at low-frequency boundary
    if bands.len() < cap && f_min <= 250.0 {
        let peak_mid = residual[8..SKETCH_GRID_BINS / 2]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if residual[0] <= -8.5 && (peak_mid - residual[0]) >= 7.5 && residual[0] < residual[10] - 4.0
        {
            // Find -3 dB cutoff relative to local plateau after the low-cut knee
            let plateau_db = peak_mid.min(1.5).max(-2.5);
            let target_3db = plateau_db - 3.01;
            let mut k_cut = 6usize;
            for k in 0..(SKETCH_GRID_BINS / 2) {
                if residual[k] >= target_3db {
                    k_cut = k;
                    break;
                }
            }
            let fc_init = grid_hz[k_cut].clamp(20.0, 600.0);
            let mut best_band = make_band(FilterShape::LowCut, fc_init as f32, 0.0, 1.0, 12.0);
            let mut best_err = f64::INFINITY;
            for &slope in &ALLOWED_CUT_SLOPES {
                for &q_try in &[0.707f32, 1.0, 1.25] {
                    for &f_mul in &[0.88, 0.94, 1.0, 1.06, 1.12] {
                        let cand = make_band(
                            FilterShape::LowCut,
                            (fc_init * f_mul).clamp(20.0, 800.0) as f32,
                            0.0,
                            q_try,
                            slope,
                        );
                        let mut err = 0.0;
                        for k in 0..(k_cut + 16).min(SKETCH_GRID_BINS) {
                            let w = if grid_hz[k] >= cand.freq_hz as f64 * 0.65 {
                                1.0
                            } else {
                                0.35
                            };
                            let d = residual[k] - eval_single_band_db(&cand, grid_hz[k]);
                            err += w * d * d;
                        }
                        if err < best_err {
                            best_err = err;
                            best_band = cand;
                        }
                    }
                }
            }
            lowcut_fc_opt = Some(best_band.freq_hz as f64);
            for k in 0..SKETCH_GRID_BINS {
                residual[k] -= eval_single_band_db(&best_band, grid_hz[k]);
                if grid_hz[k] < (best_band.freq_hz as f64) * 0.85 {
                    weights[k] = 0.15; // De-emphasize deep stopband below LowCut cutoff
                }
            }
            bands.push(best_band);
        } else {
            let low_avg = residual[0..6].iter().sum::<f64>() / 6.0;
            let low_spread = residual[0..6]
                .iter()
                .map(|&v| (v - low_avg).abs())
                .fold(0.0f64, f64::max);
            let mid_avg = residual[SKETCH_GRID_BINS / 3..2 * SKETCH_GRID_BINS / 3]
                .iter()
                .sum::<f64>()
                / ((SKETCH_GRID_BINS / 3) as f64);
            if low_avg.abs() >= 1.25
                && low_spread <= 1.8
                && (low_avg - mid_avg).abs() >= 1.5
                && residual[0].abs() >= 0.7 * residual[0..24].iter().map(|v| v.abs()).fold(0.0, f64::max)
            {
                let half_g = mid_avg + 0.5 * (low_avg - mid_avg);
                let mut k_shelf = 12usize;
                for k in 1..(SKETCH_GRID_BINS / 2) {
                    if (residual[k] - half_g).signum() != (residual[0] - half_g).signum() {
                        k_shelf = k;
                        break;
                    }
                }
                let shelf = make_band(
                    FilterShape::LowShelf,
                    grid_hz[k_shelf].clamp(30.0, 600.0) as f32,
                    (low_avg - mid_avg) as f32,
                    0.85,
                    12.0,
                );
                for k in 0..SKETCH_GRID_BINS {
                    residual[k] -= eval_single_band_db(&shelf, grid_hz[k]);
                }
                bands.push(shelf);
            }
        }
    }

    // 2. Detect HighCut or HighShelf at high-frequency boundary
    if bands.len() < cap && f_max >= 5_000.0 {
        let last = SKETCH_GRID_BINS - 1;
        let peak_upper_mid = residual[SKETCH_GRID_BINS / 2..last - 8]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if residual[last] <= -8.5
            && (peak_upper_mid - residual[last]) >= 7.5
            && residual[last] < residual[last - 10] - 4.0
        {
            let plateau_db = peak_upper_mid.min(1.5).max(-2.5);
            let target_3db = plateau_db - 3.01;
            let mut k_cut = last - 6;
            for k in ((SKETCH_GRID_BINS / 2)..=last).rev() {
                if residual[k] >= target_3db {
                    k_cut = k;
                    break;
                }
            }
            let fc_init = grid_hz[k_cut].clamp(1_500.0, 19_500.0);
            let mut best_band = make_band(FilterShape::HighCut, fc_init as f32, 0.0, 1.0, 12.0);
            let mut best_err = f64::INFINITY;
            for &slope in &ALLOWED_CUT_SLOPES {
                for &q_try in &[0.707f32, 1.0, 1.25] {
                    for &f_mul in &[0.88, 0.94, 1.0, 1.06, 1.12] {
                        let cand = make_band(
                            FilterShape::HighCut,
                            (fc_init * f_mul).clamp(1_000.0, 20_000.0) as f32,
                            0.0,
                            q_try,
                            slope,
                        );
                        let mut err = 0.0;
                        let start_k = k_cut.saturating_sub(16);
                        for k in start_k..SKETCH_GRID_BINS {
                            let w = if grid_hz[k] <= cand.freq_hz as f64 * 1.35 {
                                1.0
                            } else {
                                0.35
                            };
                            let d = residual[k] - eval_single_band_db(&cand, grid_hz[k]);
                            err += w * d * d;
                        }
                        if err < best_err {
                            best_err = err;
                            best_band = cand;
                        }
                    }
                }
            }
            highcut_fc_opt = Some(best_band.freq_hz as f64);
            for k in 0..SKETCH_GRID_BINS {
                residual[k] -= eval_single_band_db(&best_band, grid_hz[k]);
                if grid_hz[k] > (best_band.freq_hz as f64) * 1.15 {
                    weights[k] = 0.15;
                }
            }
            bands.push(best_band);
        } else {
            let high_avg = residual[last - 5..=last].iter().sum::<f64>() / 6.0;
            let high_max_abs = residual[3 * SKETCH_GRID_BINS / 4..=last]
                .iter()
                .map(|v| v.abs())
                .fold(0.0f64, f64::max);
            let mid_ref = residual[SKETCH_GRID_BINS / 2];
            if high_avg.abs() >= 1.0
                && (high_avg - mid_ref).abs() >= 1.2
                && residual[last].abs() >= 0.72 * high_max_abs
            {
                let half_g = mid_ref + 0.5 * (high_avg - mid_ref);
                let mut k_shelf = last - 14;
                for k in ((SKETCH_GRID_BINS / 2)..last).rev() {
                    if (residual[k] - half_g).signum() != (residual[last] - half_g).signum() {
                        k_shelf = k;
                        break;
                    }
                }
                let shelf = make_band(
                    FilterShape::HighShelf,
                    grid_hz[k_shelf].clamp(1_200.0, 16_000.0) as f32,
                    high_avg as f32,
                    0.85,
                    12.0,
                );
                for k in 0..SKETCH_GRID_BINS {
                    residual[k] -= eval_single_band_db(&shelf, grid_hz[k]);
                }
                bands.push(shelf);
            }
        }
    }

    // 3. Greedy Residual Peak Initialization for Interior Bell Bands
    let pass_lo = lowcut_fc_opt.map_or(f_min, |fc| (fc * 0.9).max(f_min));
    let pass_hi = highcut_fc_opt.map_or(f_max, |fc| (fc * 1.1).min(f_max));

    while bands.len() < cap {
        let mut best_k = None;
        let mut best_abs = 0.0f64;
        let mut weighted_sq = 0.0f64;
        let mut weight_sum = 0.0f64;
        for k in 0..SKETCH_GRID_BINS {
            let f = grid_hz[k];
            if f >= pass_lo && f <= pass_hi {
                weighted_sq += residual[k] * residual[k];
                weight_sum += 1.0;
                if residual[k].abs() > best_abs {
                    best_abs = residual[k].abs();
                    best_k = Some(k);
                }
            }
        }
        let rms = if weight_sum > 0.0 {
            (weighted_sq / weight_sum).sqrt()
        } else {
            0.0
        };
        if rms < 0.12 && best_abs < 0.30 {
            break;
        }
        let Some(pk) = best_k else {
            break;
        };
        if best_abs < 0.25 {
            break;
        }

        let peak_db = residual[pk];
        let half_target = 0.5 * peak_db;
        let mut left_k = pk;
        while left_k > 0 && residual[left_k] * peak_db.signum() > half_target.abs() {
            left_k -= 1;
        }
        let mut right_k = pk;
        while right_k + 1 < SKETCH_GRID_BINS
            && residual[right_k] * peak_db.signum() > half_target.abs()
        {
            right_k += 1;
        }
        let bw_oct = (grid_hz[right_k].log2() - grid_hz[left_k].log2()).clamp(0.15, 3.5);
        let q_est = (1.35 / bw_oct).clamp(0.25, 12.0) as f32;

        let bell = make_band(
            FilterShape::Bell,
            grid_hz[pk] as f32,
            peak_db.clamp(-30.0, 30.0) as f32,
            q_est,
            12.0,
        );
        for k in 0..SKETCH_GRID_BINS {
            residual[k] -= eval_single_band_db(&bell, grid_hz[k]);
        }
        bands.push(bell);
    }

    if bands.is_empty() {
        return bands;
    }

    // 4. Joint Damped Levenberg-Marquardt Refinement across all fitted bands
    refine_bands_levenberg_marquardt(&mut bands, &grid_hz, &target_db, &weights);

    // Sort bands ascending by frequency and assign clean sequential band defaults
    bands.sort_by(|a, b| {
        a.freq_hz
            .partial_cmp(&b.freq_hz)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    bands
}

fn make_band(
    shape: FilterShape,
    freq_hz: f32,
    gain_db: f32,
    q_ui: f32,
    slope_db_oct: f32,
) -> EqBandParams {
    let mut bp = EqBandParams::default();
    bp.used = true;
    bp.enabled = true;
    bp.shape = shape;
    bp.freq_hz = freq_hz.clamp(20.0, 20_000.0);
    bp.gain_db = gain_db.clamp(-30.0, 30.0);
    bp.q_ui = q_ui.clamp(0.1, 24.0);
    bp.slope_db_oct = slope_db_oct;
    bp
}

#[derive(Clone, Copy)]
enum ParamRef {
    LogFreq(usize),
    GainDb(usize),
    LogQ(usize),
}

fn get_param_val(bands: &[EqBandParams], p: ParamRef) -> f64 {
    match p {
        ParamRef::LogFreq(b) => (bands[b].freq_hz as f64).log2(),
        ParamRef::GainDb(b) => bands[b].gain_db as f64,
        ParamRef::LogQ(b) => (bands[b].q_ui as f64).log2(),
    }
}

fn set_param_val(bands: &mut [EqBandParams], p: ParamRef, val: f64) {
    match p {
        ParamRef::LogFreq(b) => {
            bands[b].freq_hz = val.exp2().clamp(20.0, 20_000.0) as f32;
        }
        ParamRef::GainDb(b) => {
            bands[b].gain_db = val.clamp(-30.0, 30.0) as f32;
        }
        ParamRef::LogQ(b) => {
            bands[b].q_ui = val.exp2().clamp(0.15, 20.0) as f32;
        }
    }
}

fn eval_weighted_cost(
    bands: &[EqBandParams],
    grid_hz: &[f64; SKETCH_GRID_BINS],
    target_db: &[f64; SKETCH_GRID_BINS],
    weights: &[f64; SKETCH_GRID_BINS],
    out_err: &mut [f64; SKETCH_GRID_BINS],
) -> f64 {
    let mut biquads = Vec::with_capacity(bands.len());
    for bp in bands {
        let mut bq = FractionalSlopeBiquadBand::new_unity();
        bq.configure_shape(
            bp.shape,
            bp.freq_hz as f64,
            bp.gain_db as f64,
            bp.q_ui as f64,
            bp.slope_db_oct as f64,
            FIT_SAMPLE_RATE,
            bp.shape.has_gain_q_interaction(),
        );
        biquads.push(bq);
    }

    let mut cost = 0.0f64;
    for k in 0..SKETCH_GRID_BINS {
        let mut h = Complex64::new(1.0, 0.0);
        for bq in &biquads {
            h *= bq.eval_complex_response(grid_hz[k], FIT_SAMPLE_RATE);
        }
        let pred_db = 20.0 * h.norm().max(1e-9).log10();
        let e = target_db[k] - pred_db;
        out_err[k] = e;
        cost += weights[k] * e * e;
    }
    cost
}

fn refine_bands_levenberg_marquardt(
    bands: &mut [EqBandParams],
    grid_hz: &[f64; SKETCH_GRID_BINS],
    target_db: &[f64; SKETCH_GRID_BINS],
    weights: &[f64; SKETCH_GRID_BINS],
) {
    let mut params = Vec::with_capacity(bands.len() * 3);
    for (i, bp) in bands.iter().enumerate() {
        params.push(ParamRef::LogFreq(i));
        if bp.shape.is_gain_dependent() {
            params.push(ParamRef::GainDb(i));
        }
        params.push(ParamRef::LogQ(i));
    }
    let p_len = params.len();
    if p_len == 0 {
        return;
    }

    let mut err_vec = [0.0f64; SKETCH_GRID_BINS];
    let mut cur_cost = eval_weighted_cost(bands, grid_hz, target_db, weights, &mut err_vec);
    let mut lambda = 1e-2f64;
    let eps = 1e-4f64;

    let mut jac = vec![vec![0.0f64; p_len]; SKETCH_GRID_BINS];
    let mut jtwj = vec![vec![0.0f64; p_len]; p_len];
    let mut jtwr = vec![0.0f64; p_len];
    let mut pert_err = [0.0f64; SKETCH_GRID_BINS];

    for _iter in 0..20 {
        if cur_cost < 1e-4 {
            break;
        }
        // Finite-difference Jacobian d(pred_db)/d(param) = -(pert_err - err_vec)/eps
        for (j, &pref) in params.iter().enumerate() {
            let orig = get_param_val(bands, pref);
            set_param_val(bands, pref, orig + eps);
            let _ = eval_weighted_cost(bands, grid_hz, target_db, weights, &mut pert_err);
            set_param_val(bands, pref, orig);
            for k in 0..SKETCH_GRID_BINS {
                jac[k][j] = (err_vec[k] - pert_err[k]) / eps;
            }
        }

        for r in 0..p_len {
            let mut rhs = 0.0f64;
            for k in 0..SKETCH_GRID_BINS {
                rhs += weights[k] * jac[k][r] * err_vec[k];
            }
            jtwr[r] = rhs;
            for c in r..p_len {
                let mut acc = 0.0f64;
                for k in 0..SKETCH_GRID_BINS {
                    acc += weights[k] * jac[k][r] * jac[k][c];
                }
                jtwj[r][c] = acc;
                jtwj[c][r] = acc;
            }
            jtwj[r][r] += lambda * (jtwj[r][r].max(1e-3));
        }

        let Some(delta) = solve_linear_system(&jtwj, &jtwr) else {
            lambda *= 4.0;
            continue;
        };

        let saved: Vec<EqBandParams> = bands.to_vec();
        for (j, &pref) in params.iter().enumerate() {
            let step = delta[j].clamp(-0.6, 0.6);
            let v = get_param_val(bands, pref) + step;
            set_param_val(bands, pref, v);
        }

        let mut cand_err = [0.0f64; SKETCH_GRID_BINS];
        let cand_cost = eval_weighted_cost(bands, grid_hz, target_db, weights, &mut cand_err);
        if cand_cost < cur_cost {
            cur_cost = cand_cost;
            err_vec = cand_err;
            lambda = (lambda * 0.4).max(1e-5);
        } else {
            bands.copy_from_slice(&saved);
            lambda = (lambda * 3.5).min(1e4);
        }
    }
}

fn solve_linear_system(a_in: &[Vec<f64>], b_in: &[f64]) -> Option<Vec<f64>> {
    let n = b_in.len();
    let mut a = a_in.to_vec();
    let mut b = b_in.to_vec();

    for col in 0..n {
        let mut pivot = col;
        let mut max_abs = a[col][col].abs();
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > max_abs {
                max_abs = v;
                pivot = row;
            }
        }
        if max_abs < 1e-12 {
            return None;
        }
        if pivot != col {
            a.swap(col, pivot);
            b.swap(col, pivot);
        }
        let diag = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / diag;
            b[row] -= factor * b[col];
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
        }
    }

    let mut x = vec![0.0f64; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for k in (row + 1)..n {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}
