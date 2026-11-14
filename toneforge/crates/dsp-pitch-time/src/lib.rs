//! `dsp-pitch-time` — Studio Vocal Pitch Detection (`pYIN`), True-Envelope Formant-Preserving
//! Pitch Correction (`Pitcher` Real-Time + `NewTone` Graphical Note Editor), and Transient-Locked
//! Polyphonic Time-Stretching (`e3 Generic` / Elastique Parity).
//!
//! All real-time DSP structures (`PyinPitchDetector`, `FormantPitchShifter`, `AutoPitchCorrectorNode`)
//! pre-allocate 100% of FFT/ring buffers during construction and perform **0 heap allocations**
//! during real-time audio processing.

pub mod formant_pitch_shifter;
pub mod pitch_detector;
pub mod pitcher_newtone;
pub mod polyphonic_time_stretch;

pub use formant_pitch_shifter::{
    FormantPitchShifter, estimate_vocal_formant_envelope_db, find_formant_peak_in_band,
    princarg_f32,
};
pub use pitch_detector::{
    MAX_PYIN_CANDIDATES, PYIN_MAX_FREQ_HZ, PYIN_MIN_FREQ_HZ, PYIN_NUM_THRESHOLDS,
    PitchFrameResult, PyinPitchDetector, VITERBI_PITCH_STATES,
};
pub use pitcher_newtone::{
    AutoPitchCorrectorNode, MusicalScale, VocalNoteBlob, VocalNoteGraphEngine,
};
pub use polyphonic_time_stretch::{
    TransientLockedTimeStretcher, resample_kaiser_sinc_stereo, resample_kaiser_sinc_stereo_cancellable,
};

#[cfg(test)]
mod tests {
    use super::*;
    use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode};
    use std::f64::consts::TAU;

    /// Synthesizes a realistic vocal vowel using 2nd-order all-pole/Lorentzian formant resonators
    /// `(f1, f2, f3)` with `n_harmonics` and optional vibrato (`vib_rate_hz`, `vib_depth_cents`).
    fn synth_formant_vocal_custom(
        len: usize,
        sr: f64,
        base_f0: f64,
        vib_rate_hz: f64,
        vib_depth_cents: f64,
        n_harmonics: usize,
        formants_hz: [f64; 3],
    ) -> (Vec<f32>, Vec<f64>) {
        let mut out = Vec::with_capacity(len);
        let mut inst_f0 = Vec::with_capacity(len);
        let mut phase = 0.0_f64;

        let formant_gain = |f: f64| -> f64 {
            let r1 = 1.0 / (1.0 + ((f - formants_hz[0]) / 65.0).powi(2));
            let r2 = 0.85 / (1.0 + ((f - formants_hz[1]) / 85.0).powi(2));
            let r3 = 0.50 / (1.0 + ((f - formants_hz[2]) / 120.0).powi(2));
            r1 + r2 + r3 + 0.03
        };

        for n in 0..len {
            let t = (n as f64) / sr;
            let cents = vib_depth_cents * (TAU * vib_rate_hz * t).sin();
            let f0 = base_f0 * 2.0_f64.powf(cents / 1200.0);
            inst_f0.push(f0);
            phase += TAU * f0 / sr;

            let mut acc = 0.0_f64;
            for h in 1..=n_harmonics {
                let hf = (h as f64) * f0;
                if hf < 0.46 * sr {
                    acc += formant_gain(hf) * ((h as f64) * phase).sin();
                }
            }
            out.push((acc * 0.16) as f32);
        }
        (out, inst_f0)
    }

    fn synth_formant_vocal(
        len: usize,
        sr: f64,
        base_f0: f64,
        vib_rate_hz: f64,
        vib_depth_cents: f64,
        n_harmonics: usize,
    ) -> (Vec<f32>, Vec<f64>) {
        synth_formant_vocal_custom(
            len,
            sr,
            base_f0,
            vib_rate_hz,
            vib_depth_cents,
            n_harmonics,
            [730.0, 1090.0, 2440.0],
        )
    }

    #[test]
    fn test_pyin_vibrato_cents_accuracy_and_sibilant_rejection() {
        let sr = 48_000.0;
        let voiced_len = 48_000; // 1.0s voiced vibrato A3=220 Hz ± 30 cents @ 5.5 Hz
        let (mut sig, mut true_f0) = synth_formant_vocal(voiced_len, sr, 220.0, 5.5, 30.0, 14);

        // Append 0.5s high-passed sibilant noise (5-10 kHz) using deterministic LCG + 2nd-order HPF/BPF
        let sib_len = 24_000;
        let mut rng_state = 0x1234_5678_9ABC_DEF0_u64;
        let mut z1 = 0.0_f64;
        let mut z2 = 0.0_f64;
        for _ in 0..sib_len {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let white = ((rng_state >> 32) as i32 as f64) / (i32::MAX as f64);
            // High-frequency difference filter emphasizing 5..10 kHz sibilance
            let hp = white - 1.45 * z1 + 0.65 * z2;
            z2 = z1;
            z1 = white;
            sig.push((hp * 0.12) as f32);
            true_f0.push(0.0);
        }

        let mut det = PyinPitchDetector::new(sr);
        let mut sq_err_sum = 0.0_f64;
        let mut voiced_count = 0usize;

        // Evaluate steady interior of voiced vibrato segment (`[0.1s .. 0.9s]`)
        let mut pos = 4_800usize;
        while pos <= 43_200 {
            let res = det.analyze_centered_at(&sig, pos);
            assert!(res.is_voiced, "Voiced vibrato frame at {pos} must be detected voiced");
            let err_cents = res.cents_error_from(true_f0[pos]);
            sq_err_sum += err_cents * err_cents;
            voiced_count += 1;
            pos += 256;
        }
        let rms_cents = (sq_err_sum / (voiced_count as f64)).sqrt();
        assert!(
            rms_cents < 2.0,
            "pYIN vibrato RMS error {rms_cents:.3} cents >= 2.0 cents"
        );

        // Evaluate sibilant segment (`[1.1s .. 1.45s]`)
        det.reset();
        let mut max_sib_prob = 0.0_f32;
        let mut s_pos = 52_800usize;
        while s_pos <= 69_600 {
            let res = det.analyze_centered_at(&sig, s_pos);
            max_sib_prob = max_sib_prob.max(res.voiced_prob);
            assert!(!res.is_voiced, "Sibilant frame at {s_pos} must not trigger voiced octave jump");
            s_pos += 256;
        }
        assert!(
            max_sib_prob < 0.20,
            "Sibilant voiced_prob {max_sib_prob:.3} >= 0.20"
        );
    }

    #[test]
    fn test_formant_preserving_pitch_shift_plus_5_semitones() {
        let sr = 48_000.0;
        // Peterson & Barney adult vocal `/ae/` (`F1=660 Hz, F2=1720 Hz, F3=2410 Hz`) at A3=220 Hz -> D4=293.66 Hz (+5 st)
        let (vocal_in, _) =
            synth_formant_vocal_custom(48_000, sr, 220.0, 0.0, 0.0, 24, [660.0, 1720.0, 2410.0]);

        let mut shifter = FormantPitchShifter::new(sr);
        let shifted = shifter.process_offline_compensated(&vocal_in, 220.0, 5.0, 0.0, true);
        assert_eq!(shifted.len(), vocal_in.len());
        assert!(shifted.iter().all(|s| s.is_finite() && s.abs() <= 1.0));

        // Verify output fundamental frequency is 220 * 2^(5/12) = 293.665 Hz (+500 ± 5 cents)
        let mut det = PyinPitchDetector::new(sr);
        let pitch_out = det.analyze_centered_at(&shifted, 24_000);
        let expected_f0 = 220.0 * 2.0_f64.powf(5.0 / 12.0);
        let f0_err_cents = pitch_out.cents_error_from(expected_f0).abs();
        assert!(
            pitch_out.is_voiced && f0_err_cents <= 5.0,
            "Shifted pitch f0={} Hz error={f0_err_cents:.2} cents > 5.0 cents",
            pitch_out.f0_hz
        );

        // Verify formant preservation (F1 ~ 660 Hz, F2 ~ 1720 Hz shift < 8% vs input; naive would shift +33.5% to 881 & 2296 Hz!)
        let (freqs_in, env_in) =
            estimate_vocal_formant_envelope_db(&vocal_in[8_000..40_000], sr, 220.0, 2048);
        let (freqs_out, env_out) = estimate_vocal_formant_envelope_db(
            &shifted[8_000..40_000],
            sr,
            expected_f0 as f32,
            2048,
        );

        let f1_in = find_formant_peak_in_band(&freqs_in, &env_in, 450.0, 1000.0);
        let f2_in = find_formant_peak_in_band(&freqs_in, &env_in, 1300.0, 2150.0);
        let f1_out = find_formant_peak_in_band(&freqs_out, &env_out, 450.0, 1000.0);
        let f2_out = find_formant_peak_in_band(&freqs_out, &env_out, 1300.0, 2150.0);

        let f1_shift_pct = ((f1_out - f1_in).abs() / f1_in) * 100.0;
        let f2_shift_pct = ((f2_out - f2_in).abs() / f2_in) * 100.0;
        assert!(
            f1_shift_pct < 8.0 && f2_shift_pct < 8.0,
            "Formants shifted too much! F1: {f1_in:.1}->{f1_out:.1} Hz ({f1_shift_pct:.2}%), F2: {f2_in:.1}->{f2_out:.1} Hz ({f2_shift_pct:.2}%)"
        );

        // Also test male vocal D3=146.83 Hz `/a/` (`F1=730 Hz, F2=1150 Hz, F3=2440 Hz`) shifted +5 st -> G3=196.0 Hz
        let (male_in, _) =
            synth_formant_vocal_custom(48_000, sr, 146.83, 0.0, 0.0, 30, [730.0, 1150.0, 2440.0]);
        let male_out = shifter.process_offline_compensated(&male_in, 146.83, 5.0, 0.0, true);
        let (mf_in, me_in) =
            estimate_vocal_formant_envelope_db(&male_in[8_000..40_000], sr, 146.83, 2048);
        let (mf_out, me_out) = estimate_vocal_formant_envelope_db(
            &male_out[8_000..40_000],
            sr,
            146.83 * 2.0_f32.powf(5.0 / 12.0),
            2048,
        );
        let mf1_in = find_formant_peak_in_band(&mf_in, &me_in, 500.0, 920.0);
        let mf2_in = find_formant_peak_in_band(&mf_in, &me_in, 950.0, 1450.0);
        let mf1_out = find_formant_peak_in_band(&mf_out, &me_out, 500.0, 920.0);
        let mf2_out = find_formant_peak_in_band(&mf_out, &me_out, 950.0, 1450.0);
        let mf1_pct = ((mf1_out - mf1_in).abs() / mf1_in) * 100.0;
        let mf2_pct = ((mf2_out - mf2_in).abs() / mf2_in) * 100.0;
        assert!(
            mf1_pct < 8.0 && mf2_pct < 8.0,
            "Male /a/ formants shifted too much! F1: {mf1_in:.1}->{mf1_out:.1} Hz ({mf1_pct:.2}%), F2: {mf2_in:.1}->{mf2_out:.1} Hz ({mf2_pct:.2}%)"
        );
    }

    #[test]
    fn test_pitcher_rt_latency_and_newtone_editor() {
        let sr = 48_000.0;
        let mut pitcher = AutoPitchCorrectorNode::new(sr);
        pitcher.scale = MusicalScale::Major;
        pitcher.root_key = 0; // C Major
        pitcher.retune_speed_ms = 5.0;
        let lat_ms = (pitcher.latency_samples() as f64) * 1000.0 / sr;
        assert!(lat_ms < 12.0, "Pitcher RT latency {lat_ms:.2} ms >= 12.0 ms");

        // Detuned A3 + 35 cents (`~224.5 Hz`) -> should snap cleanly to A3 (`220 Hz`, MIDI 57 in C Major)
        let (detuned, _) = synth_formant_vocal(24_000, sr, 224.5, 0.0, 0.0, 10);
        let mut stereo: Vec<StereoFrame> = detuned.iter().map(|&x| [x, x]).collect();
        let ctx = ProcessContext {
            sample_rate: sr,
            current_sample_pos: 0,
            ppq_tick: 0,
            bpm: 120.0,
            slice_frames: 256,
            is_playing: true,
            is_offline_rendering: false,
            ppq_resolution: daw_core_graph::DEFAULT_PPQ,
        };
        for chunk in stereo.chunks_mut(256) {
            pitcher.process_slice(&ctx, chunk, None, &mut []);
        }
        assert!(stereo.iter().all(|f| f[0].is_finite() && f[1].is_finite()));

        // NewTone graphical note engine: two notes (A3 -> C4), split, transpose, time-warp & render
        let (n1, _) = synth_formant_vocal(16_000, sr, 220.0, 5.0, 15.0, 10);
        let (n2, _) = synth_formant_vocal(16_000, sr, 261.63, 5.0, 15.0, 10);
        let mut phrase = Vec::with_capacity(36_000);
        for &s in &n1 {
            phrase.push([s, s]);
        }
        phrase.resize(phrase.len() + 4_000, [0.0, 0.0]);
        for &s in &n2 {
            phrase.push([s, s]);
        }

        let mut graph = VocalNoteGraphEngine::analyze_from_stereo(phrase, 48_000);
        assert!(graph.blobs.len() >= 2, "Should segment at least 2 vocal note blobs");
        let first_id = graph.blobs[0].blob_id;
        graph.move_note_pitch(first_id, 60.0, 0.0).unwrap();
        graph.set_note_expression(first_id, 0.8, 0.5, 0.0).unwrap();
        let mid = (graph.blobs[0].start_sample + graph.blobs[0].end_sample) / 2;
        let _ = graph.split_note_at_sample(first_id, mid).unwrap();
        let rendered = graph.render_edited_vocal_stereo();
        assert!(!rendered.is_empty());
        assert!(rendered.iter().all(|f| f[0].is_finite() && f[1].is_finite()));
    }

    #[test]
    fn test_transient_locked_polyphonic_time_stretcher_crest_and_stereo_corr() {
        let sr = 48_000.0;
        let len = 96_000usize; // 2.0 seconds
        let mut loop_in = vec![[0.0_f32; 2]; len];
        let kick_positions = [0usize, 24_000, 48_000, 72_000];
        let snare_positions = [12_000usize, 36_000, 60_000, 84_000];

        for n in 0..len {
            let t = (n as f64) / sr;
            // Sustained stereo Cmaj7 chord (C4, E4, G4, B4) with stereo chorus detuning
            let chord_l = 0.08 * (TAU * 261.63 * t).sin()
                + 0.07 * (TAU * 329.63 * t).sin()
                + 0.07 * (TAU * 392.00 * t).sin()
                + 0.06 * (TAU * 493.88 * t).sin();
            let chord_r = 0.08 * (TAU * 261.63 * t + 0.12).sin()
                + 0.07 * (TAU * 329.63 * t - 0.10).sin()
                + 0.07 * (TAU * 392.00 * t + 0.08).sin()
                + 0.06 * (TAU * 493.88 * t - 0.09).sin();

            let mut drum = 0.0_f64;
            for &kp in &kick_positions {
                if n >= kp && n < kp + 4_800 {
                    let kt = (n - kp) as f64 / sr;
                    let click = 0.55 * (-kt * 450.0).exp() * (TAU * 900.0 * kt).sin();
                    let body = 0.75 * (-kt * 28.0).exp() * (TAU * 60.0 * kt).sin();
                    drum += click + body;
                }
            }
            for &sp in &snare_positions {
                if n >= sp && n < sp + 3_600 {
                    let st = (n - sp) as f64 / sr;
                    let snap = 0.60 * (-st * 260.0).exp() * ((n as f64 * 12.9898).sin() * 43758.5).fract();
                    let tone = 0.45 * (-st * 45.0).exp() * (TAU * 185.0 * st).sin();
                    drum += snap + tone;
                }
            }

            loop_in[n] = [(chord_l + drum) as f32, (chord_r + drum) as f32];
        }

        let measure_kick_crest = |buf: &[StereoFrame], onset: usize| -> f64 {
            let win = (0.020 * sr) as usize; // 20 ms
            let end = (onset + win).min(buf.len());
            let slice = &buf[onset..end];
            let mut pk = 0.0_f64;
            let mut sq = 0.0_f64;
            for f in slice {
                let m = 0.5 * ((f[0] as f64) + (f[1] as f64));
                pk = pk.max(m.abs());
                sq += m * m;
            }
            let rms = (sq / (slice.len().max(1) as f64)).sqrt().max(1.0e-12);
            pk / rms
        };

        let measure_stereo_corr = |buf: &[StereoFrame]| -> f64 {
            let mut sl2 = 0.0_f64;
            let mut sr2 = 0.0_f64;
            let mut slr = 0.0_f64;
            for f in buf {
                let l = f[0] as f64;
                let r = f[1] as f64;
                sl2 += l * l;
                sr2 += r * r;
                slr += l * r;
            }
            slr / (sl2 * sr2).sqrt().max(1.0e-18)
        };

        let orig_crest = measure_kick_crest(&loop_in, 24_000);
        let mut stretcher = TransientLockedTimeStretcher::new(sr);

        for &ratio in &[1.50_f32, 0.75_f32] {
            let out = stretcher.stretch_stereo_offline(&loop_in, ratio, 0.0);
            let expected_len = ((len as f64) * (ratio as f64)).round() as usize;
            assert!(
                out.len().abs_diff(expected_len) <= 4,
                "Ratio {ratio}: len {} != expected {expected_len}",
                out.len()
            );
            let stretched_kick_onset = (24_000.0 * (ratio as f64)).round() as usize;
            let out_crest = measure_kick_crest(&out, stretched_kick_onset);
            let crest_retention = out_crest / orig_crest;
            assert!(
                crest_retention >= 0.85,
                "Ratio {ratio}: Kick crest retention {:.3} < 0.85 (orig={orig_crest:.2}, out={out_crest:.2})",
                crest_retention
            );
            let corr = measure_stereo_corr(&out);
            assert!(
                corr >= 0.95,
                "Ratio {ratio}: Stereo correlation {corr:.4} < 0.95"
            );
        }
    }
}
