//! Real-Time `AutoPitchCorrectorNode` (`Pitcher`) & Graphical `VocalNoteGraphEngine` (`NewTone`)
//! (`pitcher_newtone.rs`).
//!
//! - [`AutoPitchCorrectorNode`]: Implements `daw_core_graph::UnifiedDspNode` with `< 12 ms` algorithmic
//!   latency (`512` samples = `10.67 ms` @ `48 kHz`), zero real-time heap allocations, scale-quantized
//!   retuning (`Root Key`, `MusicalScale`, `Retune Speed 0..400 ms`, `Humanize 0..100%`), and
//!   True-Envelope Cepstral Formant Preservation (`Formant Shift ±12 st`).
//! - [`VocalNoteGraphEngine`]: Offline/interactive graphical note segmentation & editing engine (`NewTone`
//!   parity) producing editable [`VocalNoteBlob`]s (`Pitch Center`, `Fine Pitch`, `Pitch Drift`,
//!   `Vibrato Scaling`, `Formant Shift`, `Razor Split`, `Time Warp`) and rendering phase-coherent stereo
//!   audio ready to commit directly onto the ToneForge Playlist.

use crate::formant_pitch_shifter::FormantPitchShifter;
use crate::pitch_detector::{PitchFrameResult, PyinPitchDetector};
use crate::polyphonic_time_stretch::TransientLockedTimeStretcher;
use daw_core_graph::{ProcessContext, StereoFrame, UnifiedDspNode};
use serde::{Deserialize, Serialize};

/// Musical scale mask (`12` semitone degrees relative to `root_key` `0 = C .. 11 = B`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicalScale {
    Chromatic,
    Major,
    Minor,
    PentatonicMajor,
    PentatonicMinor,
    HarmonicMinor,
    Dorian,
    Blues,
    Custom(u16),
}

impl Default for MusicalScale {
    fn default() -> Self {
        Self::Chromatic
    }
}

impl MusicalScale {
    /// Returns the 12-bit degree bitmask (`bit d == 1` means semitone offset `d` from `root_key` is active).
    #[inline]
    pub const fn degree_mask(self) -> u16 {
        match self {
            Self::Chromatic => 0b1111_1111_1111,
            // C D E F G A B -> 0, 2, 4, 5, 7, 9, 11
            Self::Major => (1 << 0) | (1 << 2) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 9) | (1 << 11),
            // Natural Minor: 0, 2, 3, 5, 7, 8, 10
            Self::Minor => (1 << 0) | (1 << 2) | (1 << 3) | (1 << 5) | (1 << 7) | (1 << 8) | (1 << 10),
            // Major Pentatonic: 0, 2, 4, 7, 9
            Self::PentatonicMajor => (1 << 0) | (1 << 2) | (1 << 4) | (1 << 7) | (1 << 9),
            // Minor Pentatonic: 0, 3, 5, 7, 10
            Self::PentatonicMinor => (1 << 0) | (1 << 3) | (1 << 5) | (1 << 7) | (1 << 10),
            // Harmonic Minor: 0, 2, 3, 5, 7, 8, 11
            Self::HarmonicMinor => {
                (1 << 0) | (1 << 2) | (1 << 3) | (1 << 5) | (1 << 7) | (1 << 8) | (1 << 11)
            }
            // Dorian: 0, 2, 3, 5, 7, 9, 10
            Self::Dorian => (1 << 0) | (1 << 2) | (1 << 3) | (1 << 5) | (1 << 7) | (1 << 9) | (1 << 10),
            // Blues: 0, 3, 5, 6, 7, 10
            Self::Blues => (1 << 0) | (1 << 3) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 10),
            Self::Custom(mask) => {
                let m = mask & 0x0FFF;
                if m == 0 { 1 } else { m }
            }
        }
    }

    #[inline]
    pub fn from_index(idx: u8) -> Self {
        match idx {
            0 => Self::Chromatic,
            1 => Self::Major,
            2 => Self::Minor,
            3 => Self::PentatonicMajor,
            4 => Self::PentatonicMinor,
            5 => Self::HarmonicMinor,
            6 => Self::Dorian,
            7 => Self::Blues,
            _ => Self::Chromatic,
        }
    }

    #[inline]
    pub const fn to_index(self) -> u8 {
        match self {
            Self::Chromatic => 0,
            Self::Major => 1,
            Self::Minor => 2,
            Self::PentatonicMajor => 3,
            Self::PentatonicMinor => 4,
            Self::HarmonicMinor => 5,
            Self::Dorian => 6,
            Self::Blues => 7,
            Self::Custom(_) => 8,
        }
    }

    /// Snaps a continuous MIDI note (`midi_float`) to the nearest allowed scale degree relative to `root_key` (`0..11`).
    #[inline]
    pub fn snap_midi_note(self, root_key: u8, midi_float: f32) -> f32 {
        let mask = self.degree_mask();
        let root = (root_key % 12) as i32;
        let center_int = midi_float.round() as i32;
        let mut best_note = center_int;
        let mut best_dist = f32::MAX;
        for cand in (center_int - 12)..=(center_int + 12) {
            let deg = ((cand - root).rem_euclid(12)) as u16;
            if (mask & (1 << deg)) != 0 {
                let dist = ((cand as f32) - midi_float).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_note = cand;
                }
            }
        }
        best_note as f32
    }
}

/// Real-Time Vocal Pitch Corrector (`FL Studio Pitcher` parity) implementing `UnifiedDspNode`.
///
/// Latency is strictly `< 12 ms` (`512` samples = `10.67 ms` @ `48 kHz`, `11.61 ms` @ `44.1 kHz`,
/// `10.67 ms` @ `96 kHz`). Zero heap allocations occur in `process_slice`.
pub struct AutoPitchCorrectorNode {
    sample_rate: f64,
    detector: PyinPitchDetector,
    shifter_l: FormantPitchShifter,
    shifter_r: FormantPitchShifter,
    dry_delay_l: Vec<f32>,
    dry_delay_r: Vec<f32>,
    dry_delay_mask: usize,
    dry_write_pos: usize,

    // Parameters
    pub root_key: u8,
    pub scale: MusicalScale,
    /// Retune speed in milliseconds (`0.0 ms` = instant T-Pain hard-tune .. `400.0 ms` = transparent).
    pub retune_speed_ms: f32,
    /// Humanize factor (`0.0 ..= 1.0`): relaxes retune speed on sustained notes to preserve natural vibrato.
    pub humanize: f32,
    /// Preserves vocal tract formants $F_1, F_2, F_3$ via True-Envelope Cepstral Liftering.
    pub formant_preserve: bool,
    /// Independent gender/throat formant shift (`-12.0 ..= +12.0` semitones).
    pub formant_shift_semitones: f32,
    /// Wet/dry mix (`0.0 ..= 1.0`).
    pub mix: f32,
    pub bypass: bool,

    // Smoothed real-time pitch tracking state
    smoothed_shift_st: f32,
    sustained_frames_on_note: u32,
    last_target_midi: i32,
    latest_detected_midi: f32,
    latest_target_midi: f32,
    latest_voiced_prob: f32,
}

impl AutoPitchCorrectorNode {
    pub fn new(sample_rate: f64) -> Self {
        let sr = sample_rate.clamp(8_000.0, 192_000.0);
        let shifter_l = FormantPitchShifter::new_low_latency_rt(sr);
        let shifter_r = FormantPitchShifter::new_low_latency_rt(sr);
        let hop = (shifter_l.latency_samples() / 4).max(64);
        let detector = PyinPitchDetector::with_range(sr, 55.0, 1200.0, hop);
        let delay_cap = (shifter_l.latency_samples() * 4).next_power_of_two();

        Self {
            sample_rate: sr,
            detector,
            shifter_l,
            shifter_r,
            dry_delay_l: vec![0.0; delay_cap],
            dry_delay_r: vec![0.0; delay_cap],
            dry_delay_mask: delay_cap - 1,
            dry_write_pos: 0,
            root_key: 0, // C
            scale: MusicalScale::Chromatic,
            retune_speed_ms: 15.0,
            humanize: 0.25,
            formant_preserve: true,
            formant_shift_semitones: 0.0,
            mix: 1.0,
            bypass: false,
            smoothed_shift_st: 0.0,
            sustained_frames_on_note: 0,
            last_target_midi: 60,
            latest_detected_midi: 0.0,
            latest_target_midi: 0.0,
            latest_voiced_prob: 0.0,
        }
    }

    #[inline]
    pub fn latest_pitch_telemetry(&self) -> (f32, f32, f32) {
        (
            self.latest_detected_midi,
            self.latest_target_midi,
            self.latest_voiced_prob,
        )
    }
}

impl UnifiedDspNode for AutoPitchCorrectorNode {
    fn name(&self) -> &'static str {
        "ToneForge Pitcher Real-Time Vocal Auto-Tune"
    }

    fn latency_samples(&self) -> usize {
        if self.bypass {
            0
        } else {
            self.shifter_l.latency_samples()
        }
    }

    fn prepare(&mut self, sample_rate: f64, _max_block_size: usize) {
        let sr = sample_rate.clamp(8_000.0, 192_000.0);
        if (sr - self.sample_rate).abs() > 1.0 {
            let root = self.root_key;
            let scale = self.scale;
            let speed = self.retune_speed_ms;
            let hum = self.humanize;
            let fp = self.formant_preserve;
            let fs_st = self.formant_shift_semitones;
            let mix = self.mix;
            let byp = self.bypass;
            *self = Self::new(sr);
            self.root_key = root;
            self.scale = scale;
            self.retune_speed_ms = speed;
            self.humanize = hum;
            self.formant_preserve = fp;
            self.formant_shift_semitones = fs_st;
            self.mix = mix;
            self.bypass = byp;
        }
    }

    fn reset_state(&mut self) {
        self.detector.reset();
        self.shifter_l.reset();
        self.shifter_r.reset();
        self.dry_delay_l.fill(0.0);
        self.dry_delay_r.fill(0.0);
        self.dry_write_pos = 0;
        self.smoothed_shift_st = 0.0;
        self.sustained_frames_on_note = 0;
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.root_key = ((v * 11.0).round() as u8).min(11),
            1 => self.scale = MusicalScale::from_index((v * 7.0).round() as u8),
            2 => self.retune_speed_ms = v * 400.0,
            3 => self.humanize = v,
            4 => self.formant_preserve = v >= 0.5,
            5 => self.formant_shift_semitones = -12.0 + v * 24.0,
            6 => self.mix = v,
            7 => self.bypass = v >= 0.5,
            _ => {}
        }
    }

    fn process_slice(
        &mut self,
        _ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        if self.bypass || main_io.is_empty() {
            return;
        }

        let lat = self.shifter_l.latency_samples();
        let dmask = self.dry_delay_mask;
        let sr_f32 = self.sample_rate as f32;
        let wet = self.mix.clamp(0.0, 1.0);
        let dry = 1.0 - wet;

        for frame in main_io.iter_mut() {
            let mono = 0.5 * (frame[0] + frame[1]);
            let pitch: PitchFrameResult = self.detector.push_samples(std::slice::from_ref(&mono));

            let desired_shift = if pitch.is_voiced && pitch.f0_hz > 40.0 {
                let snapped_midi = self.scale.snap_midi_note(self.root_key, pitch.midi_note_float);
                let snapped_int = snapped_midi.round() as i32;
                if snapped_int == self.last_target_midi {
                    self.sustained_frames_on_note = self.sustained_frames_on_note.saturating_add(1);
                } else {
                    self.last_target_midi = snapped_int;
                    self.sustained_frames_on_note = 0;
                }
                self.latest_detected_midi = pitch.midi_note_float;
                self.latest_target_midi = snapped_midi;
                self.latest_voiced_prob = pitch.voiced_prob;
                (snapped_midi - pitch.midi_note_float).clamp(-12.0, 12.0)
            } else {
                self.latest_voiced_prob = pitch.voiced_prob;
                // Decay shift smoothly toward 0 on unvoiced consonants/breaths so sibilants stay natural!
                0.0
            };

            // Humanize slows down correction on sustained vibrato (> 60 ms into a note)
            let sustain_sec = (self.sustained_frames_on_note as f32) / sr_f32;
            let humanize_tau_ms = if sustain_sec > 0.06 {
                self.humanize * 140.0 * ((sustain_sec - 0.06) * 5.0).min(1.0)
            } else {
                0.0
            };
            let eff_tau_sec = ((self.retune_speed_ms + humanize_tau_ms) * 0.001).max(0.0002);
            let alpha = 1.0 - (-1.0 / (eff_tau_sec * sr_f32)).exp();
            self.smoothed_shift_st += alpha * (desired_shift - self.smoothed_shift_st);

            let active_shift = if pitch.is_voiced {
                self.smoothed_shift_st
            } else {
                self.smoothed_shift_st * 0.5
            };

            self.shifter_l.set_detected_pitch(pitch.f0_hz, pitch.voiced_prob);
            self.shifter_r.set_detected_pitch(pitch.f0_hz, pitch.voiced_prob);
            self.shifter_l.set_shift_params(
                active_shift,
                self.formant_shift_semitones,
                self.formant_preserve,
            );
            self.shifter_r.set_shift_params(
                active_shift,
                self.formant_shift_semitones,
                self.formant_preserve,
            );

            // Delay dry signal by exact lookahead `lat` so parallel Wet/Dry mix is 100% phase-aligned
            let wpos = self.dry_write_pos & dmask;
            self.dry_delay_l[wpos] = frame[0];
            self.dry_delay_r[wpos] = frame[1];
            let rpos = self.dry_write_pos.wrapping_sub(lat) & dmask;
            let dry_l = self.dry_delay_l[rpos];
            let dry_r = self.dry_delay_r[rpos];
            self.dry_write_pos = self.dry_write_pos.wrapping_add(1);

            let wet_l = self.shifter_l.process_sample(frame[0]);
            let wet_r = if (frame[0] - frame[1]).abs() < 1.0e-6 {
                // Keep shifter_r internal state warm while saving redundant dual-mono FFTs if identical
                let _ = self.shifter_r.process_sample(frame[1]);
                wet_l
            } else {
                self.shifter_r.process_sample(frame[1])
            };

            frame[0] = dry * dry_l + wet * wet_l;
            frame[1] = dry * dry_r + wet * wet_r;
        }
    }
}

/// Graphical vocal note segment (`FL Studio NewTone` / `Melodyne` blob parity).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VocalNoteBlob {
    pub blob_id: u32,
    pub start_sample: usize,
    pub end_sample: usize,
    /// Original median MIDI pitch detected by `pYIN`.
    pub detected_midi_note: f32,
    /// Target quantized/dragged integer or continuous MIDI note (`0..127`).
    pub target_midi_note: f32,
    /// Fine pitch offset in cents (`-100.0 ..= +100.0`).
    pub fine_pitch_cents: f32,
    /// Flattens slow intra-note pitch drift (`0.0` = original drift, `1.0` = rock-steady center).
    pub pitch_drift_correction: f32,
    /// Scales natural pitch modulation / vibrato around the note center (`0.0` = flat, `1.0` = natural, `2.0` = exaggerated).
    pub vibrato_scaling: f32,
    /// Per-note throat/gender formant shift (`-12.0 ..= +12.0` semitones).
    pub formant_shift_semitones: f32,
    /// Target timeline start sample after horizontal time-warping.
    pub warped_start_sample: usize,
    /// Target timeline end sample after horizontal time-warping.
    pub warped_end_sample: usize,
    /// Per-blob gain trim in dB (`-24.0 ..= +12.0`).
    pub gain_db: f32,
    /// Sampled pitch trajectory (in continuous MIDI note units, 1 point per analysis hop) for 120 FPS UI rendering.
    pub pitch_contour_midi: Vec<f32>,
}

impl VocalNoteBlob {
    #[inline]
    pub fn duration_samples(&self) -> usize {
        self.end_sample.saturating_sub(self.start_sample)
    }

    #[inline]
    pub fn warped_duration_samples(&self) -> usize {
        self.warped_end_sample.saturating_sub(self.warped_start_sample)
    }

    #[inline]
    pub fn effective_target_midi(&self) -> f32 {
        self.target_midi_note + self.fine_pitch_cents * 0.01
    }
}

/// Complete NewTone Graphical Vocal Pitch & Time Editor Engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VocalNoteGraphEngine {
    pub sample_rate: u32,
    pub hop_samples: usize,
    pub source_frames: Vec<StereoFrame>,
    pub blobs: Vec<VocalNoteBlob>,
    next_blob_id: u32,
}

impl VocalNoteGraphEngine {
    /// Analyzes a stereo (or dual-mono) vocal clip via `PyinPitchDetector` and segments it into
    /// editable [`VocalNoteBlob`]s.
    pub fn analyze_from_stereo(source_frames: Vec<StereoFrame>, sample_rate: u32) -> Self {
        let sr = sample_rate.clamp(8_000, 192_000);
        let hop = ((0.005333 * (sr as f64)).round() as usize).max(64); // ~5.33 ms (`256` @ 48k)
        let mono: Vec<f32> = source_frames
            .iter()
            .map(|f| 0.5 * (f[0] + f[1]))
            .collect();

        let mut det = PyinPitchDetector::with_range(sr as f64, 50.0, 1200.0, hop);
        let total_frames = mono.len();
        let num_hops = total_frames.div_ceil(hop);
        let mut hop_results = Vec::with_capacity(num_hops);

        for h in 0..num_hops {
            let center = (h * hop + hop / 2).min(total_frames.saturating_sub(1));
            let res = det.analyze_centered_at(&mono, center);
            hop_results.push(res);
        }

        // Segment contiguous voiced hops into notes; split when pitch jumps > 0.70 semitones for >= 3 hops
        let mut blobs = Vec::new();
        let mut next_id = 1u32;
        let min_note_hops = 4usize; // ~21 ms minimum note blob

        let mut h = 0usize;
        while h < num_hops {
            if !hop_results[h].is_voiced || hop_results[h].f0_hz <= 0.0 {
                h += 1;
                continue;
            }
            let seg_start_h = h;
            let mut contour = Vec::new();
            let mut running_median = hop_results[h].midi_note_float;
            contour.push(hop_results[h].midi_note_float);
            h += 1;

            while h < num_hops && hop_results[h].is_voiced && hop_results[h].f0_hz > 0.0 {
                let m = hop_results[h].midi_note_float;
                // Check if sustained step transition occurred (> 0.75 semitones for 3 consecutive frames)
                if (m - running_median).abs() > 0.75 && h + 2 < num_hops {
                    let m1 = hop_results[h + 1];
                    let m2 = hop_results[h + 2];
                    if m1.is_voiced
                        && m2.is_voiced
                        && (m1.midi_note_float - running_median).abs() > 0.75
                        && (m2.midi_note_float - running_median).abs() > 0.75
                    {
                        break;
                    }
                }
                running_median = 0.85 * running_median + 0.15 * m;
                contour.push(m);
                h += 1;
            }

            if contour.len() >= min_note_hops {
                let mut sorted = contour.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median_midi = sorted[sorted.len() / 2];
                let start_sample = seg_start_h * hop;
                let end_sample = (h * hop).min(total_frames);
                if end_sample > start_sample {
                    blobs.push(VocalNoteBlob {
                        blob_id: next_id,
                        start_sample,
                        end_sample,
                        detected_midi_note: median_midi,
                        target_midi_note: median_midi.round(),
                        fine_pitch_cents: ((median_midi - median_midi.round()) * 100.0)
                            .clamp(-100.0, 100.0),
                        pitch_drift_correction: 0.0,
                        vibrato_scaling: 1.0,
                        formant_shift_semitones: 0.0,
                        warped_start_sample: start_sample,
                        warped_end_sample: end_sample,
                        gain_db: 0.0,
                        pitch_contour_midi: contour,
                    });
                    next_id += 1;
                }
            }
        }

        Self {
            sample_rate: sr,
            hop_samples: hop,
            source_frames,
            blobs,
            next_blob_id: next_id,
        }
    }

    /// Updates a note blob's target pitch (`target_midi_note`, `fine_pitch_cents`).
    pub fn move_note_pitch(
        &mut self,
        blob_id: u32,
        target_midi_note: f32,
        fine_pitch_cents: f32,
    ) -> Result<(), String> {
        let blob = self
            .blobs
            .iter_mut()
            .find(|b| b.blob_id == blob_id)
            .ok_or_else(|| format!("VocalNoteBlob {} not found", blob_id))?;
        blob.target_midi_note = target_midi_note.clamp(24.0, 108.0);
        blob.fine_pitch_cents = fine_pitch_cents.clamp(-100.0, 100.0);
        Ok(())
    }

    /// Updates intra-note expression (`pitch_drift_correction`, `vibrato_scaling`, `formant_shift_semitones`).
    pub fn set_note_expression(
        &mut self,
        blob_id: u32,
        pitch_drift_correction: f32,
        vibrato_scaling: f32,
        formant_shift_semitones: f32,
    ) -> Result<(), String> {
        let blob = self
            .blobs
            .iter_mut()
            .find(|b| b.blob_id == blob_id)
            .ok_or_else(|| format!("VocalNoteBlob {} not found", blob_id))?;
        blob.pitch_drift_correction = pitch_drift_correction.clamp(0.0, 1.0);
        blob.vibrato_scaling = vibrato_scaling.clamp(0.0, 2.0);
        blob.formant_shift_semitones = formant_shift_semitones.clamp(-12.0, 12.0);
        Ok(())
    }

    /// Razor-cuts an existing `VocalNoteBlob` at `split_sample` into two independent note blobs.
    pub fn split_note_at_sample(&mut self, blob_id: u32, split_sample: usize) -> Result<(u32, u32), String> {
        let idx = self
            .blobs
            .iter()
            .position(|b| b.blob_id == blob_id)
            .ok_or_else(|| format!("VocalNoteBlob {} not found", blob_id))?;
        let orig = self.blobs[idx].clone();
        if split_sample <= orig.start_sample + self.hop_samples
            || split_sample + self.hop_samples >= orig.end_sample
        {
            return Err("split_sample too close to VocalNoteBlob boundary".to_string());
        }

        let frac = (split_sample - orig.start_sample) as f64 / (orig.duration_samples() as f64);
        let split_contour_idx = ((orig.pitch_contour_midi.len() as f64) * frac)
            .round()
            .clamp(1.0, (orig.pitch_contour_midi.len().saturating_sub(1)) as f64)
            as usize;
        let warped_split = orig.warped_start_sample
            + ((orig.warped_duration_samples() as f64) * frac).round() as usize;

        let left_contour = orig.pitch_contour_midi[..split_contour_idx].to_vec();
        let right_contour = orig.pitch_contour_midi[split_contour_idx..].to_vec();

        let new_id = self.next_blob_id;
        self.next_blob_id += 1;

        self.blobs[idx].end_sample = split_sample;
        self.blobs[idx].warped_end_sample = warped_split;
        self.blobs[idx].pitch_contour_midi = left_contour;

        let right_blob = VocalNoteBlob {
            blob_id: new_id,
            start_sample: split_sample,
            end_sample: orig.end_sample,
            detected_midi_note: orig.detected_midi_note,
            target_midi_note: orig.target_midi_note,
            fine_pitch_cents: orig.fine_pitch_cents,
            pitch_drift_correction: orig.pitch_drift_correction,
            vibrato_scaling: orig.vibrato_scaling,
            formant_shift_semitones: orig.formant_shift_semitones,
            warped_start_sample: warped_split,
            warped_end_sample: orig.warped_end_sample,
            gain_db: orig.gain_db,
            pitch_contour_midi: right_contour,
        };
        self.blobs.insert(idx + 1, right_blob);
        Ok((blob_id, new_id))
    }

    /// Time-warps a `VocalNoteBlob` onto `[warped_start_sample, warped_end_sample]`.
    pub fn warp_note_timing(
        &mut self,
        blob_id: u32,
        warped_start_sample: usize,
        warped_end_sample: usize,
    ) -> Result<(), String> {
        let blob = self
            .blobs
            .iter_mut()
            .find(|b| b.blob_id == blob_id)
            .ok_or_else(|| format!("VocalNoteBlob {} not found", blob_id))?;
        if warped_end_sample <= warped_start_sample + 64 {
            return Err("Invalid warped note duration".to_string());
        }
        let ratio = (warped_end_sample - warped_start_sample) as f32
            / (blob.duration_samples().max(1) as f32);
        let clamped_len =
            ((blob.duration_samples() as f32) * ratio.clamp(0.5, 2.0)).round() as usize;
        blob.warped_start_sample = warped_start_sample;
        blob.warped_end_sample = warped_start_sample + clamped_len.max(64);
        Ok(())
    }

    /// Quantizes all detected note blobs toward `scale` (`amount` `0.0..=1.0`).
    pub fn snap_all_to_scale(&mut self, root_key: u8, scale: MusicalScale, amount: f32) {
        let amt = amount.clamp(0.0, 1.0);
        for b in &mut self.blobs {
            let snapped = scale.snap_midi_note(root_key, b.detected_midi_note);
            let blended = b.detected_midi_note + amt * (snapped - b.detected_midi_note);
            b.target_midi_note = blended.round();
            b.fine_pitch_cents = ((blended - b.target_midi_note) * 100.0).clamp(-100.0, 100.0);
        }
    }

    /// Renders the complete edited vocal track with True-Envelope Formant Preservation and
    /// Transient-Locked Time-Warping, preserving unvoiced consonants/breaths seamlessly.
    pub fn render_edited_vocal_stereo(&self) -> Vec<StereoFrame> {
        let n_total = self.source_frames.len();
        if n_total == 0 {
            return Vec::new();
        }
        let max_end = self
            .blobs
            .iter()
            .map(|b| b.warped_end_sample)
            .max()
            .unwrap_or(n_total)
            .max(n_total);
        let mut out = self.source_frames.clone();
        out.resize(max_end, [0.0, 0.0]);

        let sr = self.sample_rate as f64;
        let mut shifter_l = FormantPitchShifter::with_fft_size(sr, 1024);
        let mut shifter_r = FormantPitchShifter::with_fft_size(sr, 1024);
        let mut stretcher = TransientLockedTimeStretcher::new(sr);

        // Zero out original source regions of time-warped blobs before placing warped segments
        for blob in &self.blobs {
            if blob.warped_start_sample != blob.start_sample
                || blob.warped_end_sample != blob.end_sample
            {
                for f in &mut out[blob.start_sample..blob.end_sample.min(n_total)] {
                    *f = [0.0, 0.0];
                }
            }
        }

        let xfade = ((0.004 * sr).round() as usize).max(16); // 4 ms equal-power boundary crossfade

        for blob in &self.blobs {
            let s0 = blob.start_sample.min(n_total);
            let s1 = blob.end_sample.min(n_total);
            if s1 <= s0 + 16 {
                continue;
            }
            let seg_len = s1 - s0;
            let lat = shifter_l.latency_samples();
            shifter_l.reset();
            shifter_r.reset();

            let base_f0 = 440.0 * 2.0_f32.powf((blob.detected_midi_note - 69.0) / 12.0);
            shifter_l.set_detected_pitch(base_f0, 1.0);
            shifter_r.set_detected_pitch(base_f0, 1.0);

            let gain_lin = 10.0_f32.powf(blob.gain_db / 20.0);
            let target_midi = blob.effective_target_midi();

            let mut pitched_seg = Vec::with_capacity(seg_len + lat);
            for i in 0..(seg_len + lat) {
                let c_idx = ((i.min(seg_len - 1)) / self.hop_samples.max(1))
                    .min(blob.pitch_contour_midi.len().saturating_sub(1));
                let inst_midi = blob
                    .pitch_contour_midi
                    .get(c_idx)
                    .copied()
                    .unwrap_or(blob.detected_midi_note);

                // Decompose instantaneous contour into center + drift + vibrato deviation
                let dev = inst_midi - blob.detected_midi_note;
                let desired_inst_midi = target_midi
                    + dev * blob.vibrato_scaling * (1.0 - 0.85 * blob.pitch_drift_correction);
                let shift_st = (desired_inst_midi - inst_midi).clamp(-24.0, 24.0);

                shifter_l.set_shift_params(shift_st, blob.formant_shift_semitones, true);
                shifter_r.set_shift_params(shift_st, blob.formant_shift_semitones, true);

                let src_fr = if i < seg_len {
                    self.source_frames[s0 + i]
                } else {
                    [0.0, 0.0]
                };
                let yl = shifter_l.process_sample(src_fr[0]) * gain_lin;
                let yr = if (src_fr[0] - src_fr[1]).abs() < 1.0e-6 {
                    let _ = shifter_r.process_sample(src_fr[1]);
                    yl
                } else {
                    shifter_r.process_sample(src_fr[1]) * gain_lin
                };
                pitched_seg.push([yl, yr]);
            }

            let aligned_seg = &pitched_seg[lat.min(pitched_seg.len())..];
            let warped_len = blob.warped_duration_samples().max(16);
            let final_seg: Vec<StereoFrame> = if warped_len != aligned_seg.len() {
                let ratio = (warped_len as f32) / (aligned_seg.len().max(1) as f32);
                stretcher.stretch_stereo_offline(aligned_seg, ratio.clamp(0.5, 2.0), 0.0)
            } else {
                aligned_seg.to_vec()
            };

            let w0 = blob.warped_start_sample;
            for (i, &fr) in final_seg.iter().enumerate() {
                let dst_idx = w0 + i;
                if dst_idx >= out.len() {
                    break;
                }
                let fade = if i < xfade {
                    let t = (i as f32) / (xfade as f32);
                    (0.5 * std::f32::consts::PI * t).sin()
                } else if i + xfade >= final_seg.len() {
                    let rem = final_seg.len().saturating_sub(1 + i);
                    let t = (rem as f32) / (xfade as f32);
                    (0.5 * std::f32::consts::PI * t).sin()
                } else {
                    1.0
                };
                out[dst_idx][0] = out[dst_idx][0] * (1.0 - fade) + fr[0] * fade;
                out[dst_idx][1] = out[dst_idx][1] * (1.0 - fade) + fr[1] * fade;
            }
        }

        out
    }
}
