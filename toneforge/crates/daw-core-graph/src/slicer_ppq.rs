//! Hybrid PPQ Tick & Sample Slicer (`sub_d0ece0` at `0x00d0ece0` in `FLEngine_x64.dll`)
//! and `TempoMap` (`TTempoMap` `0x012c7b88` / `TFLStudioTempoMap` `0x00b444e0`).
//!
//! Splits hardware audio blocks (`64..4096` samples) at exact PPQ Tick boundaries using a
//! 64-bit floating-point sample accumulator (`0x15e3450`) and 8-sample SIMD alignment (`0x15e1b90`).
//! Zero Heap Allocation during `process_buffer`.

/// Default PPQ resolution in FL Studio 26 (`96` ticks per quarter note).
pub const DEFAULT_PPQ: u32 = 96;

/// Minimum supported PPQ resolution (`24` ticks per quarter note).
pub const MIN_PPQ: u32 = 24;

/// Maximum supported PPQ resolution in FL Studio 26 (`960` ticks per quarter note).
pub const MAX_PPQ: u32 = 960;

/// Standard FL Studio 26 PPQ resolutions (`96..=960`).
pub const SUPPORTED_PPQ_RESOLUTIONS: [u32; 6] = [96, 144, 192, 384, 480, 960];

/// Maximum pre-allocated tempo change points in `TempoMap`.
pub const MAX_TEMPO_POINTS: usize = 256;

/// A tempo change event on the project timeline (`TTempoMap` `0x012c7b88`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TempoPoint {
    pub ppq_tick: i64,
    pub bpm: f64,
}

/// Pre-allocated tempo map (`TTempoMap` `0x012c7b88`) supporting zero-allocation lookup at tick boundaries.
#[derive(Clone, Debug)]
pub struct TempoMap {
    points: Vec<TempoPoint>,
    count: usize,
    default_bpm: f64,
}

impl Default for TempoMap {
    fn default() -> Self {
        Self::new(128.0)
    }
}

impl TempoMap {
    pub fn new(default_bpm: f64) -> Self {
        let clamped_bpm = default_bpm.clamp(10.0, 999.0);
        Self {
            points: vec![
                TempoPoint {
                    ppq_tick: 0,
                    bpm: clamped_bpm
                };
                MAX_TEMPO_POINTS
            ],
            count: 0,
            default_bpm: clamped_bpm,
        }
    }

    /// Inserts or updates a tempo automation point (Zero Heap Allocation).
    pub fn add_point(&mut self, ppq_tick: i64, bpm: f64) -> bool {
        let bpm = bpm.clamp(10.0, 999.0);
        for i in 0..self.count {
            if self.points[i].ppq_tick == ppq_tick {
                self.points[i].bpm = bpm;
                return true;
            }
        }
        if self.count >= self.points.len() {
            return false;
        }
        let mut pos = self.count;
        while pos > 0 && self.points[pos - 1].ppq_tick > ppq_tick {
            self.points[pos] = self.points[pos - 1];
            pos -= 1;
        }
        self.points[pos] = TempoPoint { ppq_tick, bpm };
        self.count += 1;
        true
    }

    pub fn clear(&mut self) {
        self.count = 0;
    }

    /// Returns the slice of active tempo points.
    #[inline]
    pub fn active_points(&self) -> &[TempoPoint] {
        &self.points[..self.count]
    }

    /// Queries the active BPM at `ppq_tick` (Zero Heap Allocation).
    #[inline]
    pub fn bpm_at_tick(&self, ppq_tick: i64) -> f64 {
        if self.count == 0 || ppq_tick < self.points[0].ppq_tick {
            return self.default_bpm;
        }
        let active = &self.points[..self.count];
        let idx = active.partition_point(|p| p.ppq_tick <= ppq_tick).saturating_sub(1);
        active[idx].bpm
    }
}

/// Hybrid PPQ Tick / Sample Slicer (`sub_d0ece0` at `0x00d0ece0` in `FLEngine_x64.dll`).
#[derive(Clone, Debug)]
pub struct HybridPpqSlicer {
    pub sample_rate: f64,
    pub bpm: f64,
    pub ppq: u32,
    pub current_sample_pos: i64,    // `.bss` `0x15e0b78` (`CurrentSamplePos`)
    pub next_tick_sample_pos: i64,  // `.bss` `0x15e1a00` (`NextTickSamplePos`)
    pub tick_sample_accum_f64: f64, // `.bss` `0x15e3450` (`TickSampleAccumulator`)
    pub samples_per_tick_f64: f64,  // `.bss` `0x15e2600` (`SamplesPerTick`)
    pub global_ppq_tick: i64,       // `.bss` `0x15e3aa0` (`GlobalPPQTickCounter`)
    pub simd_align_8_samples: bool, // `.bss` `0x15e1b90` (8-sample SIMD alignment flag)
    pub tempo_map: Option<TempoMap>,
}

/// Type alias matching `04_flstudio26_daw_architecture.md`.
pub type HybridTickSlicer = HybridPpqSlicer;

impl HybridPpqSlicer {
    /// Initializes a new `HybridPpqSlicer` at `sample_rate`, `bpm`, and `ppq` (`24..=960`).
    pub fn new(sample_rate: f64, bpm: f64, ppq: u32) -> Self {
        let safe_sr = sample_rate.max(8000.0);
        let safe_bpm = bpm.clamp(10.0, 999.0);
        let safe_ppq = ppq.clamp(MIN_PPQ, MAX_PPQ);
        let spt = Self::compute_samples_per_tick(safe_sr, safe_bpm, safe_ppq);
        Self {
            sample_rate: safe_sr,
            bpm: safe_bpm,
            ppq: safe_ppq,
            current_sample_pos: 0,
            next_tick_sample_pos: 0,
            tick_sample_accum_f64: 0.0,
            samples_per_tick_f64: spt,
            global_ppq_tick: 0,
            simd_align_8_samples: true,
            tempo_map: None,
        }
    }

    /// Exact formula from `0x15e2600`: `SamplesPerTick = (60.0 * Fs) / (BPM * PPQ)`.
    #[inline]
    pub fn compute_samples_per_tick(sample_rate: f64, bpm: f64, ppq: u32) -> f64 {
        (60.0 * sample_rate) / (bpm.max(1e-6) * (ppq.max(1) as f64))
    }

    /// Updates BPM dynamically and recalculates `samples_per_tick_f64` (Zero Heap Allocation).
    #[inline]
    pub fn set_bpm(&mut self, bpm: f64) {
        self.bpm = bpm.clamp(10.0, 999.0);
        self.samples_per_tick_f64 =
            Self::compute_samples_per_tick(self.sample_rate, self.bpm, self.ppq);
    }

    /// Updates PPQ resolution (`24..=960`) dynamically and recalculates `samples_per_tick_f64` (Zero Heap Allocation).
    #[inline]
    pub fn set_ppq(&mut self, ppq: u32) {
        self.ppq = ppq.clamp(MIN_PPQ, MAX_PPQ);
        self.samples_per_tick_f64 =
            Self::compute_samples_per_tick(self.sample_rate, self.bpm, self.ppq);
    }

    /// Updates sample rate and recalculates `samples_per_tick_f64`.
    #[inline]
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate.max(8000.0);
        self.samples_per_tick_f64 =
            Self::compute_samples_per_tick(self.sample_rate, self.bpm, self.ppq);
    }

    /// Converts a PPQ tick position (integer or fractional) to an exact sample position,
    /// accounting for `tempo_map` segments if present (Zero Heap Allocation).
    #[inline]
    pub fn tick_to_sample_pos(&self, tick: f64) -> i64 {
        let target_tick = tick.max(0.0);
        if let Some(ref map) = self.tempo_map {
            let pts = map.active_points();
            if !pts.is_empty() {
                let mut cur_tick = 0.0f64;
                let mut accum_samples = 0.0f64;
                let mut cur_bpm = map.default_bpm;
                for pt in pts {
                    let pt_tick = (pt.ppq_tick.max(0)) as f64;
                    if target_tick <= pt_tick {
                        break;
                    }
                    if pt_tick > cur_tick {
                        let dt = pt_tick - cur_tick;
                        accum_samples +=
                            dt * Self::compute_samples_per_tick(self.sample_rate, cur_bpm, self.ppq);
                        cur_tick = pt_tick;
                    }
                    cur_bpm = pt.bpm;
                }
                if target_tick > cur_tick {
                    accum_samples += (target_tick - cur_tick)
                        * Self::compute_samples_per_tick(self.sample_rate, cur_bpm, self.ppq);
                }
                return accum_samples.round() as i64;
            }
        }
        (target_tick * self.samples_per_tick_f64).round() as i64
    }

    /// Converts a sample position to a continuous PPQ tick position (Zero Heap Allocation).
    #[inline]
    pub fn sample_pos_to_tick(&self, sample_pos: i64) -> f64 {
        let target_samples = (sample_pos.max(0)) as f64;
        if let Some(ref map) = self.tempo_map {
            let pts = map.active_points();
            if !pts.is_empty() {
                let mut cur_tick = 0.0f64;
                let mut cur_samples = 0.0f64;
                let mut cur_bpm = map.default_bpm;
                for pt in pts {
                    let pt_tick = (pt.ppq_tick.max(0)) as f64;
                    if pt_tick > cur_tick {
                        let spt =
                            Self::compute_samples_per_tick(self.sample_rate, cur_bpm, self.ppq);
                        let seg_samples = (pt_tick - cur_tick) * spt;
                        if cur_samples + seg_samples >= target_samples {
                            return cur_tick + (target_samples - cur_samples) / spt.max(1e-9);
                        }
                        cur_samples += seg_samples;
                        cur_tick = pt_tick;
                    }
                    cur_bpm = pt.bpm;
                }
                let spt = Self::compute_samples_per_tick(self.sample_rate, cur_bpm, self.ppq);
                return cur_tick + (target_samples - cur_samples) / spt.max(1e-9);
            }
        }
        target_samples / self.samples_per_tick_f64.max(1e-9)
    }

    /// Resets playback position to sample 0 and tick 0.
    #[inline]
    pub fn reset(&mut self) {
        self.seek(0, 0);
    }

    /// Seeks transport to `(sample_pos, ppq_tick)` while synchronizing `tick_sample_accum_f64`
    /// and `next_tick_sample_pos` so that `ppq_tick` fires immediately at `sample_pos` and
    /// subsequent ticks advance at `samples_per_tick_f64` without backward stall or forward burst.
    #[inline]
    pub fn seek(&mut self, sample_pos: i64, ppq_tick: i64) {
        let clamped_pos = sample_pos.max(0);
        let clamped_tick = ppq_tick.max(0);
        self.current_sample_pos = clamped_pos;
        self.next_tick_sample_pos = clamped_pos;
        self.tick_sample_accum_f64 = clamped_pos as f64;
        self.global_ppq_tick = clamped_tick;
    }

    /// Quantizes a floating-point sample position to the next tick boundary:
    /// - When `simd_align_8_samples` is true (`0x15e1b90`): `((accum / 8.0).round() as i64) << 3` (`0xd0f08b`).
    /// - When false: `accum.round() as i64` (`0xd0eff6`).
    #[inline]
    fn quantize_next_tick_pos(&self) -> i64 {
        if self.simd_align_8_samples {
            let aligned = ((self.tick_sample_accum_f64 / 8.0).round() as i64) << 3;
            if aligned <= self.current_sample_pos {
                ((self.current_sample_pos >> 3) + 1) << 3
            } else {
                aligned
            }
        } else {
            let rounded = self.tick_sample_accum_f64.round() as i64;
            rounded.max(self.current_sample_pos + 1)
        }
    }

    /// Splits a hardware audio block of `total_frames` at exact PPQ Tick boundaries (`sub_d0ece0`).
    ///
    /// - `on_tick(ppq_tick, sample_pos)` is invoked at each PPQ Tick boundary (`0xd0ef28`).
    /// - `render_slice(buffer_offset, slice_len)` is invoked for each sub-block (`sub_d0d4c0`).
    ///
    /// Zero Heap Allocation.
    #[inline]
    pub fn process_buffer<FOnTick, FRenderSlice>(
        &mut self,
        total_frames: usize,
        mut on_tick: FOnTick,
        mut render_slice: FRenderSlice,
    ) where
        FOnTick: FnMut(i64, i64),
        FRenderSlice: FnMut(usize, usize),
    {
        if total_frames == 0 {
            return;
        }
        let target_end_sample = self.current_sample_pos + (total_frames as i64);
        let mut buffer_offset = 0usize;

        while self.current_sample_pos < target_end_sample {
            if self.current_sample_pos >= self.next_tick_sample_pos {
                if let Some(ref map) = self.tempo_map {
                    let new_bpm = map.bpm_at_tick(self.global_ppq_tick);
                    if (new_bpm - self.bpm).abs() > 1e-9 {
                        self.set_bpm(new_bpm);
                    }
                }
                on_tick(self.global_ppq_tick, self.current_sample_pos);
                self.tick_sample_accum_f64 += self.samples_per_tick_f64;
                self.next_tick_sample_pos = self.quantize_next_tick_pos();
                self.global_ppq_tick += 1;
            }

            let slice_end = self.next_tick_sample_pos.min(target_end_sample);
            let slice_len = (slice_end - self.current_sample_pos).max(1) as usize;
            render_slice(buffer_offset, slice_len);

            self.current_sample_pos += slice_len as i64;
            buffer_offset += slice_len;
        }
    }

    /// Stateful variant of [`Self::process_buffer`] that passes `&mut S` into both callbacks
    /// so callers can mutate a single state struct without conflicting closure borrows.
    #[inline]
    pub fn process_buffer_state<S, FOnTick, FRenderSlice>(
        &mut self,
        state: &mut S,
        total_frames: usize,
        mut on_tick: FOnTick,
        mut render_slice: FRenderSlice,
    ) where
        FOnTick: FnMut(&mut S, i64, i64, f64),
        FRenderSlice: FnMut(&mut S, usize, usize, i64, f64),
    {
        if total_frames == 0 {
            return;
        }
        let target_end_sample = self.current_sample_pos + (total_frames as i64);
        let mut buffer_offset = 0usize;

        while self.current_sample_pos < target_end_sample {
            if self.current_sample_pos >= self.next_tick_sample_pos {
                if let Some(ref map) = self.tempo_map {
                    let new_bpm = map.bpm_at_tick(self.global_ppq_tick);
                    if (new_bpm - self.bpm).abs() > 1e-9 {
                        self.set_bpm(new_bpm);
                    }
                }
                on_tick(state, self.global_ppq_tick, self.current_sample_pos, self.bpm);
                self.tick_sample_accum_f64 += self.samples_per_tick_f64;
                self.next_tick_sample_pos = self.quantize_next_tick_pos();
                self.global_ppq_tick += 1;
            }

            let slice_end = self.next_tick_sample_pos.min(target_end_sample);
            let slice_len = (slice_end - self.current_sample_pos).max(1) as usize;
            let active_tick = (self.global_ppq_tick - 1).max(0);
            render_slice(state, buffer_offset, slice_len, active_tick, self.bpm);

            self.current_sample_pos += slice_len as i64;
            buffer_offset += slice_len;
        }
    }
}
