//! `collision_bus`: 100% Lock-Free Cross-Track Spectral Collision Bus for `FabFilter Pro-Q 4`.
//!
//! Stores 502 Mixer Track slots × 1,024 logarithmic frequency bins (`20 Hz .. 20 kHz`) using
//! `AtomicU32` (`f32::to_bits`) and `AtomicU64` sequence epochs (`0` `Mutex`, `0` heap allocations
//! on the real-time audio thread). Supports both all-instance collision detection and Pro-Q 4
//! Instance List **Reference Track** masking comparison.

use crate::auto_gain_512::SPECTRUM_FRAME_BINS;
use daw_core_graph::MAX_MIXER_TRACKS;
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

pub const MAX_COLLISION_SLOTS: usize = MAX_MIXER_TRACKS;
pub const NO_REFERENCE_TRACK: u16 = u16::MAX;

const SILENT_DB_BITS: u32 = (-120.0f32).to_bits();
const COLLISION_FLOOR_DB: f32 = -60.0;
const COLLISION_SPAN_DB: f32 = 66.0; // -60 dBFS -> 0.0, +6 dBFS -> 1.0 (strictly monotonic, no flat-top plateau)

/// Lock-free 502-track × 1024-bin spectral collision bus (`0` `Mutex`, `0` RT allocation).
pub struct ProQ4CollisionBus {
    spectra_bits: Box<[[AtomicU32; SPECTRUM_FRAME_BINS]]>,
    active_epoch: Box<[AtomicU64]>,
    reference_track: Box<[AtomicU16]>,
    epoch_seq: AtomicU64,
}

impl Default for ProQ4CollisionBus {
    fn default() -> Self {
        Self::new()
    }
}

impl ProQ4CollisionBus {
    pub fn new() -> Self {
        let mut spectra_vec = Vec::with_capacity(MAX_COLLISION_SLOTS);
        let mut epoch_vec = Vec::with_capacity(MAX_COLLISION_SLOTS);
        let mut ref_vec = Vec::with_capacity(MAX_COLLISION_SLOTS);

        for _ in 0..MAX_COLLISION_SLOTS {
            spectra_vec.push(std::array::from_fn(|_| AtomicU32::new(SILENT_DB_BITS)));
            epoch_vec.push(AtomicU64::new(0));
            ref_vec.push(AtomicU16::new(NO_REFERENCE_TRACK));
        }

        Self {
            spectra_bits: spectra_vec.into_boxed_slice(),
            active_epoch: epoch_vec.into_boxed_slice(),
            reference_track: ref_vec.into_boxed_slice(),
            epoch_seq: AtomicU64::new(1),
        }
    }

    pub fn new_arc() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Global singleton bus used by default when a `ProQ4Processor` is not explicitly bound to
    /// an isolated `MixerEngine` collision bus.
    pub fn global() -> &'static Arc<Self> {
        static GLOBAL_BUS: OnceLock<Arc<ProQ4CollisionBus>> = OnceLock::new();
        GLOBAL_BUS.get_or_init(Self::new_arc)
    }

    #[inline]
    fn map_slot(instance_or_track_id: u32) -> usize {
        (instance_or_track_id as usize) % MAX_COLLISION_SLOTS
    }

    /// Assigns an optional Reference Track (`Some(0..502)`) or `None` (compare against all other active tracks)
    /// for `observer_track`.
    #[inline]
    pub fn set_reference_track(&self, observer_track: u32, reference: Option<u16>) {
        let slot = Self::map_slot(observer_track);
        let val = reference
            .filter(|&r| (r as usize) < MAX_COLLISION_SLOTS)
            .unwrap_or(NO_REFERENCE_TRACK);
        self.reference_track[slot].store(val, Ordering::Relaxed);
    }

    #[inline]
    pub fn reference_track(&self, observer_track: u32) -> Option<u16> {
        let slot = Self::map_slot(observer_track);
        let v = self.reference_track[slot].load(Ordering::Relaxed);
        if v == NO_REFERENCE_TRACK {
            None
        } else {
            Some(v)
        }
    }

    /// Clears a single track slot from active collision comparison.
    pub fn clear_track(&self, track_id: u32) {
        let slot = Self::map_slot(track_id);
        self.active_epoch[slot].store(0, Ordering::Release);
        for bin in &self.spectra_bits[slot] {
            bin.store(SILENT_DB_BITS, Ordering::Relaxed);
        }
    }

    /// Clears all 502 track slots.
    pub fn reset_all(&self) {
        for slot in 0..MAX_COLLISION_SLOTS {
            self.active_epoch[slot].store(0, Ordering::Release);
            self.reference_track[slot].store(NO_REFERENCE_TRACK, Ordering::Relaxed);
            for bin in &self.spectra_bits[slot] {
                bin.store(SILENT_DB_BITS, Ordering::Relaxed);
            }
        }
    }

    /// Wait-free publication of a track's 1024-bin post-EQ spectrum (safe on RT audio thread).
    #[inline]
    pub fn publish_track_spectrum(
        &self,
        track_id: u32,
        post_db: &[f32; SPECTRUM_FRAME_BINS],
    ) {
        let slot = Self::map_slot(track_id);
        let row = &self.spectra_bits[slot];
        for b in 0..SPECTRUM_FRAME_BINS {
            row[b].store(post_db[b].to_bits(), Ordering::Relaxed);
        }
        let seq = self.epoch_seq.fetch_add(1, Ordering::Relaxed);
        self.active_epoch[slot].store(seq.max(1), Ordering::Release);
    }

    /// Computes the 1,024-bin spectral masking collision heatmask in `[0.0, 1.0]` for `observer_track`
    /// without locks or allocations.
    pub fn compute_collisions_for_track(
        &self,
        observer_track: u32,
        self_post_db: &[f32; SPECTRUM_FRAME_BINS],
        out_collision_mask: &mut [f32; SPECTRUM_FRAME_BINS],
    ) {
        let self_slot = Self::map_slot(observer_track);
        out_collision_mask.fill(0.0);

        let ref_setting = self.reference_track[self_slot].load(Ordering::Relaxed);
        if ref_setting != NO_REFERENCE_TRACK {
            let ref_slot = (ref_setting as usize) % MAX_COLLISION_SLOTS;
            if ref_slot != self_slot && self.active_epoch[ref_slot].load(Ordering::Acquire) != 0 {
                self.accumulate_pair_collision(ref_slot, self_post_db, out_collision_mask);
            }
            return;
        }

        // Compare against all recently active other tracks
        let newest = self.epoch_seq.load(Ordering::Relaxed);
        let stale_window = 4096u64;
        for other_slot in 0..MAX_COLLISION_SLOTS {
            if other_slot == self_slot {
                continue;
            }
            let ep = self.active_epoch[other_slot].load(Ordering::Acquire);
            if ep == 0 || newest.saturating_sub(ep) > stale_window {
                continue;
            }
            self.accumulate_pair_collision(other_slot, self_post_db, out_collision_mask);
        }
    }

    #[inline]
    pub fn publish_and_compute_collisions(
        &self,
        observer_track: u32,
        self_post_db: &[f32; SPECTRUM_FRAME_BINS],
        out_collision_mask: &mut [f32; SPECTRUM_FRAME_BINS],
    ) {
        self.publish_track_spectrum(observer_track, self_post_db);
        self.compute_collisions_for_track(observer_track, self_post_db, out_collision_mask);
    }

    #[inline]
    fn accumulate_pair_collision(
        &self,
        other_slot: usize,
        self_post_db: &[f32; SPECTRUM_FRAME_BINS],
        out_collision_mask: &mut [f32; SPECTRUM_FRAME_BINS],
    ) {
        let other_row = &self.spectra_bits[other_slot];
        let inv_span = 1.0 / COLLISION_SPAN_DB;
        for b in 0..SPECTRUM_FRAME_BINS {
            let other_db = f32::from_bits(other_row[b].load(Ordering::Relaxed));
            let s = ((self_post_db[b] - COLLISION_FLOOR_DB) * inv_span).clamp(0.0, 1.0);
            let o = ((other_db - COLLISION_FLOOR_DB) * inv_span).clamp(0.0, 1.0);
            if s > 0.0 && o > 0.0 {
                // Smooth strictly-monotonic masking overlap peaking sharply where both tracks collide
                let min_so = s.min(o);
                let geo_so = (s * o).sqrt();
                let overlap = (0.65 * min_so + 0.35 * geo_so).clamp(0.0, 1.0);
                if overlap > out_collision_mask[b] {
                    out_collision_mask[b] = overlap;
                }
            }
        }
    }

    /// Helper converting a bin index `0..1024` back to its center frequency in Hz (`20 Hz .. 20 kHz`).
    #[inline]
    pub fn bin_to_freq_hz(bin_index: usize) -> f32 {
        let min_oct = 20.0f32.log2();
        let max_oct = 20_000.0f32.log2();
        let step = (max_oct - min_oct) / ((SPECTRUM_FRAME_BINS - 1) as f32);
        (min_oct + (bin_index.min(SPECTRUM_FRAME_BINS - 1) as f32) * step).exp2()
    }

    /// Helper converting a frequency in Hz (`20 Hz .. 20 kHz`) to the nearest `0..1024` spectrum bin.
    #[inline]
    pub fn freq_hz_to_bin(freq_hz: f32) -> usize {
        let min_oct = 20.0f32.log2();
        let max_oct = 20_000.0f32.log2();
        let norm = ((freq_hz.clamp(20.0, 20_000.0).log2() - min_oct) / (max_oct - min_oct))
            .clamp(0.0, 1.0);
        (norm * ((SPECTRUM_FRAME_BINS - 1) as f32)).round() as usize
    }
}
