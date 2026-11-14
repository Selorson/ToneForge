//! Placeholder for an unoccupied FX insert slot (FL Studio-style fixed 10-slot rack).
//!
//! Mixer insert slots are fixed positions `0..MAX_INSERT_SLOTS`. Removing a plug-in never shifts the
//! later slots down; the vacated position is filled with an [`EmptySlotNode`] so every key that
//! addresses a plug-in by `(track_index, slot_index)` (automation, controller links, presets, editor
//! windows, Patcher canvases) keeps pointing at the same plug-in.

use crate::{ProcessContext, StereoFrame, UnifiedDspNode};

/// Display / telemetry name reported by [`EmptySlotNode`].
pub const EMPTY_SLOT_NODE_NAME: &str = "Empty";

/// Zero-latency pass-through node standing in for an empty insert slot.
///
/// `process_slice` leaves the buffer untouched and `set_parameter_normalized` ignores every
/// parameter, so stale automation / controller links that still target the slot are harmless.
#[derive(Debug, Default, Clone, Copy)]
pub struct EmptySlotNode;

impl EmptySlotNode {
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl UnifiedDspNode for EmptySlotNode {
    fn name(&self) -> &'static str {
        EMPTY_SLOT_NODE_NAME
    }

    fn latency_samples(&self) -> usize {
        0
    }

    fn prepare(&mut self, _sample_rate: f64, _max_block_size: usize) {}

    fn reset_state(&mut self) {}

    fn set_parameter_normalized(&mut self, _param_id: u32, _norm_val: f32) {}

    fn process_slice(
        &mut self,
        _ctx: &ProcessContext,
        _main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
    }
}
