//! Background renders of stretched / pitch-shifted Playlist clips.
//!
//! A phase-vocoder render of a 3.5-minute stem costs 5–12 s of CPU, so it must never run on the
//! Tauri main thread nor be repeated for edits that do not change the rendered audio:
//! - renders are cached by [`RenderCacheKey`] (source buffer identity + source range + ratio +
//!   pitch + sample rate); start tick, gain, fades and mute only change the player's region table;
//! - every clip carries a generation counter: scheduling a new render (or deleting the clip)
//!   bumps it, and a superseded render stops at its next cancellation poll instead of finishing
//!   and doubling the ~85 MB working memory.

use daw_core_graph::StereoFrame;
use daw_tauri_host::audio_assets::DerivedRenderKey;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

/// Renders no longer used by the Playlist kept for instant undo / A-B toggles (bytes).
const SPARE_CACHE_BUDGET_BYTES: usize = 768 * 1024 * 1024;

/// Cache identity of a render: the render key plus the identity of the source buffer, so a
/// re-decoded or replaced asset (new `Arc`) never reuses a render of the old audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderCacheKey {
    pub render: DerivedRenderKey,
    source_tag: usize,
}

impl RenderCacheKey {
    pub fn new(render: DerivedRenderKey, source: &Arc<Vec<StereoFrame>>) -> Self {
        Self {
            render,
            source_tag: Arc::as_ptr(source) as usize,
        }
    }
}

struct CacheEntry {
    frames: Arc<Vec<StereoFrame>>,
    /// Keeps the source allocation's address reserved (no ABA on `source_tag`) without keeping
    /// its samples alive; a dead source invalidates the entry.
    source: Weak<Vec<StereoFrame>>,
    last_use: u64,
}

/// A render in flight for one clip.
pub struct PendingRender {
    pub key: RenderCacheKey,
    pub target_track: u16,
}

/// Work item executed on a background thread.
pub struct RenderJob {
    pub clip_id: u32,
    pub key: RenderCacheKey,
    pub target_track: u16,
    pub generation: u64,
    pub generation_counter: Arc<AtomicU64>,
    pub source: Arc<Vec<StereoFrame>>,
    /// Already rendered audio (cache hit): only the player channel has to be built.
    pub cached: Option<Arc<Vec<StereoFrame>>>,
    /// Region table at scheduling time, used if the live project no longer matches (e.g. a
    /// project being opened is published after the render finishes).
    pub fallback_regions: (Vec<daw_core_graph::StemClipRegion>, usize),
}

impl RenderJob {
    /// `false` once a newer render was scheduled for this clip or the clip was removed.
    #[inline]
    pub fn is_current(&self) -> bool {
        self.generation_counter.load(Ordering::Acquire) == self.generation
    }
}

/// Render cache + per-clip live / pending bookkeeping (guarded by one mutex in the controller;
/// lock order: `project_state` -> `derived`).
#[derive(Default)]
pub struct DerivedRenderState {
    cache: HashMap<RenderCacheKey, CacheEntry>,
    /// Render currently loaded in the engine for each clip (key, target track).
    pub live: HashMap<u32, (RenderCacheKey, u16)>,
    /// Render being computed for each clip.
    pub pending: HashMap<u32, PendingRender>,
    generations: HashMap<u32, Arc<AtomicU64>>,
    tick: u64,
}

impl DerivedRenderState {
    /// Invalidates any in-flight render of `clip_id`; returns the new generation and its counter.
    pub fn bump_generation(&mut self, clip_id: u32) -> (u64, Arc<AtomicU64>) {
        let counter = Arc::clone(
            self.generations
                .entry(clip_id)
                .or_insert_with(|| Arc::new(AtomicU64::new(0))),
        );
        let generation = counter.fetch_add(1, Ordering::AcqRel) + 1;
        (generation, counter)
    }

    /// Cancels the in-flight render of `clip_id`, if any.
    pub fn cancel(&mut self, clip_id: u32) {
        if self.pending.remove(&clip_id).is_some() {
            self.bump_generation(clip_id);
        }
    }

    /// Cached render for `key` (refreshing its LRU stamp).
    pub fn lookup(&mut self, key: &RenderCacheKey) -> Option<Arc<Vec<StereoFrame>>> {
        self.tick += 1;
        let tick = self.tick;
        match self.cache.get_mut(key) {
            Some(e) if e.source.strong_count() > 0 => {
                e.last_use = tick;
                Some(Arc::clone(&e.frames))
            }
            Some(_) => {
                self.cache.remove(key);
                None
            }
            None => None,
        }
    }

    pub fn insert(
        &mut self,
        key: RenderCacheKey,
        frames: Arc<Vec<StereoFrame>>,
        source: &Arc<Vec<StereoFrame>>,
    ) {
        self.tick += 1;
        self.cache.insert(
            key,
            CacheEntry {
                frames,
                source: Arc::downgrade(source),
                last_use: self.tick,
            },
        );
    }

    /// Drops renders of dead sources and least-recently-used renders not referenced by the
    /// Playlist beyond [`SPARE_CACHE_BUDGET_BYTES`].
    pub fn evict(&mut self) {
        self.cache.retain(|_, e| e.source.strong_count() > 0);
        let in_use: HashSet<RenderCacheKey> = self
            .live
            .values()
            .map(|(k, _)| *k)
            .chain(self.pending.values().map(|p| p.key))
            .collect();
        let mut spare: Vec<(u64, RenderCacheKey, usize)> = self
            .cache
            .iter()
            .filter(|(k, _)| !in_use.contains(k))
            .map(|(k, e)| (e.last_use, *k, e.frames.len() * std::mem::size_of::<StereoFrame>()))
            .collect();
        spare.sort_by(|a, b| b.0.cmp(&a.0));
        let mut kept = 0usize;
        for (_, key, bytes) in spare {
            kept += bytes;
            if kept > SPARE_CACHE_BUDGET_BYTES {
                self.cache.remove(&key);
            }
        }
    }

    /// Number of clips whose render is still being computed.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    #[cfg(test)]
    fn cache_len(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(asset_id: u32, src: &Arc<Vec<StereoFrame>>) -> RenderCacheKey {
        RenderCacheKey::new(
            DerivedRenderKey {
                asset_id,
                src_start: 0,
                src_len: src.len(),
                ratio_bits: 1.05f32.to_bits(),
                pitch_bits: 0.0f32.to_bits(),
                sample_rate_hz: 48_000,
            },
            src,
        )
    }

    #[test]
    fn generation_bump_supersedes_previous_job() {
        let mut st = DerivedRenderState::default();
        let (g1, c1) = st.bump_generation(7);
        let (g2, _) = st.bump_generation(7);
        assert!(g2 > g1);
        assert_ne!(c1.load(Ordering::Acquire), g1, "old job must see itself superseded");
    }

    #[test]
    fn cache_entry_dies_with_its_source() {
        let mut st = DerivedRenderState::default();
        let src = Arc::new(vec![[0.0f32; 2]; 16]);
        let k = key(1, &src);
        st.insert(k, Arc::new(vec![[0.0; 2]; 17]), &src);
        assert!(st.lookup(&k).is_some());
        drop(src);
        assert!(st.lookup(&k).is_none());
        assert_eq!(st.cache_len(), 0);
    }

    #[test]
    fn replaced_source_is_a_different_key() {
        let a = Arc::new(vec![[0.0f32; 2]; 16]);
        let b = Arc::new(vec![[0.0f32; 2]; 16]);
        assert_ne!(key(1, &a), key(1, &b));
    }
}
