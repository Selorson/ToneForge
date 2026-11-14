//! Project audio assets: every piece of audio the engine plays from disk (loaded songs, UVR5 stems,
//! committed NewTone vocals, recorded takes) is described by an [`AudioAssetDto`] persisted in
//! [`ToneForgeProjectState::audio_assets`]. The same description rebuilds the live real-time graph,
//! the offline render graph and a re-opened project, so all three play identical audio.
//!
//! Playlist `AudioStem` clips whose `source_id` equals an asset id become sample-accurate
//! [`StemClipRegion`]s of that asset's `AudioStemClipPlayerNode` (move / slice / slip / fade / gain /
//! mute all audible). Clips with a non-unity `stretch_ratio` or `pitch_semitones` are rendered
//! offline with `TransientLockedTimeStretcher` into a derived player (no varispeed).

use crate::project_state::{PlaylistClipDto, ToneForgeProjectState};
use daw_core_graph::{
    AudioStemClipPlayerNode, ChannelGenerator, MAX_BLOCK_SIZE, MAX_STEM_CLIP_REGIONS, StereoFrame,
    StemClipRegion, UnifiedDspNode,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Asset id of the song loaded with "Open audio" (continuous player on Insert 1).
pub const ASSET_ID_CUSTOM_AUDIO: u32 = 900;
/// Asset ids `101..=104` hold the 4 UVR5 stems bounced to Inserts `1..=4`.
pub const ASSET_ID_UVR5_BASE: u32 = 100;
/// Committed NewTone vocal on Insert `t` uses asset id `ASSET_ID_NEWTONE_BASE + t`.
pub const ASSET_ID_NEWTONE_BASE: u32 = 9_000;
/// Recorded takes use asset ids `ASSET_ID_TAKE_BASE..`.
pub const ASSET_ID_TAKE_BASE: u32 = 10_000;
/// Stretched / pitched clip renders use channel id `ASSET_ID_DERIVED_BASE + clip_id`.
pub const ASSET_ID_DERIVED_BASE: u32 = 1_000_000;

pub const KIND_CUSTOM_AUDIO: &str = "CustomAudio";
pub const KIND_UVR5_STEM: &str = "Uvr5Stem";
pub const KIND_NEWTONE_COMMIT: &str = "NewToneCommit";
pub const KIND_RECORDED_TAKE: &str = "RecordedTake";

/// Serializable description of one audio asset used by the project.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioAssetDto {
    /// Channel id of the asset's player in the mixer graph (also the `source_id` of its clips).
    pub asset_id: u32,
    /// `CustomAudio` | `Uvr5Stem` | `NewToneCommit` | `RecordedTake`.
    pub kind: String,
    pub name: String,
    /// Absolute path of the audio file on disk (WAV / FLAC / MP3 / ...).
    pub file_path: String,
    /// Mixer insert the player feeds.
    pub target_track: u16,
    /// `true`: plays continuously from timeline sample 0 (songs, stems, commits).
    /// `false`: plays only through its Playlist `AudioStem` clips (recorded takes).
    pub continuous: bool,
    #[serde(default)]
    pub loop_playback: bool,
}

impl AudioAssetDto {
    fn player_name(&self) -> &'static str {
        match self.kind.as_str() {
            KIND_CUSTOM_AUDIO => "Custom Audio Stem",
            KIND_UVR5_STEM => "UVR5 Separated Stem Playback Generator",
            KIND_NEWTONE_COMMIT => "NewTone Vocal Commit",
            _ => "Recorded Take Audio",
        }
    }
}

/// Persistent folder for audio the application creates (takes, stems, committed vocals):
/// `%USERPROFILE%\Documents\ToneForge\<sub>` (falls back to the executable's folder).
pub fn toneforge_user_audio_dir(sub: &str) -> PathBuf {
    let base = std::env::var_os("USERPROFILE")
        .map(|p| PathBuf::from(p).join("Documents").join("ToneForge"))
        .or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|e| e.parent().map(|p| p.join("ToneForge")))
        })
        .unwrap_or_else(|| PathBuf::from("ToneForge"));
    let dir = base.join(sub);
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Decodes `asset.file_path` and resamples it to `engine_sample_rate_hz`.
pub fn load_asset_frames(
    asset: &AudioAssetDto,
    engine_sample_rate_hz: u32,
) -> Result<Vec<StereoFrame>, String> {
    let decoded = crate::decode_audio_file_native(Path::new(&asset.file_path))
        .map_err(|e| format!("Audio asset '{}' ({}): {e}", asset.name, asset.file_path))?;
    if decoded.sample_rate_hz == engine_sample_rate_hz {
        Ok(decoded.frames)
    } else {
        Ok(crate::resample_stereo_frames(
            &decoded.frames,
            decoded.sample_rate_hz,
            engine_sample_rate_hz,
        ))
    }
}

/// Engine samples per PPQ tick for the project's tempo and resolution.
#[inline]
pub fn samples_per_tick(ps: &ToneForgeProjectState, sample_rate: f64) -> f64 {
    sample_rate * 60.0 / (ps.bpm.max(1.0) * (ps.ppq_resolution.max(24) as f64))
}

/// `true` when a clip needs a time-stretched / pitch-shifted render instead of direct playback.
#[inline]
pub fn clip_needs_derived_render(clip: &PlaylistClipDto) -> bool {
    (clip.stretch_ratio - 1.0).abs() > 1e-4 || clip.pitch_semitones.abs() > 1e-3
}

fn region_for_clip(clip: &PlaylistClipDto, spt: f64, source_offset: usize) -> StemClipRegion {
    let start = ((clip.start_tick as f64) * spt).round() as i64;
    let dur = ((clip.duration_ticks.max(1) as f64) * spt).round().max(1.0) as usize;
    StemClipRegion {
        clip_id: clip.clip_id,
        timeline_start_sample: start,
        timeline_duration_samples: dur,
        source_offset_samples: source_offset,
        fade_in_samples: clip.fade_in_samples.min(dur),
        fade_out_samples: clip.fade_out_samples.min(dur),
        gain: clip.gain,
        gain_end: clip.gain,
        pan: 0.0,
        pitch_semitones: 0.0,
        stretch_ratio: 1.0,
        equal_power_fades: true,
        take_lane_id: clip.take_lane_id,
    }
}

/// Source offset (in source samples) addressed by a clip: slip offset + clip offset in ticks.
#[inline]
fn clip_source_offset(clip: &PlaylistClipDto, spt: f64) -> usize {
    clip.source_offset_samples + ((clip.clip_offset_ticks.max(0) as f64) * spt).round() as usize
}

/// Builds the pre-sized (`MAX_STEM_CLIP_REGIONS`) region table for a clip-driven asset.
/// Returns `(table, count)`.
pub fn asset_region_table(
    ps: &ToneForgeProjectState,
    asset_id: u32,
    sample_rate: f64,
) -> (Vec<StemClipRegion>, usize) {
    let spt = samples_per_tick(ps, sample_rate);
    let mut table = vec![StemClipRegion::default(); MAX_STEM_CLIP_REGIONS];
    let mut count = 0usize;
    for clip in &ps.playlist_clips {
        if count >= MAX_STEM_CLIP_REGIONS {
            break;
        }
        if clip.kind != "AudioStem"
            || clip.source_id != asset_id
            || clip.muted
            || clip_needs_derived_render(clip)
        {
            continue;
        }
        table[count] = region_for_clip(clip, spt, clip_source_offset(clip, spt));
        count += 1;
    }
    (table, count)
}

/// Builds the player channel for `asset` from `frames` (already at `sample_rate`).
pub fn build_asset_channel(
    asset: &AudioAssetDto,
    frames: Vec<StereoFrame>,
    ps: &ToneForgeProjectState,
    sample_rate: f64,
) -> ChannelGenerator {
    build_asset_channel_shared(asset, std::sync::Arc::new(frames), ps, sample_rate)
}

/// [`build_asset_channel`] playing `frames` shared with the asset cache (no copy).
pub fn build_asset_channel_shared(
    asset: &AudioAssetDto,
    frames: std::sync::Arc<Vec<StereoFrame>>,
    ps: &ToneForgeProjectState,
    sample_rate: f64,
) -> ChannelGenerator {
    let mut player = AudioStemClipPlayerNode::new_shared(asset.asset_id, asset.player_name(), frames);
    if asset.continuous {
        player.continuous_mode = true;
        player.loop_enabled = asset.loop_playback;
    } else {
        let (mut table, count) = asset_region_table(ps, asset.asset_id, sample_rate);
        player.swap_stem_clip_regions(&mut table, count);
    }
    player.prepare(sample_rate, MAX_BLOCK_SIZE);
    let mut ch = ChannelGenerator::new(asset.asset_id, asset.target_track as usize, Box::new(player));
    ch.prepare(sample_rate, MAX_BLOCK_SIZE);
    ch
}

/// Identity of a stretched / pitch-shifted render: everything that changes the rendered audio and
/// nothing that only changes playback (start tick, gain, fades, mute are applied by the region table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DerivedRenderKey {
    pub asset_id: u32,
    pub src_start: usize,
    pub src_len: usize,
    pub ratio_bits: u32,
    pub pitch_bits: u32,
    pub sample_rate_hz: u32,
}

impl DerivedRenderKey {
    #[inline]
    pub fn ratio(&self) -> f32 {
        f32::from_bits(self.ratio_bits)
    }

    #[inline]
    pub fn pitch_semitones(&self) -> f32 {
        f32::from_bits(self.pitch_bits)
    }
}

/// Render key of `clip` addressing a source of `source_len` frames at `sample_rate`.
pub fn derived_render_key(
    clip: &PlaylistClipDto,
    source_len: usize,
    ps: &ToneForgeProjectState,
    sample_rate: f64,
) -> DerivedRenderKey {
    let spt = samples_per_tick(ps, sample_rate);
    let ratio = clip.stretch_ratio.clamp(0.25, 4.0);
    let dur = ((clip.duration_ticks.max(1) as f64) * spt).round().max(1.0) as usize;
    let src_start = clip_source_offset(clip, spt).min(source_len);
    let src_len = (((dur as f64) / (ratio as f64)).ceil() as usize + 1).min(source_len - src_start);
    // `+ 0.0` folds `-0.0` into `0.0` so equal pitches always hash equally.
    let pitch = clip.pitch_semitones + 0.0;
    DerivedRenderKey {
        asset_id: clip.source_id,
        src_start,
        src_len,
        ratio_bits: ratio.to_bits(),
        pitch_bits: pitch.to_bits(),
        sample_rate_hz: sample_rate.round().max(1.0) as u32,
    }
}

/// Runs the `TransientLockedTimeStretcher` for `key` over `source`; `None` when `cancelled` fired.
pub fn render_derived_clip_audio(
    key: &DerivedRenderKey,
    source: &[StereoFrame],
    cancelled: &dyn Fn() -> bool,
) -> Option<Vec<StereoFrame>> {
    let start = key.src_start.min(source.len());
    let end = (start + key.src_len).min(source.len());
    let mut stretcher = dsp_pitch_time::TransientLockedTimeStretcher::new(key.sample_rate_hz as f64);
    stretcher.stretch_stereo_offline_cancellable(
        &source[start..end],
        key.ratio(),
        key.pitch_semitones(),
        cancelled,
    )
}

/// Region table of a derived (rendered) clip player: one region, empty when the clip is muted.
pub fn derived_clip_region_table(
    clip: &PlaylistClipDto,
    ps: &ToneForgeProjectState,
    sample_rate: f64,
) -> (Vec<StemClipRegion>, usize) {
    let spt = samples_per_tick(ps, sample_rate);
    let mut table = vec![StemClipRegion::default(); MAX_STEM_CLIP_REGIONS];
    let mut count = 0usize;
    if !clip.muted {
        table[0] = region_for_clip(clip, spt, 0);
        count = 1;
    }
    (table, count)
}

/// Channel id of the derived player of `clip_id`.
#[inline]
pub fn derived_channel_id(clip_id: u32) -> u32 {
    ASSET_ID_DERIVED_BASE + clip_id
}

/// Builds the derived player channel of `clip_id` around an already rendered (shared) buffer and
/// a region table from [`derived_clip_region_table`].
pub fn build_derived_clip_channel_with_regions(
    clip_id: u32,
    rendered: std::sync::Arc<Vec<StereoFrame>>,
    target_track: u16,
    mut table: Vec<StemClipRegion>,
    count: usize,
    sample_rate: f64,
) -> ChannelGenerator {
    let mut player =
        AudioStemClipPlayerNode::new_shared(derived_channel_id(clip_id), "Stretched Audio Clip", rendered);
    player.swap_stem_clip_regions(&mut table, count);
    player.prepare(sample_rate, MAX_BLOCK_SIZE);
    let mut ch = ChannelGenerator::new(derived_channel_id(clip_id), target_track as usize, Box::new(player));
    ch.prepare(sample_rate, MAX_BLOCK_SIZE);
    ch
}

/// Builds the derived player channel of `clip` around an already rendered buffer.
pub fn build_derived_clip_channel_from_render(
    clip: &PlaylistClipDto,
    rendered: std::sync::Arc<Vec<StereoFrame>>,
    target_track: u16,
    ps: &ToneForgeProjectState,
    sample_rate: f64,
) -> ChannelGenerator {
    let (table, count) = derived_clip_region_table(clip, ps, sample_rate);
    build_derived_clip_channel_with_regions(clip.clip_id, rendered, target_track, table, count, sample_rate)
}

/// Renders a stretched / pitch-shifted clip into its own player (`ASSET_ID_DERIVED_BASE + clip_id`)
/// with `TransientLockedTimeStretcher` (tempo and pitch independent; transients locked).
pub fn build_derived_clip_channel(
    clip: &PlaylistClipDto,
    source: &[StereoFrame],
    target_track: u16,
    ps: &ToneForgeProjectState,
    sample_rate: f64,
) -> ChannelGenerator {
    let key = derived_render_key(clip, source.len(), ps, sample_rate);
    let rendered = std::sync::Arc::new(render_derived_clip_audio(&key, source, &|| false).unwrap_or_default());
    build_derived_clip_channel_from_render(clip, rendered, target_track, ps, sample_rate)
}

/// Clip-driven assets (not continuous) referenced by `ps`.
pub fn clip_driven_asset_ids(ps: &ToneForgeProjectState) -> Vec<u32> {
    ps.audio_assets
        .iter()
        .filter(|a| !a.continuous)
        .map(|a| a.asset_id)
        .collect()
}

/// Next free recorded-take asset id.
pub fn next_take_asset_id(ps: &ToneForgeProjectState) -> u32 {
    ps.audio_assets
        .iter()
        .map(|a| a.asset_id)
        .filter(|&id| (ASSET_ID_TAKE_BASE..ASSET_ID_DERIVED_BASE).contains(&id))
        .max()
        .map(|m| m + 1)
        .unwrap_or(ASSET_ID_TAKE_BASE)
}

/// Inserts or replaces `asset` (matched by `asset_id`) in the project.
pub fn upsert_asset(ps: &mut ToneForgeProjectState, asset: AudioAssetDto) {
    if let Some(existing) = ps
        .audio_assets
        .iter_mut()
        .find(|a| a.asset_id == asset.asset_id)
    {
        *existing = asset;
    } else {
        ps.audio_assets.push(asset);
    }
}

/// Builds every asset channel (and derived clip channels) for a fresh engine graph — used by the
/// offline render and by project load. `frames_for` returns the asset audio at `sample_rate`.
pub fn build_all_asset_channels<F>(
    ps: &ToneForgeProjectState,
    sample_rate: f64,
    mut frames_for: F,
) -> Result<Vec<ChannelGenerator>, String>
where
    F: FnMut(&AudioAssetDto) -> Result<std::sync::Arc<Vec<StereoFrame>>, String>,
{
    let mut out = Vec::new();
    for asset in &ps.audio_assets {
        let frames = frames_for(asset)?;
        for clip in &ps.playlist_clips {
            if clip.kind == "AudioStem"
                && clip.source_id == asset.asset_id
                && !asset.continuous
                && clip_needs_derived_render(clip)
            {
                out.push(build_derived_clip_channel(
                    clip,
                    &frames,
                    asset.target_track,
                    ps,
                    sample_rate,
                ));
            }
        }
        out.push(build_asset_channel_shared(asset, frames, ps, sample_rate));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_table_maps_clips_to_sample_accurate_regions() {
        let mut ps = ToneForgeProjectState::new_default(48_000.0, 120.0);
        ps.ppq_resolution = 96;
        ps.playlist_clips.push(PlaylistClipDto {
            clip_id: 7,
            kind: "AudioStem".to_string(),
            source_id: ASSET_ID_TAKE_BASE,
            start_tick: 96,
            duration_ticks: 48,
            fade_in_samples: 64,
            source_offset_samples: 100,
            ..PlaylistClipDto::default()
        });
        let (table, count) = asset_region_table(&ps, ASSET_ID_TAKE_BASE, 48_000.0);
        assert_eq!(count, 1);
        assert_eq!(table.len(), MAX_STEM_CLIP_REGIONS);
        // 120 BPM, 96 PPQ @ 48 kHz -> 250 samples per tick.
        assert_eq!(table[0].timeline_start_sample, 24_000);
        assert_eq!(table[0].timeline_duration_samples, 12_000);
        assert_eq!(table[0].source_offset_samples, 100);
        assert_eq!(table[0].fade_in_samples, 64);
    }
}
