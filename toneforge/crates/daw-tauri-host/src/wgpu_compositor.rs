//! WGPU Multi-Viewport 120 FPS Compositor & Uniform Packers (`wgpu_compositor.rs`).
//!
//! Provides `#[repr(C)]` `Pod` uniform and storage instance structures for the 120 FPS WGSL shaders:
//! - `proq4_curve.wgsl` / `reduction_eq_curve.wgsl` (`BiquadCascadeUniforms`)
//! - `soothe3_heatmap.wgsl` / `spectrogram_heatmap.wgsl` (`Soothe3HeatmapUniforms`)
//! - `mixer_meters.wgsl` / `sdf_uber_controls.wgsl` (`SdfControlInstance`, `MixerMetersUniforms`)

use bytemuck::{Pod, Zeroable};
use dsp_proq4_eq::{PROQ4_NUM_BANDS, ProQ4Processor};

/// GPU-instanced SDF Control primitive (`64` bytes per widget instance).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct SdfControlInstance {
    /// `[x, y, width, height]` in physical viewport pixels.
    pub rect_bounds: [f32; 4],
    /// `[norm_value, mod_value, peak_db, gr_db]`.
    pub values: [f32; 4],
    /// Packed RGBA8 primary color.
    pub color_primary: u32,
    /// Packed RGBA8 accent color.
    pub color_accent: u32,
    /// `0=Knob, 1=Meter, 2=Fader, 3=Button, 4=NodePort`.
    pub primitive_type: u32,
    /// `bit0=hovered, bit1=dragging, bit2=bypassed, bit3=clipping`.
    pub state_flags: u32,
    /// `[corner_radius, border_thickness, glow_intensity, reserved]`.
    pub style_params: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<SdfControlInstance>() == 64);

/// Uniform buffer layout for the Pro-Q 4 96-section GPU Shader (`proq4_curve.wgsl` & `reduction_eq_curve.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct BiquadCascadeUniforms {
    pub num_active_sections: u32,
    pub sample_rate: f32,
    pub min_freq_hz: f32,
    pub max_freq_hz: f32,
    /// `[b0, b1, b2, alpha_frac]` for up to 96 cascaded sections.
    pub coeffs_b: [[f32; 4]; 96],
    /// `[1.0, a1, a2, 0.0]` for up to 96 cascaded sections.
    pub coeffs_a: [[f32; 4]; 96],
}

impl Default for BiquadCascadeUniforms {
    fn default() -> Self {
        Self {
            num_active_sections: 0,
            sample_rate: 48000.0,
            min_freq_hz: 10.0,
            max_freq_hz: 30000.0,
            coeffs_b: [[1.0, 0.0, 0.0, 0.0]; 96],
            coeffs_a: [[1.0, 0.0, 0.0, 0.0]; 96],
        }
    }
}

impl BiquadCascadeUniforms {
    /// Packs all active biquad sections from a 24-Band [`ProQ4Processor`] into a GPU uniform buffer
    /// for 120 FPS evaluation in `proq4_curve.wgsl`.
    pub fn from_proq4_processor(proc: &ProQ4Processor) -> Self {
        let mut u = Self {
            num_active_sections: 0,
            sample_rate: proc.sample_rate as f32,
            min_freq_hz: 10.0,
            max_freq_hz: 30_000.0,
            ..Self::default()
        };

        let mut sec_cursor = 0usize;
        for b in 0..PROQ4_NUM_BANDS {
            if !proc.band_enabled[b] {
                continue;
            }
            let band = &proc.bands[b];
            let count = band.sections_n1.max(band.sections_n2);
            for s in 0..count {
                if sec_cursor >= 96 {
                    break;
                }
                let [b0, b1, b2, a1, a2] = if band.alpha_frac > 1e-6 && s < band.sections_n2 {
                    let c1 = if s < band.sections_n1 {
                        band.cascade_n1[s]
                    } else {
                        [1.0, 0.0, 0.0, 0.0, 0.0]
                    };
                    let c2 = band.cascade_n2[s];
                    let af = band.alpha_frac;
                    let inv = 1.0 - af;
                    [
                        inv * c1[0] + af * c2[0],
                        inv * c1[1] + af * c2[1],
                        inv * c1[2] + af * c2[2],
                        inv * c1[3] + af * c2[3],
                        inv * c1[4] + af * c2[4],
                    ]
                } else {
                    band.cascade_n1[s]
                };

                u.coeffs_b[sec_cursor] = [b0 as f32, b1 as f32, b2 as f32, band.alpha_frac as f32];
                u.coeffs_a[sec_cursor] = [1.0, a1 as f32, a2 as f32, 0.0];
                sec_cursor += 1;
            }
        }
        u.num_active_sections = sec_cursor as u32;
        u
    }
}

/// Uniform buffer layout for `soothe3_heatmap.wgsl` (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Soothe3HeatmapUniforms {
    pub min_db: f32,
    pub max_db: f32,
    pub tilt_db_oct: f32,
    pub ring_offset_v: f32,
    pub max_cut_db: f32,
    pub sharpness: f32,
    pub palette_mode: u32,
    pub delta_solo: u32,
}

const _: () = assert!(std::mem::size_of::<Soothe3HeatmapUniforms>() == 32);

/// Uniform buffer layout for `mixer_meters.wgsl` (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct MixerMetersUniforms {
    pub viewport_width: f32,
    pub viewport_height: f32,
    pub visible_track_start: u32,
    pub visible_track_count: u32,
    pub master_lufs_momentary: f32,
    pub master_lufs_short_term: f32,
    pub master_lufs_integrated: f32,
    pub master_gr_db: f32,
}

const _: () = assert!(std::mem::size_of::<MixerMetersUniforms>() == 32);
