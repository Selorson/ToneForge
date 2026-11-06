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

/// Uniform buffer layout for `reduction_eq_curve.wgsl` (`4,160` bytes: 64B header + `ys: [[f32; 4]; 256]`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct CurveUniforms {
    pub stroke_color: [f32; 4],
    pub fill_top_color: [f32; 4],
    pub fill_bot_color: [f32; 4],
    pub resolution: [f32; 2],
    pub stroke_width: f32,
    pub baseline_y: f32,
    /// 1024 normalized Y coordinates packed into 256 `vec4<f32>` (`4,096` bytes).
    pub ys: [[f32; 4]; 256],
}

const _: () = assert!(std::mem::size_of::<CurveUniforms>() == 4160);

impl CurveUniforms {
    /// Packs a 1024-point normalized curve (`[f32; 1024]`) into `CurveUniforms` (`4,160` bytes).
    pub fn from_curve_1024(
        width_px: f32,
        height_px: f32,
        baseline_y: f32,
        stroke_color: [f32; 4],
        points_norm: &[f32; 1024],
    ) -> Self {
        let mut ys = [[0.0f32; 4]; 256];
        for i in 0..256 {
            let base = i * 4;
            ys[i] = [
                points_norm[base],
                points_norm[base + 1],
                points_norm[base + 2],
                points_norm[base + 3],
            ];
        }
        Self {
            stroke_color,
            fill_top_color: [stroke_color[0], stroke_color[1], stroke_color[2], 0.25],
            fill_bot_color: [stroke_color[0], stroke_color[1], stroke_color[2], 0.02],
            resolution: [width_px, height_px],
            stroke_width: 2.0,
            baseline_y,
            ys,
        }
    }
}

/// Uniform buffer layout for `spectrogram_heatmap.wgsl` (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct HeatmapUniforms {
    pub min_db: f32,
    pub max_db: f32,
    pub tilt_db_oct: f32,
    pub ring_offset_v: f32,
    pub palette_mode: u32,
    pub collision_on: u32,
    pub _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<HeatmapUniforms>() == 32);

/// The 7 SDF primitives ported from Soothe3 `shapeFragmentShader` (offset `34002400`).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdfPrimitiveKind {
    RoundedRect = 0,
    SemiCircleRing = 1,
    Rect = 2,
    EllipseNewtonRaphson = 3,
    MsdfGlyph = 4,
    SpriteCapsule = 5,
    Squircle = 6,
}

/// CSG Boolean operations ported from Soothe3 `shapeFragmentShader`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsgOperation {
    Union = 0,
    Intersect = 1,
    Subtract = 2,
    Replace = 3,
}

/// Per-instance 64-byte descriptor for `sdf_uber_controls.wgsl` (`4 x vec4<f32> = 64` bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct SdfPrimitive {
    /// `(x, y, w, h)` in screen pixels (`16` bytes).
    pub bbox: [f32; 4],
    /// `(r_tl, r_tr, r_br, r_bl)` or `(r_outer, r_inner, angle_0, angle_1)` (`16` bytes).
    pub radii: [f32; 4],
    /// RGBA fill color (`16` bytes).
    pub fill_color: [f32; 4],
    /// `(border_thickness, glow_sigma, value_norm, packed_prim_and_csg_op)` (`16` bytes).
    pub params: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<SdfPrimitive>() == 64);

impl SdfPrimitive {
    #[inline]
    pub fn new(
        bbox: [f32; 4],
        radii: [f32; 4],
        fill_color: [f32; 4],
        border_thickness: f32,
        glow_sigma: f32,
        value_norm: f32,
        kind: SdfPrimitiveKind,
        csg_op: CsgOperation,
    ) -> Self {
        let packed = (kind as u32) | ((csg_op as u32) << 8);
        Self {
            bbox,
            radii,
            fill_color,
            params: [border_thickness, glow_sigma, value_norm, packed as f32],
        }
    }
}

/// Evaluates CSG boolean combination (`Union`, `Intersect`, `Subtract`, `Replace`) on two signed distances.
#[inline]
pub fn csg_combine(d_a: f32, d_b: f32, op: CsgOperation) -> f32 {
    match op {
        CsgOperation::Union => d_a.min(d_b),
        CsgOperation::Intersect => d_a.max(d_b),
        CsgOperation::Subtract => d_a.max(-d_b),
        CsgOperation::Replace => d_b,
    }
}

/// Evaluates the signed distance at `local_px = [x, y]` for any of the 7 Soothe3 SDF primitives.
pub fn eval_sdf_primitive(
    local_px: [f32; 2],
    half_ext: [f32; 2],
    radii: [f32; 4],
    kind: SdfPrimitiveKind,
) -> f32 {
    let px = local_px[0];
    let py = local_px[1];
    let bx = half_ext[0].max(1.0);
    let by = half_ext[1].max(1.0);

    match kind {
        SdfPrimitiveKind::RoundedRect => {
            let rx = if px > 0.0 {
                [radii[0], radii[1]]
            } else {
                [radii[2], radii[3]]
            };
            let rad = if py > 0.0 { rx[1] } else { rx[0] };
            let qx = px.abs() - bx + rad;
            let qy = py.abs() - by + rad;
            let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
            qx.max(qy).min(0.0) + outside - rad
        }
        SdfPrimitiveKind::SemiCircleRing => {
            let r_mid = 0.5 * (radii[0] + radii[1]);
            let half_t = 0.5 * (radii[0] - radii[1]).abs();
            let ap = radii[2];
            let (sc_x, sc_y) = (ap.sin(), ap.cos());
            let qx = px.abs();
            let qy = py;
            let d_sc = if sc_y * qx > sc_x * qy {
                ((qx - sc_x * r_mid).powi(2) + (qy - sc_y * r_mid).powi(2)).sqrt()
            } else {
                ((qx * qx + qy * qy).sqrt() - r_mid).abs()
            };
            d_sc - half_t
        }
        SdfPrimitiveKind::Rect => {
            let dx = px.abs() - bx;
            let dy = py.abs() - by;
            (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt() + dx.max(dy).min(0.0)
        }
        SdfPrimitiveKind::EllipseNewtonRaphson => {
            let pax = px.abs();
            let pay = py.abs();
            let mut t = std::f32::consts::FRAC_PI_4;
            for _ in 0..3 {
                let (st, ct) = t.sin_cos();
                let ex = (bx * bx - by * by) * ct.powi(3) / bx;
                let ey = (by * by - bx * bx) * st.powi(3) / by;
                let rx = bx * ct - ex;
                let ry = by * st - ey;
                let qx = pax - ex;
                let qy = pay - ey;
                let r_len = rx.hypot(ry).max(1e-4);
                let q_len = qx.hypot(qy).max(1e-4);
                let delta_c = r_len * ((rx * qy - ry * qx) / (r_len * q_len)).clamp(-1.0, 1.0).asin();
                let delta_t = delta_c / (bx * bx * st * st + by * by * ct * ct).sqrt().max(1e-4);
                t = (t + delta_t).clamp(0.0, std::f32::consts::FRAC_PI_2);
            }
            let (st, ct) = t.sin_cos();
            let dist = (pax - bx * ct).hypot(pay - by * st);
            if (pax * pax) / (bx * bx) + (pay * pay) / (by * by) < 1.0 {
                -dist
            } else {
                dist
            }
        }
        SdfPrimitiveKind::MsdfGlyph => {
            let m = radii[0].min(radii[1]).max(radii[0].max(radii[1]).min(radii[2])).max(2.0);
            eval_sdf_primitive(local_px, half_ext, [m; 4], SdfPrimitiveKind::RoundedRect)
        }
        SdfPrimitiveKind::SpriteCapsule => {
            let r = radii[0].max(0.0);
            let qx = px.abs() - (bx - r).max(0.0);
            let qy = py.abs() - (by - r).max(0.0);
            qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - r
        }
        SdfPrimitiveKind::Squircle => {
            let r = radii[0].max(0.0);
            let qx = (px.abs() - bx + r).max(0.0);
            let qy = (py.abs() - by + r).max(0.0);
            let l4 = (qx.powi(4) + qy.powi(4)).powf(0.25);
            l4 + (px.abs() - bx + r).max(py.abs() - by + r).min(0.0) - r
        }
    }
}

