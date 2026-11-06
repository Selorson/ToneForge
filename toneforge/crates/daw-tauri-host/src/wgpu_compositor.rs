//! WGPU Multi-Viewport 120 FPS Compositor & SDF Instance Packing (`wgpu_compositor.rs`).
//!
//! Matches Section 4.1 (`SdfControlInstance`) and Section 4.2 (`BiquadCascadeUniforms`)
//! of `05_master_rust_svelte_blueprint.md`.

use bytemuck::{Pod, Zeroable};

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

/// Uniform buffer layout for the Pro-Q 4 96-section GPU Compute Shader (`reduction_eq_curve.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct BiquadCascadeUniforms {
    pub num_active_sections: u32,
    pub sample_rate: f32,
    pub min_freq_hz: f32,
    pub max_freq_hz: f32,
    /// `[b0, b1, b2, 0.0]` for up to 96 cascaded sections.
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
