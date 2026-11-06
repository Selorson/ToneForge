// ============================================================================
// TONEFORGE WGSL SHADER 2: 1024-Bin Scrolling Spectrogram & Masking Heatmap
// Ported from Pro-Q 4 Analyzer & Soothe3 Dynamic Spectral Display
// ============================================================================

struct HeatmapUniforms {
    min_db: f32,           // e.g. -90.0 dB
    max_db: f32,           // e.g.   0.0 dB
    tilt_db_oct: f32,      // e.g.   4.5 dB/oct pink-noise tilt around 1 kHz
    ring_offset_v: f32,    // Circular ring-buffer V scroll offset in [0.0, 1.0)
    palette_mode: u32,     // 0 = Pro-Q 4 Cyan/Blue, 1 = Soothe3 Violet/Gold, 2 = Ozone 12 Teal/Emerald
    collision_on: u32,     // 1 = overlay Pro-Q 4 crimson collision mask
    _pad: vec2<u32>,
};

@group(0) @binding(0) var<uniform> u: HeatmapUniforms;
@group(0) @binding(1) var tex_spec: texture_2d<f32>;     // 1024 x H R32Float spectrogram ring
@group(0) @binding(2) var tex_mask: texture_2d<f32>;     // 1024 x 1 R32Float collision/reduction mask
@group(0) @binding(3) var samp_lin: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_heatmap(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

// 5-stop perceptual color LUTs for Pro-Q 4, Soothe3, and Ozone 12
fn palette_proq4(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.02, 0.03, 0.06);
    let c1 = vec3<f32>(0.05, 0.18, 0.42);
    let c2 = vec3<f32>(0.08, 0.52, 0.78);
    let c3 = vec3<f32>(0.25, 0.88, 0.95);
    let c4 = vec3<f32>(0.92, 0.99, 1.00);
    let s = clamp(t, 0.0, 1.0) * 4.0;
    if (s < 1.0) { return mix(c0, c1, s); }
    if (s < 2.0) { return mix(c1, c2, s - 1.0); }
    if (s < 3.0) { return mix(c2, c3, s - 2.0); }
    return mix(c3, c4, s - 3.0);
}

fn palette_soothe3(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.04, 0.02, 0.07);
    let c1 = vec3<f32>(0.22, 0.08, 0.38);
    let c2 = vec3<f32>(0.58, 0.16, 0.52);
    let c3 = vec3<f32>(0.92, 0.45, 0.22);
    let c4 = vec3<f32>(1.00, 0.92, 0.65);
    let s = clamp(t, 0.0, 1.0) * 4.0;
    if (s < 1.0) { return mix(c0, c1, s); }
    if (s < 2.0) { return mix(c1, c2, s - 1.0); }
    if (s < 3.0) { return mix(c2, c3, s - 2.0); }
    return mix(c3, c4, s - 3.0);
}

fn palette_ozone12(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.03, 0.06, 0.07);
    let c1 = vec3<f32>(0.04, 0.24, 0.28);
    let c2 = vec3<f32>(0.08, 0.56, 0.52);
    let c3 = vec3<f32>(0.28, 0.88, 0.72);
    let c4 = vec3<f32>(0.88, 1.00, 0.95);
    let s = clamp(t, 0.0, 1.0) * 4.0;
    if (s < 1.0) { return mix(c0, c1, s); }
    if (s < 2.0) { return mix(c1, c2, s - 1.0); }
    if (s < 3.0) { return mix(c2, c3, s - 2.0); }
    return mix(c3, c4, s - 3.0);
}

@fragment
fn fs_heatmap(in: VsOut) -> @location(0) vec4<f32> {
    // Wrap V coordinate around ring buffer
    let scrolled_v = fract(in.uv.y + u.ring_offset_v);
    let raw_db = textureSampleLevel(tex_spec, samp_lin, vec2<f32>(in.uv.x, scrolled_v), 0.0).r;

    // Apply visual tilt around 1 kHz (at log-frequency uv.x = 0.566 for 20 Hz..20 kHz)
    let octaves_from_1k = (in.uv.x - 0.566) * 9.965784; // log2(20000/20) = 9.965784 octaves
    let tilted_db = raw_db + octaves_from_1k * u.tilt_db_oct;
    let norm_mag = clamp((tilted_db - u.min_db) / max(u.max_db - u.min_db, 1.0), 0.0, 1.0);

    var rgb: vec3<f32>;
    if (u.palette_mode == 0u) {
        rgb = palette_proq4(norm_mag);
    } else if (u.palette_mode == 1u) {
        rgb = palette_soothe3(norm_mag);
    } else {
        rgb = palette_ozone12(norm_mag);
    }

    // Optional Pro-Q 4 Spectral Collision glow overlay
    if (u.collision_on != 0u) {
        let coll = clamp(textureSampleLevel(tex_mask, samp_lin, vec2<f32>(in.uv.x, 0.5), 0.0).r, 0.0, 1.0);
        let crimson = vec3<f32>(1.0, 0.18, 0.24);
        rgb = mix(rgb, crimson, coll * 0.72);
    }

    return vec4<f32>(rgb, clamp(norm_mag * 1.15, 0.08, 0.96));
}
