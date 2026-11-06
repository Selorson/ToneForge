// ============================================================================
// Pro-Q 4 24-Band (96-Section Cascade) Frequency Response & 1024-Bin Spectrum
// 120 FPS WGSL Compute + Fragment Shader (`proq4_curve.wgsl`)
// ============================================================================

struct BiquadCascadeUniforms {
    num_active_sections: u32,
    sample_rate: f32,
    min_freq_hz: f32,
    max_freq_hz: f32,
    coeffs_b: array<vec4<f32>, 96>,
    coeffs_a: array<vec4<f32>, 96>,
};

struct ProQ4SpectrumBuffer {
    pre_db: array<f32, 1024>,
    post_db: array<f32, 1024>,
    sc_db: array<f32, 1024>,
    collision_mask: array<f32, 1024>,
};

@group(0) @binding(0) var<uniform> eq_uniforms: BiquadCascadeUniforms;
@group(0) @binding(1) var<storage, read> spectrum_buf: ProQ4SpectrumBuffer;
@group(0) @binding(2) var<storage, read_write> out_magnitude_db: array<f32, 2048>;
@group(0) @binding(3) var<storage, read_write> out_phase_rad: array<f32, 2048>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn complex_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let denom = max(dot(b, b), 1e-30);
    return vec2<f32>(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / denom;
}

// Evaluates the 96-section cascaded biquad transfer function H(e^{j*omega}) in dB
fn evaluate_cascade_db(norm_x: f32) -> f32 {
    let freq_hz = eq_uniforms.min_freq_hz * pow(eq_uniforms.max_freq_hz / eq_uniforms.min_freq_hz, clamp(norm_x, 0.0, 1.0));
    let omega = 6.28318530718 * freq_hz / max(eq_uniforms.sample_rate, 8000.0);

    let z1 = vec2<f32>(cos(omega), -sin(omega));
    let z2 = vec2<f32>(cos(2.0 * omega), -sin(2.0 * omega));

    var h_total = vec2<f32>(1.0, 0.0);
    let count = min(eq_uniforms.num_active_sections, 96u);

    for (var s: u32 = 0u; s < count; s = s + 1u) {
        let b = eq_uniforms.coeffs_b[s];
        let a = eq_uniforms.coeffs_a[s];

        let num = vec2<f32>(b.x, 0.0) + b.y * z1 + b.z * z2;
        let den = vec2<f32>(1.0, 0.0) + a.y * z1 + a.z * z2;
        h_total = complex_mul(h_total, complex_div(num, den));
    }

    let mag_sq = max(dot(h_total, h_total), 1e-24);
    return 10.0 * log2(mag_sq) * 0.30102999566;
}

@compute @workgroup_size(64)
fn cs_eval_proq4_curve(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let freq_idx = global_id.x;
    if (freq_idx >= 2048u) { return; }

    let norm_x = f32(freq_idx) / 2047.0;
    let freq_hz = eq_uniforms.min_freq_hz * pow(eq_uniforms.max_freq_hz / eq_uniforms.min_freq_hz, norm_x);
    let omega = 6.28318530718 * freq_hz / max(eq_uniforms.sample_rate, 8000.0);

    let z1 = vec2<f32>(cos(omega), -sin(omega));
    let z2 = vec2<f32>(cos(2.0 * omega), -sin(2.0 * omega));

    var h_total = vec2<f32>(1.0, 0.0);
    let count = min(eq_uniforms.num_active_sections, 96u);

    for (var s: u32 = 0u; s < count; s = s + 1u) {
        let b = eq_uniforms.coeffs_b[s];
        let a = eq_uniforms.coeffs_a[s];
        let num = vec2<f32>(b.x, 0.0) + b.y * z1 + b.z * z2;
        let den = vec2<f32>(1.0, 0.0) + a.y * z1 + a.z * z2;
        h_total = complex_mul(h_total, complex_div(num, den));
    }

    let mag_sq = max(dot(h_total, h_total), 1e-24);
    out_magnitude_db[freq_idx] = 10.0 * log2(mag_sq) * 0.30102999566;
    out_phase_rad[freq_idx] = atan2(h_total.y, h_total.x);
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0)
    );
    let p = positions[vid];
    var out: VsOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let bin_f = clamp(in.uv.x * 1023.0, 0.0, 1023.0);
    let b0 = u32(floor(bin_f));
    let b1 = min(b0 + 1u, 1023u);
    let frac = fract(bin_f);

    // 1. Interpolate 1024-bin Pre/Post/Collision spectrum from 0x03000
    let pre_db = mix(spectrum_buf.pre_db[b0], spectrum_buf.pre_db[b1], frac);
    let post_db = mix(spectrum_buf.post_db[b0], spectrum_buf.post_db[b1], frac);
    let coll = mix(spectrum_buf.collision_mask[b0], spectrum_buf.collision_mask[b1], frac);

    let spec_y_pre = 1.0 - clamp((pre_db + 90.0) / 90.0, 0.0, 1.0);
    let spec_y_post = 1.0 - clamp((post_db + 90.0) / 90.0, 0.0, 1.0);

    // 2. Evaluate 24-Band Composite EQ Curve (-30 dB .. +30 dB mapped to uv.y 1.0 .. 0.0)
    let curve_db = evaluate_cascade_db(in.uv.x);
    let curve_y = 0.5 - clamp(curve_db / 60.0, -0.5, 0.5);

    // Screen-space analytical anti-aliasing via fwidth()
    let dist_curve = abs(in.uv.y - curve_y);
    let aa_width = max(fwidth(dist_curve) * 1.5, 0.0025);
    let stroke_alpha = 1.0 - smoothstep(0.0, aa_width, dist_curve);

    var color = vec3<f32>(0.035, 0.05, 0.085);

    // Pre/Post Spectrum analyzer fill (completely hidden when paused at -120 dB)
    if (pre_db > -115.0 && in.uv.y >= spec_y_pre) {
        color = mix(color, vec3<f32>(0.18, 0.32, 0.52), 0.32);
    }
    if (post_db > -115.0 && in.uv.y >= spec_y_post) {
        color = mix(color, vec3<f32>(0.16, 0.72, 0.96), 0.45);
    }

    // Collision Mask red warning glow (0x06000)
    if (post_db > -115.0 && coll > 0.05 && in.uv.y >= spec_y_post) {
        color = mix(color, vec3<f32>(0.98, 0.20, 0.32), clamp(coll, 0.0, 0.85));
    }

    // Golden-cyan EQ Fill between 0 dB (uv.y = 0.5) and curve_y
    let in_eq_fill = (in.uv.y >= min(0.5, curve_y)) && (in.uv.y <= max(0.5, curve_y));
    if (in_eq_fill) {
        color = mix(color, vec3<f32>(0.98, 0.76, 0.18), 0.22);
    }

    // Anti-aliased 24-Band Composite Response Stroke
    color = mix(color, vec3<f32>(1.0, 0.88, 0.32), stroke_alpha);

    return vec4<f32>(color, 1.0);
}
