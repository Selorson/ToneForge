// Parallel Reduction Compute Shader for 96-Section Biquad Cascade Frequency Response
// Evaluates H(e^{j\omega_k}) across 2048 log-spaced frequency points (10 Hz .. 30 kHz)

struct BiquadCascadeUniforms {
    num_active_sections: u32,
    sample_rate: f32,
    min_freq_hz: f32,
    max_freq_hz: f32,
    coeffs_b: array<vec4<f32>, 96>,
    coeffs_a: array<vec4<f32>, 96>,
};

@group(0) @binding(0) var<uniform> eq_uniforms: BiquadCascadeUniforms;
@group(0) @binding(1) var<storage, read_write> out_magnitude_db: array<f32, 2048>;
@group(0) @binding(2) var<storage, read_write> out_phase_rad: array<f32, 2048>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn complex_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let denom = max(dot(b, b), 1e-30);
    return vec2<f32>(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / denom;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let freq_idx = global_id.x;
    if (freq_idx >= 2048u) { return; }

    let norm_x = f32(freq_idx) / 2047.0;
    let freq_hz = eq_uniforms.min_freq_hz * pow(eq_uniforms.max_freq_hz / eq_uniforms.min_freq_hz, norm_x);
    let omega = 6.28318530718 * freq_hz / eq_uniforms.sample_rate;

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
