// GPU-Instanced Signed Distance Field (SDF) Uber-Shader for 500-Track Mixer & Knobs
struct SdfControlInstance {
    rect_bounds: vec4<f32>,
    values: vec4<f32>,
    color_primary: u32,
    color_accent: u32,
    primitive_type: u32,
    state_flags: u32,
    style_params: vec4<f32>,
};

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r, r);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0, 0.0))) - r;
}

fn sd_arc(p_in: vec2<f32>, sc_aperture: vec2<f32>, ra: f32, rb: f32) -> f32 {
    var p = p_in;
    p.x = abs(p.x);
    if (sc_aperture.y * p.x > sc_aperture.x * p.y) {
        return length(p - sc_aperture * ra) - rb;
    }
    return abs(length(p) - ra) - rb;
}
