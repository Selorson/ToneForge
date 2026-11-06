// ============================================================================
// TONEFORGE WGSL SHADER 1: 1024-Point Anti-Aliased EQ & Resonance Curve
// Ported from Soothe3 `reductionCurveFragmentShader` (offset 33997280)
// ============================================================================

struct CurveUniforms {
    stroke_color: vec4<f32>,    // RGBA line color
    fill_top_color: vec4<f32>,  // RGBA gradient near curve
    fill_bot_color: vec4<f32>,  // RGBA gradient at baseline (0 dB)
    resolution: vec2<f32>,      // (width_px, height_px)
    stroke_width: f32,          // e.g. 2.0 px
    baseline_y: f32,            // Normalized Y of 0 dB line (0.5 for Pro-Q 4, 0.0 for Soothe3)
    // 1024 normalized Y coordinates packed into 256 vec4<f32> (4,096 bytes)
    ys: array<vec4<f32>, 256>,
};

@group(0) @binding(0) var<uniform> u: CurveUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_curve(@builtin(vertex_index) vid: u32) -> VsOut {
    // Full-screen triangle (3 vertices, no vertex buffer needed)
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

fn read_y(idx: i32) -> f32 {
    let clamped = clamp(idx, 0, 1023);
    let vec_idx = clamped >> 2;
    let comp_idx = clamped & 3;
    return u.ys[vec_idx][comp_idx];
}

// Evaluates Catmull-Rom spline value and analytical first derivative dy/dx at normalized x in [0, 1]
fn sample_curve_catmull_rom(u_x: f32) -> vec2<f32> {
    let max_idx = 1023.0;
    let scaled_x = clamp(u_x, 0.0, 1.0) * max_idx;
    let i = i32(floor(scaled_x));
    let t = fract(scaled_x);

    let y0 = read_y(i - 1);
    let y1 = read_y(i);
    let y2 = read_y(i + 1);
    let y3 = read_y(i + 2);

    let a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
    let a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let a2 = -0.5 * y0 + 0.5 * y2;
    let a3 = y1;

    let t2 = t * t;
    let val = ((a0 * t + a1) * t + a2) * t + a3;
    let dval_dt = (3.0 * a0 * t2 + 2.0 * a1 * t + a2) * max_idx;
    return vec2<f32>(val, dval_dt);
}

@fragment
fn fs_curve(in: VsOut) -> @location(0) vec4<f32> {
    let eval = sample_curve_catmull_rom(in.uv.x);
    let curve_y_px = eval.x * u.resolution.y;
    let frag_y_px = in.uv.y * u.resolution.y;
    let base_y_px = u.baseline_y * u.resolution.y;

    // Screen-space derivative dy/dx in pixels per pixel
    let dy_dx = eval.y * (u.resolution.y / max(u.resolution.x, 1.0));
    // Exact perpendicular distance to the curve using first-order gradient normalization
    let perp_dist_px = abs(frag_y_px - curve_y_px) / sqrt(1.0 + dy_dx * dy_dx);

    // 1. Sub-pixel anti-aliased stroke coverage
    let half_w = 0.5 * u.stroke_width;
    let stroke_alpha = clamp(half_w + 0.5 - perp_dist_px, 0.0, 1.0);

    // 2. Soft exponential glow around the curve
    let glow_alpha = exp(-perp_dist_px / 14.0) * 0.35;

    // 3. Gradient fill between curve_y_px and base_y_px (0 dB reference line)
    let min_y = min(curve_y_px, base_y_px);
    let max_y = max(curve_y_px, base_y_px);
    let inside_fill = step(min_y, frag_y_px) * step(frag_y_px, max_y);
    let fill_t = clamp(abs(frag_y_px - base_y_px) / max(abs(curve_y_px - base_y_px), 1.0), 0.0, 1.0);
    let fill_col = mix(u.fill_bot_color, u.fill_top_color, fill_t);

    var out_col = fill_col * inside_fill;
    out_col = mix(out_col, u.stroke_color, max(stroke_alpha, glow_alpha * u.stroke_color.a));
    out_col.a = max(out_col.a * inside_fill, max(stroke_alpha * u.stroke_color.a, glow_alpha * u.stroke_color.a));
    return out_col;
}

