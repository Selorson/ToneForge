// ============================================================================
// TONEFORGE WGSL SHADER 3: Uber-SDF Hardware Controls (Knobs, Meters, Nodes)
// Ported from Soothe3 `shapeFragmentShader` (offset 34002400) — CSG 7 Primitives
// ============================================================================

// Per-instance 64-byte descriptor (`4 x vec4<f32> = 64` bytes) for instanced drawing
struct SdfPrimitive {
    bbox: vec4<f32>,         // (x, y, w, h) in screen pixels (16 B)
    radii: vec4<f32>,        // (r_tl, r_tr, r_br, r_bl) OR (r_outer, r_inner, angle_0, angle_1) (16 B)
    fill_color: vec4<f32>,   // RGBA fill color (16 B)
    params: vec4<f32>,       // (border_thickness, glow_sigma, value_norm, packed_prim_and_csg_op) (16 B)
};

struct ScreenUniforms {
    screen_size: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> screen: ScreenUniforms;
@group(0) @binding(1) var<storage, read> primitives: array<SdfPrimitive>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local_px: vec2<f32>,
    @location(1) @interpolate(flat) inst_id: u32,
};

@vertex
fn vs_sdf(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
    let prim = primitives[iid];
    let corner = vec2<f32>(f32(vid & 1u), f32((vid >> 1u) & 1u));
    let pad = prim.params.y * 2.5 + 4.0; // Expand quad to fit outer glow/shadow
    let px_pos = prim.bbox.xy - vec2<f32>(pad) + corner * (prim.bbox.zw + vec2<f32>(pad * 2.0));
    let ndc = vec2<f32>(
        (px_pos.x / screen.screen_size.x) * 2.0 - 1.0,
        1.0 - (px_pos.y / screen.screen_size.y) * 2.0
    );

    var out: VsOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.local_px = px_pos - (prim.bbox.xy + 0.5 * prim.bbox.zw);
    out.inst_id = iid;
    return out;
}

// ----------------------------------------------------------------------------
// 7 SDF Primitives ported from Soothe3 `shapeFragmentShader` (`0..6`):
// 0 = rounded_rect, 1 = semi_circle_ring, 2 = rect, 3 = Newton-Raphson ellipse,
// 4 = MSDF text glyph, 5 = SDF sprite, 6 = L4 squircle
// ----------------------------------------------------------------------------

// Primitive 0: Rounded Box SDF
fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: vec4<f32>) -> f32 {
    var rx = select(r.zw, r.xy, p.x > 0.0);
    let rad = select(rx.y, rx.x, p.y > 0.0);
    let q = abs(p) - b + vec2<f32>(rad);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - rad;
}

// Primitive 1: Rotary Knob / Peak Arc Ring SDF (`r_outer`, `r_inner`, `sweep_angle`)
fn sd_ring_arc(p: vec2<f32>, r_mid: f32, half_thick: f32, aperture_cos_sin: vec2<f32>) -> f32 {
    var q = vec2<f32>(abs(p.x), p.y);
    let d_sc = select(
        length(q - aperture_cos_sin * r_mid),
        abs(length(q) - r_mid),
        aperture_cos_sin.y * q.x > aperture_cos_sin.x * q.y
    );
    return d_sc - half_thick;
}

// Primitive 2: Sharp Axis-Aligned Box SDF
fn sd_rect(p: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

// Primitive 3: Newton-Raphson Exact Ellipse SDF
fn sd_ellipse_nr(p: vec2<f32>, ab: vec2<f32>) -> f32 {
    let pa = abs(p);
    let a = max(ab.x, 1.0);
    let b = max(ab.y, 1.0);
    var t = 0.78539816339;
    for (var iter = 0; iter < 3; iter = iter + 1) {
        let ct = cos(t);
        let st = sin(t);
        let ex = (a * a - b * b) * ct * ct * ct / a;
        let ey = (b * b - a * a) * st * st * st / b;
        let rx = a * ct - ex;
        let ry = b * st - ey;
        let qx = pa.x - ex;
        let qy = pa.y - ey;
        let r_len = max(hypot(rx, ry), 1e-4);
        let q_len = max(hypot(qx, qy), 1e-4);
        let delta_c = r_len * asin(clamp((rx * qy - ry * qx) / (r_len * q_len), -1.0, 1.0));
        let delta_t = delta_c / max(sqrt(a * a * st * st + b * b * ct * ct), 1e-4);
        t = clamp(t + delta_t, 0.0, 1.57079632679);
    }
    let closest = vec2<f32>(a * cos(t), b * sin(t));
    let dist = length(pa - closest);
    let inside = (pa.x * pa.x) / (a * a) + (pa.y * pa.y) / (b * b) < 1.0;
    return select(dist, -dist, inside);
}

fn hypot(x: f32, y: f32) -> f32 {
    return sqrt(x * x + y * y);
}

// Primitive 4: Median-of-3 MSDF Glyph helper
fn sd_msdf_median(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

// Primitive 5: Radial / Pill SDF Sprite
fn sd_sprite_capsule(p: vec2<f32>, half_ext: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - max(half_ext - vec2<f32>(r), vec2<f32>(0.0));
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// Primitive 6: Squircle (L4 Superellipse) SDF
fn sd_squircle(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let d = abs(p) - b + vec2<f32>(r);
    let q = max(d, vec2<f32>(0.0));
    let l4 = pow(q.x * q.x * q.x * q.x + q.y * q.y * q.y * q.y, 0.25);
    return l4 + min(max(d.x, d.y), 0.0) - r;
}

// CSG Boolean Operations (`0=union`, `1=intersect`, `2=subtract`, `3=replace`)
fn csg_combine(d_a: f32, d_b: f32, op: u32) -> f32 {
    if (op == 0u) { return min(d_a, d_b); }
    if (op == 1u) { return max(d_a, d_b); }
    if (op == 2u) { return max(d_a, -d_b); }
    return d_b;
}

fn eval_primitive_sdf(p: vec2<f32>, half_ext: vec2<f32>, radii: vec4<f32>, prim_type: u32) -> f32 {
    if (prim_type == 0u) {
        return sd_rounded_box(p, half_ext, radii);
    } else if (prim_type == 1u) {
        let r_mid = 0.5 * (radii.x + radii.y);
        let half_t = 0.5 * abs(radii.x - radii.y);
        let ap = radii.z;
        return sd_ring_arc(p, r_mid, half_t, vec2<f32>(sin(ap), cos(ap)));
    } else if (prim_type == 2u) {
        return sd_rect(p, half_ext);
    } else if (prim_type == 3u) {
        return sd_ellipse_nr(p, half_ext);
    } else if (prim_type == 4u) {
        let m = sd_msdf_median(radii.x, radii.y, radii.z);
        return sd_rounded_box(p, half_ext, vec4<f32>(max(m, 2.0)));
    } else if (prim_type == 5u) {
        return sd_sprite_capsule(p, half_ext, radii.x);
    } else {
        return sd_squircle(p, half_ext, radii.x);
    }
}

@fragment
fn fs_sdf(in: VsOut) -> @location(0) vec4<f32> {
    let prim = primitives[in.inst_id];
    let packed = u32(prim.params.w);
    let prim_type = packed & 0xFFu;
    let csg_op = (packed >> 8u) & 0xFFu;
    let half_ext = 0.5 * prim.bbox.zw;

    var d = eval_primitive_sdf(in.local_px, half_ext, prim.radii, prim_type);
    if (csg_op != 0u) {
        let inner_d = sd_rounded_box(in.local_px, half_ext * 0.65, prim.radii * 0.5);
        d = csg_combine(d, inner_d, csg_op);
    }

    let aa = max(fwidth(d), 0.75);
    let fill_mask = 1.0 - smoothstep(-aa, aa, d);
    let border_w = prim.params.x;
    let border_mask = (1.0 - smoothstep(-aa, aa, abs(d) - 0.5 * border_w)) * step(0.01, border_w);
    let glow_sigma = max(prim.params.y, 0.5);
    let glow_mask = exp(-max(d, 0.0) / glow_sigma) * 0.4;

    let border_col = vec4<f32>(min(prim.fill_color.rgb * 1.35 + vec3<f32>(0.15), vec3<f32>(1.0)), prim.fill_color.a);
    var col = prim.fill_color * fill_mask;
    col = mix(col, border_col, border_mask);
    col = col + border_col * glow_mask * (1.0 - fill_mask);
    return col;
}
