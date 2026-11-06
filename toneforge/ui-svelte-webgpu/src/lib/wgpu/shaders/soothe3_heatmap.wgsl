// ============================================================================
// oeksound Soothe3 Dynamic Resonance Reduction & 20x1024 Scrolling Heatmap
// 120 FPS WGSL Shader (`soothe3_heatmap.wgsl`)
// ============================================================================

struct Soothe3HeatmapUniforms {
    min_db: f32,
    max_db: f32,
    tilt_db_oct: f32,
    ring_offset_v: f32,
    max_cut_db: f32,
    sharpness: f32,
    palette_mode: u32,
    delta_solo: u32,
};

struct Soothe3TelemetryBuffer {
    reduction_db: array<f32, 1024>,
    input_spec_db: array<f32, 1024>,
    spectrogram_ring: array<f32, 20480>, // 20 rows x 1024 bins at 0x0B000
};

@group(0) @binding(0) var<uniform> u_soothe: Soothe3HeatmapUniforms;
@group(0) @binding(1) var<storage, read> soothe_buf: Soothe3TelemetryBuffer;

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

    // 1. Sample 20-row scrolling spectrogram ring buffer (0x0B000)
    let row_f = fract(in.uv.y + u_soothe.ring_offset_v) * 20.0;
    let row_idx = min(u32(floor(row_f)), 19u);
    let spec_db = mix(
        soothe_buf.spectrogram_ring[row_idx * 1024u + b0],
        soothe_buf.spectrogram_ring[row_idx * 1024u + b1],
        frac
    );

    // Apply 4.5 dB/oct visual tilt compensation around 1 kHz (norm_x = 0.56)
    let octaves_from_1k = (in.uv.x - 0.56) * 9.96578;
    let tilted_db = spec_db + octaves_from_1k * u_soothe.tilt_db_oct;
    let norm_spec = clamp((tilted_db - u_soothe.min_db) / max(u_soothe.max_db - u_soothe.min_db, 1.0), 0.0, 1.0);

    // 2. Sample Soothe3 dynamic resonance reduction curve (0x07000, 512/1024 bins)
    let gr_db = mix(soothe_buf.reduction_db[b0], soothe_buf.reduction_db[b1], frac);
    let norm_cut = clamp(abs(gr_db) / max(abs(u_soothe.max_cut_db), 6.0), 0.0, 1.0);

    // Base dark indigo-cyan spectrogram palette
    var color = mix(
        vec3<f32>(0.03, 0.04, 0.09),
        vec3<f32>(0.12, 0.64, 0.92),
        pow(norm_spec, 1.35)
    );

    // Dynamic Cut Curve hanging from top (uv.y = 0.0 down to cut_y)
    let cut_y = clamp(norm_cut, 0.0, 0.95);
    let dist_cut = abs(in.uv.y - cut_y);
    let aa = max(fwidth(dist_cut) * 1.5, 0.003);
    let stroke = 1.0 - smoothstep(0.0, aa, dist_cut);

    if (in.uv.y <= cut_y) {
        let fill_col = select(vec3<f32>(0.96, 0.26, 0.52), vec3<f32>(1.0, 0.42, 0.18), u_soothe.delta_solo != 0u);
        color = mix(color, fill_col, 0.42 + 0.35 * norm_cut);
    }

    // Resonance suppression hotspot glow
    color = color + vec3<f32>(0.95, 0.22, 0.48) * (norm_cut * norm_spec * 0.55);

    // Crisp anti-aliased reduction curve stroke
    color = mix(color, vec3<f32>(1.0, 0.58, 0.78), stroke);

    return vec4<f32>(color, 1.0);
}
