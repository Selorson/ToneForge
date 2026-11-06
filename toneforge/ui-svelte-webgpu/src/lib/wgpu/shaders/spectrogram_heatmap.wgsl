// Spectrogram Ring-Texture & Soothe3 Delta Suppression Heatmap Shader
struct SpectrogramUniforms {
    ring_write_row: u32,
    total_rows: u32,
    min_db: f32,
    max_db: f32,
};

@group(0) @binding(0) var<uniform> spec_u: SpectrogramUniforms;
@group(0) @binding(1) var tex_spectrum: texture_2d<f32>;
@group(0) @binding(2) var tex_soothe_gr: texture_2d<f32>;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let bin_idx = i32(clamp(uv.x * 2048.0, 0.0, 2047.0));
    let row_offset = u32(uv.y * f32(spec_u.total_rows));
    let ring_row = i32((spec_u.ring_write_row + spec_u.total_rows - row_offset) % spec_u.total_rows);

    let mag_db = textureLoad(tex_spectrum, vec2<i32>(bin_idx, ring_row), 0).r;
    let gr_db = textureLoad(tex_soothe_gr, vec2<i32>(bin_idx, 0), 0).r;

    let norm_mag = clamp((mag_db - spec_u.min_db) / (spec_u.max_db - spec_u.min_db), 0.0, 1.0);
    let norm_gr = clamp(abs(gr_db) / 24.0, 0.0, 1.0);

    let base_color = mix(vec3<f32>(0.04, 0.06, 0.12), vec3<f32>(0.15, 0.78, 0.96), norm_mag);
    let delta_glow = vec3<f32>(1.0, 0.22, 0.45) * norm_gr;

    return vec4<f32>(base_color + delta_glow, 1.0);
}
