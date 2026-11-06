// ============================================================================
// 502-Track Mixer Peak/RMS/GR Meters + Ozone 12 Vectorscope & Tonal Balance
// 120 FPS GPU-Instanced WGSL Shader (`mixer_meters.wgsl`)
// ============================================================================

struct MixerMetersUniforms {
    viewport_width: f32,
    viewport_height: f32,
    visible_track_start: u32,
    visible_track_count: u32,
    master_lufs_momentary: f32,
    master_lufs_short_term: f32,
    master_lufs_integrated: f32,
    master_gr_db: f32,
};

struct TrackMeterGpu {
    peak_l_db: f32,
    peak_r_db: f32,
    rms_l_db: f32,
    rms_r_db: f32,
    gr_db: f32,
    activity_flags: u32,
};

struct Ozone12GpuTelemetry {
    irc_gr_trace: array<f32, 512>,
    polar_sample_xy: array<vec2<f32>, 512>,
    tonal_balance_bars: array<f32, 256>,
    stem_rms_db: vec4<f32>,
    imager_corr_bands: vec4<f32>,
    exciter_drive_gr: array<f32, 248>,
};

@group(0) @binding(0) var<uniform> u_meters: MixerMetersUniforms;
@group(0) @binding(1) var<storage, read> track_meters: array<TrackMeterGpu, 502>;
@group(0) @binding(2) var<storage, read> ozone_telemetry: Ozone12GpuTelemetry;

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

fn db_to_meter_norm(db: f32) -> f32 {
    return clamp((db + 60.0) / 60.0, 0.0, 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var col = vec3<f32>(0.04, 0.055, 0.09);

    // Left 65% of viewport: 502-Track Mixer Peak/RMS/GR Meter Bridge
    if (in.uv.x < 0.65) {
        let local_x = in.uv.x / 0.65;
        let count = max(u_meters.visible_track_count, 1u);
        let slot = u32(floor(local_x * f32(count)));
        let cell_x = fract(local_x * f32(count));
        let track_idx = min(u_meters.visible_track_start + slot, 501u);
        let m = track_meters[track_idx];

        let fill_y = 1.0 - in.uv.y;
        let peak_l = db_to_meter_norm(m.peak_l_db);
        let peak_r = db_to_meter_norm(m.peak_r_db);
        let rms_l = db_to_meter_norm(m.rms_l_db);
        let rms_r = db_to_meter_norm(m.rms_r_db);
        let gr_norm = clamp(abs(m.gr_db) / 24.0, 0.0, 1.0);

        // Left channel bar (cell_x in 0.15..0.44), Right channel bar (cell_x in 0.48..0.77), GR bar (cell_x in 0.81..0.93)
        let is_left_bar = cell_x >= 0.15 && cell_x <= 0.44;
        let is_right_bar = cell_x >= 0.48 && cell_x <= 0.77;
        let is_gr_bar = cell_x >= 0.81 && cell_x <= 0.93;
        if (is_left_bar || is_right_bar) {
            let pk = select(peak_r, peak_l, is_left_bar);
            let rm = select(rms_r, rms_l, is_left_bar);
            if (fill_y <= pk) {
                let meter_col = mix(
                    vec3<f32>(0.12, 0.86, 0.58),
                    vec3<f32>(0.98, 0.78, 0.18),
                    smoothstep(0.65, 0.92, fill_y)
                );
                let clip_col = mix(meter_col, vec3<f32>(1.0, 0.20, 0.30), smoothstep(0.92, 0.99, fill_y));
                let brightness = select(0.55, 1.0, fill_y <= rm);
                col = clip_col * brightness;
            }
        } else if (is_gr_bar && in.uv.y <= gr_norm) {
            col = vec3<f32>(0.98, 0.38, 0.22);
        }
        return vec4<f32>(col, 1.0);
    }

    // Right 35% of viewport: Top half = Ozone 12 Polar Vectorscope XY, Bottom half = 256-Bar Tonal Balance
    let right_x = (in.uv.x - 0.65) / 0.35;
    if (in.uv.y < 0.5) {
        let p_uv = vec2<f32>(right_x * 2.0 - 1.0, (0.5 - in.uv.y) * 4.0 - 1.0);
        let r = length(p_uv);
        // Polar graticule rings
        if (abs(r - 0.45) < 0.012 || abs(r - 0.85) < 0.012) {
            col = vec3<f32>(0.14, 0.22, 0.34);
        }
        // Accumulate proximity to 64 sampled polar coordinates from 0x09800
        var glow = 0.0;
        for (var i: u32 = 0u; i < 64u; i = i + 1u) {
            let s = ozone_telemetry.polar_sample_xy[i * 8u];
            let d = length(p_uv - s);
            glow = glow + exp(-d * 38.0);
        }
        col = col + vec3<f32>(0.18, 0.92, 0.76) * clamp(glow * 0.22, 0.0, 1.0);
    } else {
        // Ozone 12 256-Bar Tonal Balance Target Corridor (0x0A800)
        let local_y = 1.0 - (in.uv.y - 0.5) * 2.0;
        let bar_idx = min(u32(floor(right_x * 255.0)), 255u);
        let val = clamp(ozone_telemetry.tonal_balance_bars[bar_idx], 0.0, 1.0);
        let target_low = 0.38;
        let target_high = 0.68;

        if (local_y >= target_low && local_y <= target_high) {
            col = mix(col, vec3<f32>(0.16, 0.34, 0.52), 0.35);
        }
        if (abs(local_y - val) < 0.025) {
            col = vec3<f32>(0.32, 0.88, 1.0);
        }
    }

    return vec4<f32>(col, 1.0);
}
