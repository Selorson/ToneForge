//! Long-run simulations of the hardware-input bridges against a faithful replica of the output
//! SRC's pull schedule (`WasapiStreamingResampler::render_hardware_frames`):
//!
//! 1. ASIO + SRC (input and output on ONE clock): the clock-locked bridge must never underrun /
//!    overrun for 44.1/88.2/96/176.4/192 kHz × 64..1024-frame buffers over 60 s, and the measured
//!    loopback impulse delay must equal
//!    `AdaptiveDllDriftResampler::clock_locked_duplex_extra_latency_engine_frames`.
//! 2. WASAPI capture (separate clock, 10 ms packets with late-delivery jitter): the DLL bridge must
//!    never xrun, must settle on its fixed target fill, and its delay must match
//!    `dll_input_delay_engine_frames + render_lookahead_engine_frames` up to the capture packet
//!    phase (0..1 packet).
//!
//! Run with `--nocapture` to print the measured tables.

use daw_core_graph::AdaptiveDllDriftResampler;
use daw_core_graph::audio_io::CaptureBridgeMode;
use std::f64::consts::PI;

const ENGINE_SR: u32 = 48_000;
const QUANTUM: usize = 256;
const SIM_SECONDS: f64 = 60.0;

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let h = 0.25 * x * x;
    for k in 1..=40 {
        let kf = k as f64;
        term *= h / (kf * kf);
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}

/// Windowed-sinc kernel value with the output SRC's design (`cutoff` relative to the input
/// Nyquist, Kaiser beta 9.5 over ±64 taps).
fn out_src_kernel(rel_x: f64, cutoff: f64) -> f64 {
    let win_pos = (rel_x / 64.0).clamp(-1.0, 1.0);
    let kaiser = bessel_i0(9.5 * (1.0 - win_pos * win_pos).max(0.0).sqrt()) / bessel_i0(9.5);
    let arg = PI * cutoff * rel_x;
    let sinc = if arg.abs() < 1e-12 { cutoff } else { arg.sin() / (PI * rel_x) };
    sinc * kaiser
}

/// Replica of the pull schedule and frame mapping of `render_hardware_frames`, for a signal made
/// of a single engine-rate unit impulse (all other engine frames are zero, so the convolution is
/// evaluated only where the impulse is inside the 128-tap window).
struct OutSrcReplica {
    passthrough: bool,
    step: f64,
    frac: f64,
    /// Absolute ring position of the read head.
    base: u64,
    /// Absolute ring position of the write head (lead-in zeros included).
    written: u64,
    lead: u64,
    cutoff: f64,
}

impl OutSrcReplica {
    fn new(engine_sr: u32, hw_sr: u32) -> Self {
        let passthrough = engine_sr == hw_sr;
        let lead = if passthrough { 0 } else { 64 };
        Self {
            passthrough,
            step: (engine_sr as f64) / (hw_sr as f64),
            frac: 0.0,
            base: 0,
            written: lead,
            lead,
            cutoff: 0.94 * ((hw_sr as f64) / (engine_sr as f64)).min(1.0),
        }
    }

    /// Produces one hardware frame; `render(b)` is called whenever engine block `b` is pulled.
    fn next(&mut self, impulse_engine_frame: u64, render: &mut dyn FnMut(u64)) -> f32 {
        let imp = impulse_engine_frame + self.lead;
        if self.passthrough {
            if self.written == self.base {
                render(self.written - self.lead);
                self.written += QUANTUM as u64;
            }
            let v = if self.base == imp { 1.0 } else { 0.0 };
            self.base += 1;
            return v;
        }
        while self.written - self.base < 128 + 4 {
            render(self.written - self.lead);
            self.written += QUANTUM as u64;
        }
        let v = if imp >= self.base && imp < self.base + 128 {
            let t = (imp - self.base) as f64;
            out_src_kernel(t - 63.0 - self.frac, self.cutoff) as f32
        } else {
            0.0
        };
        self.frac += self.step;
        let adv = self.frac.floor();
        if adv > 0.0 {
            let avail = self.written - self.base;
            self.base += (adv as u64).min(avail);
            self.frac -= adv;
        }
        v
    }
}

/// Parabolic sub-sample peak of `|x|`.
fn peak(x: &[f32]) -> (f64, f32) {
    let (i, &v) = x
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .unwrap();
    if i == 0 || i + 1 >= x.len() {
        return (i as f64, v);
    }
    let (a, b, c) = (x[i - 1].abs() as f64, x[i].abs() as f64, x[i + 1].abs() as f64);
    let den = a - 2.0 * b + c;
    let d = if den.abs() > 1e-12 { 0.5 * (a - c) / den } else { 0.0 };
    (i as f64 + d, v)
}

// ================================================================================================
// 1. ASIO + SRC clock-locked bridge
// ================================================================================================

struct AsioResult {
    hw_sr: u32,
    buf: usize,
    cushion: usize,
    measured_extra: f64,
    predicted_extra: f64,
    peak_amp: f32,
    min_margin: i64,
    underruns: u64,
    overruns: u64,
}

fn run_asio_clock_locked(hw_sr: u32, buf: usize, cushion_override: Option<usize>, secs: f64) -> AsioResult {
    let hw_f = hw_sr as f64;
    let eng_f = ENGINE_SR as f64;
    let cushion = cushion_override
        .unwrap_or_else(|| AdaptiveDllDriftResampler::clock_locked_cushion_frames(hw_f, eng_f, buf));
    let mut bridge =
        AdaptiveDllDriftResampler::new_clock_locked_with_cushion(hw_f, eng_f, cushion);
    assert_eq!(bridge.mode(), CaptureBridgeMode::ClockLocked);
    let (mut tx, mut rx) = rtrb::RingBuffer::<[f32; 2]>::new(32_768);
    let mut src = OutSrcReplica::new(ENGINE_SR, hw_sr);

    // Engine impulse 45 s in (steady state), arbitrary sub-block offset.
    let m0: u64 = 45 * ENGINE_SR as u64 + 101;
    let win_len = 6_000usize;
    let mut window = vec![0.0f32; win_len];
    let first_real = ((cushion as f64) / (hw_f / eng_f)).ceil() as i64; // Z = ceil(C / r)

    let n_cb = (secs * hw_f / buf as f64).ceil() as usize;
    let mut hist: [Vec<f32>; 3] = [vec![0.0; buf], vec![0.0; buf], vec![0.0; buf]];
    let mut pushed: i64 = 0;
    let mut min_margin = i64::MAX;
    let mut cap_block = vec![[0.0f32; 2]; QUANTUM];

    for k in 0..n_cb {
        // ASIO bufferSwitch: 1. hardware input (ideal loopback of the output two callbacks ago:
        // the native-rate path would see it exactly 2 * buf frames later) -> rtrb.
        if k >= 2 {
            for &v in &hist[(k - 2) % 3] {
                tx.push([v, v]).expect("rtrb full");
            }
        } else {
            for _ in 0..buf {
                tx.push([0.0, 0.0]).expect("rtrb full");
            }
        }
        pushed += buf as i64;

        // 2. output SRC pulls engine blocks on demand; each engine block drains the bridge.
        let mut out_cur = std::mem::take(&mut hist[k % 3]);
        {
            let mut render = |b: u64| {
                bridge.drain_from_rtrb(&mut rx);
                let m_last = b as i64 + QUANTUM as i64 - 1;
                if m_last >= first_real {
                    // Kernel of engine frame m reads hw frames floor(m*r - C) .. +128.
                    let num = m_last * hw_sr as i64 - (cushion as i64) * ENGINE_SR as i64;
                    let x_floor = num.div_euclid(ENGINE_SR as i64);
                    min_margin = min_margin.min(pushed - (x_floor + 128));
                }
                bridge.process_output_block(&mut cap_block);
                for (i, f) in cap_block.iter().enumerate() {
                    let m = b + i as u64;
                    if m >= m0 && ((m - m0) as usize) < win_len {
                        window[(m - m0) as usize] = f[0];
                    }
                }
            };
            for slot in out_cur.iter_mut() {
                *slot = src.next(m0, &mut render);
            }
        }
        hist[k % 3] = out_cur;
    }

    let (pos, amp) = peak(&window);
    let r = hw_f / eng_f;
    AsioResult {
        hw_sr,
        buf,
        cushion,
        measured_extra: pos - 2.0 * buf as f64 / r,
        predicted_extra: AdaptiveDllDriftResampler::clock_locked_duplex_extra_latency_engine_frames(
            hw_f, eng_f, buf,
        ),
        peak_amp: amp,
        min_margin,
        underruns: bridge.underruns(),
        overruns: bridge.overruns(),
    }
}

#[test]
fn asio_src_clock_locked_bridge_60s_zero_xruns_and_exact_latency() {
    let rates = [44_100u32, 88_200, 96_000, 176_400, 192_000];
    let bufs = [64usize, 128, 256, 512, 1024];
    let mut jobs = Vec::new();
    for &sr in &rates {
        for &b in &bufs {
            jobs.push((sr, b));
        }
    }
    let results: Vec<AsioResult> = std::thread::scope(|s| {
        let handles: Vec<_> = jobs
            .iter()
            .map(|&(sr, b)| s.spawn(move || run_asio_clock_locked(sr, b, None, SIM_SECONDS)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    println!(
        "\nASIO+SRC clock-locked input bridge, {SIM_SECONDS} s per run (engine 48 kHz, 256-frame blocks)"
    );
    println!(
        "{:>7} {:>5} {:>8} {:>13} {:>13} {:>9} {:>6} {:>6} {:>6} {:>6}",
        "hw Hz", "buf", "cushion", "measured smp", "predicted smp", "extra ms", "peak", "margin", "under", "over"
    );
    let mut failures = Vec::new();
    for r in &results {
        println!(
            "{:>7} {:>5} {:>8} {:>13.2} {:>13.2} {:>9.3} {:>6.3} {:>6} {:>6} {:>6}",
            r.hw_sr,
            r.buf,
            r.cushion,
            r.measured_extra,
            r.predicted_extra,
            r.measured_extra * 1000.0 / ENGINE_SR as f64,
            r.peak_amp,
            r.min_margin,
            r.underruns,
            r.overruns
        );
        if r.underruns != 0 || r.overruns != 0 {
            failures.push(format!("{} Hz / {}: xruns u={} o={}", r.hw_sr, r.buf, r.underruns, r.overruns));
        }
        if r.min_margin < 0 {
            failures.push(format!("{} Hz / {}: negative margin {}", r.hw_sr, r.buf, r.min_margin));
        }
        if (r.measured_extra - r.predicted_extra).abs() > 0.25 {
            failures.push(format!(
                "{} Hz / {}: measured {:.3} vs predicted {:.3}",
                r.hw_sr, r.buf, r.measured_extra, r.predicted_extra
            ));
        }
        if r.peak_amp.abs() < 0.5 {
            failures.push(format!("{} Hz / {}: impulse lost (peak {})", r.hw_sr, r.buf, r.peak_amp));
        }
    }
    // Tightness: the cushion is the exact minimum for every (rate, buffer) plus the guard, so the
    // smallest observed headroom must be exactly the guard.
    let guard = daw_core_graph::audio_io::CLOCK_LOCKED_GUARD_FRAMES as i64;
    for r in &results {
        assert!(
            (0..=guard).contains(&r.min_margin),
            "{} Hz / {}: cushion is not minimal (headroom {} frames, guard {guard})",
            r.hw_sr,
            r.buf,
            r.min_margin
        );
    }
    for &sr in &rates {
        let any = AdaptiveDllDriftResampler::clock_locked_cushion_frames(sr as f64, 48_000.0, 0);
        println!("  {sr} Hz: buffer-independent bound would be {any} hw frames");
    }
    assert!(failures.is_empty(), "clock-locked bridge failures:\n{}", failures.join("\n"));
}

/// Negative control: one frame less than the computed minimum (guard removed) must underrun for
/// every rate/buffer pair, proving the per-buffer cushion (and the simulation) are exact.
#[test]
fn asio_src_clock_locked_bridge_undersized_cushion_underruns() {
    let jobs: Vec<(u32, usize)> = [44_100u32, 88_200, 96_000, 176_400, 192_000]
        .iter()
        .flat_map(|&sr| [64usize, 128, 256, 512, 1024].map(move |b| (sr, b)))
        .collect();
    let results: Vec<AsioResult> = std::thread::scope(|s| {
        let hs: Vec<_> = jobs
            .iter()
            .map(|&(sr, b)| {
                s.spawn(move || {
                    let c = AdaptiveDllDriftResampler::clock_locked_cushion_frames(sr as f64, 48_000.0, b);
                    let g = daw_core_graph::audio_io::CLOCK_LOCKED_GUARD_FRAMES;
                    run_asio_clock_locked(sr, b, Some(c - g - 1), 8.0)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for r in &results {
        assert!(
            r.underruns > 0,
            "{} Hz / {}: cushion minus guard minus 1 should underrun",
            r.hw_sr,
            r.buf
        );
    }
}

// ================================================================================================
// 2. WASAPI capture (separate clock) DLL bridge
// ================================================================================================

struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

struct DllCase {
    label: &'static str,
    cap_sr: u32,
    cap_ppm: f64,
    out_sr: u32,
    out_buf: usize,
    phase_ms: f64,
}

struct DllResult {
    label: &'static str,
    cap_sr: u32,
    phase_ms: f64,
    target: f64,
    avg_fill: f64,
    min_fill: usize,
    measured_extra: f64,
    predicted_extra: f64,
    xruns: u64,
    ppm: f64,
}

fn bl_impulse(x: f64) -> f64 {
    if x.abs() >= 32.0 {
        return 0.0;
    }
    let cutoff = 0.9;
    let u = x / 32.0;
    let w = bessel_i0(8.0 * (1.0 - u * u).max(0.0).sqrt()) / bessel_i0(8.0);
    let a = PI * cutoff * x;
    let s = if a.abs() < 1e-12 { cutoff } else { a.sin() / (PI * x) };
    s * w
}

fn run_wasapi_dll(case: &DllCase, seed: u64) -> DllResult {
    let eng_f = ENGINE_SR as f64;
    let cap_nom = case.cap_sr as f64;
    let cap_true = cap_nom * (1.0 + case.cap_ppm * 1e-6);
    let out_f = case.out_sr as f64;
    let packet = (case.cap_sr / 100) as usize; // 10 ms shared-mode packets
    let phase = case.phase_ms * 1e-3;
    let m0: u64 = 45 * ENGINE_SR as u64 + 101;

    // Pass 1: output-only (deterministic schedule) -> hardware position of the impulse.
    let q_imp = {
        let mut src = OutSrcReplica::new(ENGINE_SR, case.out_sr);
        let q_guess = ((m0 + 1) as f64 * out_f / eng_f) as usize;
        let lo = q_guess - 400;
        let mut w = vec![0.0f32; 800];
        let mut nop = |_b: u64| {};
        for q in 0..(lo + 800) {
            let v = src.next(m0, &mut nop);
            if q >= lo {
                w[q - lo] = v;
            }
        }
        lo as f64 + peak(&w).0
    };
    // Capture frame index (true capture clock) seeing the impulse: ideal acoustic loopback at
    // the hardware play time `q / out_sr` (device buffers are accounted for by the host terms).
    let j_imp = (q_imp / out_f - phase) * cap_true;

    let mut bridge = AdaptiveDllDriftResampler::new(cap_nom, eng_f);
    assert_eq!(bridge.mode(), CaptureBridgeMode::AdaptiveDll);
    let target = bridge.target_fill_frames();
    let (mut tx, mut rx) = rtrb::RingBuffer::<[f32; 2]>::new(1 << 16);
    let mut src = OutSrcReplica::new(ENGINE_SR, case.out_sr);
    let mut rng = Rng(seed | 1);

    let win_len = 8_000usize;
    let mut window = vec![0.0f32; win_len];
    let mut cap_block = vec![[0.0f32; 2]; QUANTUM];
    let mut fill_sum = 0.0f64;
    let mut fill_n = 0u64;
    let mut min_fill = usize::MAX;
    let measure_from = (25.0 * eng_f) as u64;

    let mut next_packet: usize = 0;
    let mut late_carry = 0.0f64;
    // Capture delivery time of a packet: capture of its last frame + scheduling jitter (0..1 ms
    // always, plus 2% of wake-ups up to 20 ms late = one full endpoint buffer; packets that became
    // due meanwhile are released in the same burst).
    let schedule = |p: usize, last: f64, rng: &mut Rng, carry: &mut f64| -> f64 {
        let nominal = phase + ((p + 1) * packet) as f64 / cap_true;
        if *carry <= 0.0 && rng.next_f64() < 0.02 {
            *carry = rng.next_f64() * 0.020;
        }
        let t = (nominal + rng.next_f64() * 0.001 + carry.max(0.0)).max(last);
        *carry -= packet as f64 / cap_true;
        t
    };
    let mut t_pkt = schedule(0, 0.0, &mut rng, &mut late_carry);
    let mut out_cb: usize = 0;
    let mut out_buf = vec![0.0f32; case.out_buf];
    loop {
        let t_out = (out_cb * case.out_buf) as f64 / out_f;
        if t_out > SIM_SECONDS && t_pkt > SIM_SECONDS {
            break;
        }
        if t_pkt <= t_out {
            for i in 0..packet {
                let j = (next_packet * packet + i) as f64;
                let v = bl_impulse(j - j_imp) as f32;
                tx.push([v, v]).expect("capture rtrb full");
            }
            next_packet += 1;
            t_pkt = schedule(next_packet, t_pkt, &mut rng, &mut late_carry);
        } else {
            let mut render = |b: u64| {
                bridge.drain_from_rtrb(&mut rx);
                if b >= measure_from {
                    fill_sum += bridge.available_frames() as f64;
                    fill_n += 1;
                    min_fill = min_fill.min(bridge.available_frames());
                }
                bridge.process_output_block(&mut cap_block);
                for (i, f) in cap_block.iter().enumerate() {
                    let m = b + i as u64;
                    if m >= m0 && ((m - m0) as usize) < win_len {
                        window[(m - m0) as usize] = f[0];
                    }
                }
            };
            for slot in out_buf.iter_mut() {
                *slot = src.next(m0, &mut render);
            }
            out_cb += 1;
        }
    }

    let (pos, _amp) = peak(&window);
    DllResult {
        label: case.label,
        cap_sr: case.cap_sr,
        phase_ms: case.phase_ms,
        target,
        avg_fill: fill_sum / fill_n.max(1) as f64,
        min_fill,
        measured_extra: pos,
        predicted_extra: AdaptiveDllDriftResampler::dll_input_delay_engine_frames(cap_nom, eng_f)
            + AdaptiveDllDriftResampler::render_lookahead_engine_frames(
                ENGINE_SR,
                case.out_sr,
                case.out_buf as u32,
            ),
        xruns: bridge.total_xruns(),
        ppm: bridge.current_ppm_offset(),
    }
}

#[test]
fn wasapi_capture_dll_bridge_fixed_target_zero_xruns_and_predictable_latency() {
    let mut cases = Vec::new();
    for &phase_ms in &[0.0, 3.3, 7.7] {
        cases.push(DllCase { label: "cap 48k +100ppm -> out 48k (B=480)", cap_sr: 48_000, cap_ppm: 100.0, out_sr: 48_000, out_buf: 480, phase_ms });
        cases.push(DllCase { label: "cap 48k -150ppm -> out 44.1k SRC (B=441)", cap_sr: 48_000, cap_ppm: -150.0, out_sr: 44_100, out_buf: 441, phase_ms });
        cases.push(DllCase { label: "cap 44.1k +50ppm -> out 48k (B=480)", cap_sr: 44_100, cap_ppm: 50.0, out_sr: 48_000, out_buf: 480, phase_ms });
        cases.push(DllCase { label: "cap 96k -30ppm -> out 48k (B=480)", cap_sr: 96_000, cap_ppm: -30.0, out_sr: 48_000, out_buf: 480, phase_ms });
        cases.push(DllCase { label: "cap 48k +20ppm -> ASIO 96k SRC (B=256)", cap_sr: 48_000, cap_ppm: 20.0, out_sr: 96_000, out_buf: 256, phase_ms });
    }
    let results: Vec<DllResult> = std::thread::scope(|s| {
        let hs: Vec<_> = cases
            .iter()
            .enumerate()
            .map(|(i, c)| s.spawn(move || run_wasapi_dll(c, 0x9E37_79B9_7F4A_7C15 ^ (i as u64 * 7919))))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });

    println!("\nWASAPI capture DLL bridge, {SIM_SECONDS} s per run (10 ms packets, 2% up to 20 ms late)");
    println!(
        "{:<42} {:>5} {:>7} {:>8} {:>6} {:>10} {:>10} {:>9} {:>8} {:>5}",
        "case", "phase", "target", "avg fill", "min", "meas smp", "pred smp", "resid ms", "ppm", "xrun"
    );
    let mut failures = Vec::new();
    for r in &results {
        let resid = r.measured_extra - r.predicted_extra;
        println!(
            "{:<42} {:>5.1} {:>7.0} {:>8.1} {:>6} {:>10.1} {:>10.1} {:>9.2} {:>8.2} {:>5}",
            r.label,
            r.phase_ms,
            r.target,
            r.avg_fill,
            r.min_fill,
            r.measured_extra,
            r.predicted_extra,
            resid * 1000.0 / ENGINE_SR as f64,
            r.ppm,
            r.xruns
        );
        if r.xruns != 0 {
            failures.push(format!("{} phase {}: {} xruns", r.label, r.phase_ms, r.xruns));
        }
        // When a 10 ms capture packet beats across a 10 ms callback boundary mid-window, a step of
        // one full packet P passes through a slewed controller bounded by +/-500 ppm, yielding a
        // minimum physical triangular deviation of P^2 / (2 * max_slew * window_sec) plus steady jitter.
        let pkt = (r.cap_sr / 100) as f64;
        let max_slew = 500e-6 * (r.cap_sr as f64);
        let beat_wrap_bound = (pkt * pkt) / (2.0 * max_slew * (SIM_SECONDS - 25.0)) + 0.025 * r.target;
        if (r.avg_fill - r.target).abs() > beat_wrap_bound {
            failures.push(format!("{}: avg fill {:.1} vs target {:.0} (bound {:.1})", r.label, r.avg_fill, r.target, beat_wrap_bound));
        }
        // Residual = capture-packet phase (0..10 ms) + mean delivery jitter (~0.5 ms + bursts).
        let resid_ms = resid * 1000.0 / ENGINE_SR as f64;
        if !(-1.0..=13.0).contains(&resid_ms) {
            failures.push(format!("{} phase {}: residual {:.2} ms out of range", r.label, r.phase_ms, resid_ms));
        }
    }
    assert!(failures.is_empty(), "DLL bridge failures:\n{}", failures.join("\n"));
}
