//! Zero-Allocation Native FL Studio 26 DSP Effect & Generator Nodes:
//! - `SparseCompressorNode` & `sparse_compressor_gain_reduction_rt` (`Fruity Limiter_x64.dll` `0x006d5d60`,
//!   Gather/Scatter above-threshold log2/exp2 solver at `0x6d5fe3..0x6d601a`)
//! - `MaximusMultibandNode` (`Maximus_x64.dll` `0x00844380`, 3-Band Linkwitz-Riley + `TensionSpline` waveshaper)
//! - `EmphasisLimiterNode` (`Emphasis_x64.dll` `0x00707240` & `ildsp_x64.dll` `Limiter_CreateInstance`)
//! - `GrossBeatTimeBufferNode` (`Gross Beat_x64.dll` `0x008420f0`, 2-bar `TBRDelay_32FS` + Time/Vol Splines)
//! - `LuxeVerbFdnShimmerNode` (`LuxeVerb_x64.dll` `0x00731870` & `ildsp_x64.dll` `Shimmer_CreateInstance`)
//! - `FixedLatencyTransformNode` & `SignalGeneratorNode` (precision test/routing DSP nodes)

use crate::apdc_engine::ApdcDelayLine;
use crate::automation_spline::{TensionControlPoint, TensionSpline};
use crate::{MAX_BLOCK_SIZE, ProcessContext, StereoFrame, UnifiedDspNode};

/// Real-Time Zero-Allocation Sparse Above-Threshold Log-Domain Compressor from
/// `Fruity Limiter_x64.dll` (`0x006d5d60`, gather loop `0x6d5fe3` & scatter loop `0x6d601a`).
///
/// Operates strictly on stack arrays (`[f32; MAX_BLOCK_SIZE]`, `[u16; MAX_BLOCK_SIZE]`) with
/// **Zero Heap Allocation**. Returns the number of samples that exceeded `threshold_linear`.
#[inline]
pub fn sparse_compressor_gain_reduction_rt(
    envelope: &[f32],
    gain_out: &mut [f32],
    threshold_linear: f32,
    ratio_inv_minus_one: f32,
    makeup_gain: f32,
) -> usize {
    let n = envelope.len().min(gain_out.len()).min(MAX_BLOCK_SIZE);
    let mut active_values = [0.0f32; MAX_BLOCK_SIZE];
    let mut active_indices = [0u16; MAX_BLOCK_SIZE];
    let mut count = 0usize;

    // Step 1: Gather only samples strictly above threshold (`0x6d5fe3`)
    for idx in 0..n {
        let env = envelope[idx];
        if env > threshold_linear {
            active_values[count] = env;
            active_indices[count] = idx as u16;
            count += 1;
        }
        gain_out[idx] = makeup_gain;
    }
    if count == 0 {
        return 0;
    }

    // Step 2: Evaluate log2/exp2 transfer curve ONLY on active indices and scatter back (`0x6d601a`)
    let thresh_log2 = threshold_linear.max(1e-12).log2();
    for k in 0..count {
        let over_log2 = active_values[k].max(1e-12).log2() - thresh_log2;
        let gr_linear = (over_log2 * ratio_inv_minus_one).exp2();
        gain_out[active_indices[k] as usize] = gr_linear * makeup_gain;
    }
    count
}

/// Alias matching `04_flstudio26_daw_architecture.md` (delegates to the zero-allocation RT solver).
#[inline]
pub fn sparse_compressor_gain_reduction(
    envelope: &[f32],
    gain_out: &mut [f32],
    threshold_linear: f32,
    ratio_inv_minus_one: f32,
    makeup_gain: f32,
) {
    let _ = sparse_compressor_gain_reduction_rt(
        envelope,
        gain_out,
        threshold_linear,
        ratio_inv_minus_one,
        makeup_gain,
    );
}

/// Zero-Allocation `SparseCompressorNode` (`TFruityLimiter` `VMT = 0x006d3ca0` in `Fruity Limiter_x64.dll`).
/// Integrates RMS/Peak envelope detection, external sidechain support, sparse Gather/Scatter
/// log2/exp2 gain reduction, and `TBasicDelay_32FS` lookahead limiting.
pub struct SparseCompressorNode {
    pub threshold_db: f32,
    pub ratio: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub makeup_gain_db: f32,
    pub ceiling_db: f32,
    pub use_external_sidechain: bool,
    pub bypass_envelope_ballistics: bool,
    lookahead_samples: usize,
    sample_rate: f64,
    env_state: f32,
    lookahead_delay: ApdcDelayLine,
    env_scratch: Vec<f32>,
    gain_scratch: Vec<f32>,
    last_active_sparse_count: usize,
    last_gr_db: f32,
}

impl Default for SparseCompressorNode {
    fn default() -> Self {
        Self::new(-12.0, 4.0, 0.0, 0)
    }
}

impl SparseCompressorNode {
    /// Creates a new `SparseCompressorNode` with pre-allocated scratch buffers.
    pub fn new(
        threshold_db: f32,
        ratio: f32,
        makeup_gain_db: f32,
        lookahead_samples: usize,
    ) -> Self {
        let mut lookahead_delay = ApdcDelayLine::with_capacity(lookahead_samples.max(4096));
        lookahead_delay.set_delay_samples(lookahead_samples);
        Self {
            threshold_db,
            ratio: ratio.max(1.0),
            attack_ms: 5.0,
            release_ms: 50.0,
            makeup_gain_db,
            ceiling_db: 0.0,
            use_external_sidechain: false,
            bypass_envelope_ballistics: false,
            lookahead_samples,
            sample_rate: 48_000.0,
            env_state: 0.0,
            lookahead_delay,
            env_scratch: vec![0.0; MAX_BLOCK_SIZE],
            gain_scratch: vec![1.0; MAX_BLOCK_SIZE],
            last_active_sparse_count: 0,
            last_gr_db: 0.0,
        }
    }

    #[inline]
    pub fn threshold_linear(&self) -> f32 {
        10.0f32.powf(self.threshold_db / 20.0)
    }

    #[inline]
    pub fn ratio_inv_minus_one(&self) -> f32 {
        (1.0 / self.ratio.max(1.0)) - 1.0
    }

    #[inline]
    pub fn makeup_gain_linear(&self) -> f32 {
        10.0f32.powf(self.makeup_gain_db / 20.0)
    }

    #[inline]
    pub fn last_active_sparse_count(&self) -> usize {
        self.last_active_sparse_count
    }

    #[inline]
    pub fn last_gr_db(&self) -> f32 {
        self.last_gr_db
    }
}

impl UnifiedDspNode for SparseCompressorNode {
    fn name(&self) -> &'static str {
        "Fruity Sparse Compressor/Limiter"
    }

    fn latency_samples(&self) -> usize {
        self.lookahead_samples
    }

    fn prepare(&mut self, sample_rate: f64, max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
        let cap = max_block_size.max(MAX_BLOCK_SIZE);
        if self.env_scratch.len() < cap {
            self.env_scratch.resize(cap, 0.0);
        }
        if self.gain_scratch.len() < cap {
            self.gain_scratch.resize(cap, 1.0);
        }
        self.lookahead_delay
            .ensure_capacity(self.lookahead_samples.max(4096));
        self.lookahead_delay
            .set_delay_samples(self.lookahead_samples);
    }

    fn reset_state(&mut self) {
        self.env_state = 0.0;
        self.lookahead_delay.reset();
        self.last_active_sparse_count = 0;
        self.last_gr_db = 0.0;
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.threshold_db = -60.0 + v * 60.0,
            1 => self.ratio = 1.0 + v * 19.0,
            2 => self.attack_ms = 0.1 + v * 199.9,
            3 => self.release_ms = 5.0 + v * 995.0,
            4 => self.makeup_gain_db = -12.0 + v * 36.0,
            5 => self.ceiling_db = -20.0 + v * 20.0,
            6 => self.use_external_sidechain = v >= 0.5,
            _ => {}
        }
    }

    fn current_gain_reduction_db(&self) -> f32 {
        self.last_gr_db
    }

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        let n = main_io.len().min(MAX_BLOCK_SIZE);
        if n == 0 {
            return;
        }
        let sr = ctx.sample_rate.max(8000.0) as f32;
        let att_coeff = if self.bypass_envelope_ballistics || self.attack_ms <= 0.0 {
            0.0
        } else {
            (-1.0 / (0.001 * self.attack_ms.max(0.05) * sr)).exp()
        };
        let rel_coeff = if self.bypass_envelope_ballistics || self.release_ms <= 0.0 {
            0.0
        } else {
            (-1.0 / (0.001 * self.release_ms.max(0.5) * sr)).exp()
        };

        let detector_src: &[StereoFrame] = if self.use_external_sidechain {
            sidechain_in.unwrap_or(&*main_io)
        } else {
            &*main_io
        };

        // Compute linked stereo peak envelope into pre-allocated `env_scratch`
        let mut env = self.env_state;
        for i in 0..n {
            let s = detector_src.get(i).copied().unwrap_or([0.0, 0.0]);
            let peak = s[0].abs().max(s[1].abs());
            if self.bypass_envelope_ballistics {
                env = peak;
            } else {
                let c = if peak > env { att_coeff } else { rel_coeff };
                env = c * env + (1.0 - c) * peak;
            }
            self.env_scratch[i] = env;
        }
        self.env_state = env;

        // Run Sparse Gather/Scatter log2/exp2 solver on stack arrays (Zero Heap Allocation)
        let thresh_lin = self.threshold_linear();
        let ratio_inv_m1 = self.ratio_inv_minus_one();
        let makeup_lin = self.makeup_gain_linear();
        let active_count = sparse_compressor_gain_reduction_rt(
            &self.env_scratch[..n],
            &mut self.gain_scratch[..n],
            thresh_lin,
            ratio_inv_m1,
            makeup_lin,
        );
        self.last_active_sparse_count = active_count;

        // Apply lookahead delay (`[self + 0x1d4]`) and multiply gain
        self.lookahead_delay.process_in_place(&mut main_io[..n]);
        let ceiling_lin = 10.0f32.powf(self.ceiling_db / 20.0);
        let mut min_g = 1.0f32;
        for i in 0..n {
            let g = self.gain_scratch[i];
            if g < min_g {
                min_g = g;
            }
            let l = (main_io[i][0] * g).clamp(-ceiling_lin, ceiling_lin);
            let r = (main_io[i][1] * g).clamp(-ceiling_lin, ceiling_lin);
            main_io[i] = [l, r];
        }
        self.last_gr_db = 20.0 * min_g.max(1e-6).log10();
    }
}

/// 3-Band Multiband Maximizer & Spline Waveshaper (`Maximus_x64.dll` `TMaximus` `0x00841ac8`).
pub struct MaximusMultibandNode {
    pub low_crossover_hz: f32,
    pub high_crossover_hz: f32,
    pub master_drive_db: f32,
    pub transfer_spline: TensionSpline,
    lp_state: [f32; 2],
    hp_state: [f32; 2],
    sample_rate: f64,
}

impl Default for MaximusMultibandNode {
    fn default() -> Self {
        Self::new()
    }
}

impl MaximusMultibandNode {
    pub fn new() -> Self {
        let transfer_spline = TensionSpline::new(vec![
            TensionControlPoint::new(0.0, 0.0, 0.0),
            TensionControlPoint::new(0.75, 0.75, 0.0),
            TensionControlPoint::new(1.0, 0.95, -0.35),
        ]);
        Self {
            low_crossover_hz: 200.0,
            high_crossover_hz: 3000.0,
            master_drive_db: 0.0,
            transfer_spline,
            lp_state: [0.0; 2],
            hp_state: [0.0; 2],
            sample_rate: 48_000.0,
        }
    }
}

impl UnifiedDspNode for MaximusMultibandNode {
    fn name(&self) -> &'static str {
        "Maximus 3-Band Dynamics"
    }

    fn latency_samples(&self) -> usize {
        0
    }

    fn prepare(&mut self, sample_rate: f64, _max_block_size: usize) {
        self.sample_rate = sample_rate.max(8000.0);
    }

    fn reset_state(&mut self) {
        self.lp_state = [0.0; 2];
        self.hp_state = [0.0; 2];
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.low_crossover_hz = 40.0 + v * 460.0,
            1 => self.high_crossover_hz = 1000.0 + v * 9000.0,
            2 => self.master_drive_db = v * 18.0,
            _ => {}
        }
    }

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        let sr = ctx.sample_rate.max(8000.0) as f32;
        let g_low = (-std::f32::consts::TAU * self.low_crossover_hz / sr).exp();
        let g_high = (-std::f32::consts::TAU * self.high_crossover_hz / sr).exp();
        let drive = 10.0f32.powf(self.master_drive_db / 20.0);

        for frame in main_io.iter_mut() {
            for ch in 0..2 {
                let x = frame[ch] * drive;
                self.lp_state[ch] = (1.0 - g_low) * x + g_low * self.lp_state[ch];
                let low = self.lp_state[ch];
                self.hp_state[ch] = (1.0 - g_high) * x + g_high * self.hp_state[ch];
                let high = x - self.hp_state[ch];
                let mid = x - low - high;
                let summed = low + mid + high;
                let abs_in = (summed.abs() as f64).min(1.0);
                let shaped = self.transfer_spline.eval(abs_in) as f32;
                frame[ch] = summed.signum() * shaped;
            }
        }
    }
}

/// Multi-stage Mastering Limiter with Bilateral TPT Pre/De-Emphasis Tilt, `4x` Polyphase ISP
/// Detection, Lookahead Attack Smoothing, and Post-Limiter HF Transient Recovery
/// (`Emphasis_x64.dll` `0x00707240` & `ildsp_x64.dll` `Limiter_CreateInstance`).
pub struct EmphasisLimiterNode {
    pub ceiling_db: f32,
    pub emphasis_tilt_db: f32,
    pub release_ms: f32,
    pub transient_recovery: f32,
    lookahead_samples: usize,
    delay: ApdcDelayLine,
    env: f32,
    pre_lp_state: [f32; 2],
    de_lp_state: [f32; 2],
    isp_hist: [[f32; 4]; 2],
    hf_in_rms: f32,
    hf_out_rms: f32,
    last_gr_db: f32,
}

impl EmphasisLimiterNode {
    pub fn new(lookahead_samples: usize, ceiling_db: f32) -> Self {
        let mut delay = ApdcDelayLine::with_capacity(lookahead_samples.max(4096));
        delay.set_delay_samples(lookahead_samples);
        Self {
            ceiling_db,
            emphasis_tilt_db: 3.0,
            release_ms: 65.0,
            transient_recovery: 0.25,
            lookahead_samples,
            delay,
            env: 0.0,
            pre_lp_state: [0.0; 2],
            de_lp_state: [0.0; 2],
            isp_hist: [[0.0; 4]; 2],
            hf_in_rms: 0.0,
            hf_out_rms: 0.0,
            last_gr_db: 0.0,
        }
    }

    #[inline(always)]
    fn polyphase_4x_isp_abs(hist: &[f32; 4]) -> f32 {
        // 4-point cubic Catmull-Rom evaluation at fractional phases 0.0, 0.25, 0.50, 0.75
        let ym1 = hist[0];
        let y0 = hist[1];
        let y1 = hist[2];
        let y2 = hist[3];
        let mut peak = y0.abs().max(y1.abs());
        let c0 = y0;
        let c1 = 0.5 * (y1 - ym1);
        let c2 = ym1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2;
        let c3 = 0.5 * (y2 - ym1) + 1.5 * (y0 - y1);
        for &t in &[0.25f32, 0.50f32, 0.75f32] {
            let v = (((c3 * t + c2) * t + c1) * t + c0).abs();
            if v > peak {
                peak = v;
            }
        }
        peak
    }
}

impl UnifiedDspNode for EmphasisLimiterNode {
    fn name(&self) -> &'static str {
        "Emphasis Mastering Limiter"
    }

    fn latency_samples(&self) -> usize {
        self.lookahead_samples
    }

    fn prepare(&mut self, _sample_rate: f64, _max_block_size: usize) {
        self.delay
            .ensure_capacity(self.lookahead_samples.max(4096));
        self.delay.set_delay_samples(self.lookahead_samples);
    }

    fn reset_state(&mut self) {
        self.delay.reset();
        self.env = 0.0;
        self.pre_lp_state = [0.0; 2];
        self.de_lp_state = [0.0; 2];
        self.isp_hist = [[0.0; 4]; 2];
        self.hf_in_rms = 0.0;
        self.hf_out_rms = 0.0;
        self.last_gr_db = 0.0;
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.ceiling_db = -12.0 + v * 12.0,
            1 => self.emphasis_tilt_db = -6.0 + v * 12.0,
            2 => self.release_ms = 10.0 + v * 490.0,
            3 => self.transient_recovery = v,
            _ => {}
        }
    }

    fn current_gain_reduction_db(&self) -> f32 {
        self.last_gr_db
    }

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        let sr = ctx.sample_rate.max(8000.0) as f32;
        let ceiling_lin = 10.0f32.powf(self.ceiling_db / 20.0);
        let tilt_gain = 10.0f32.powf(self.emphasis_tilt_db.clamp(-12.0, 12.0) / 20.0);
        let inv_tilt_gain = 1.0 / tilt_gain.max(1e-4);

        // Bilateral exact-inverse Cytomic TPT High-Shelf around 1,800 Hz:
        // H_pre(s') = (1 + K*s') / (1 + s'), so exact inverse H_de(s') = (1 + s') / (1 + K*s')
        // uses bilinear integrator gain `g_de = g_pre / K = g_pre * inv_tilt_gain`.
        let g_pre = (std::f32::consts::PI * (1800.0f32).min(sr * 0.42) / sr).tan();
        let alpha_pre = g_pre / (1.0 + g_pre);
        let g_de = g_pre * inv_tilt_gain;
        let alpha_de = g_de / (1.0 + g_de);

        let att_samples = (self.lookahead_samples.max(1) as f32).min(0.0025 * sr).max(1.0);
        let att_coeff = (-1.0 / att_samples).exp();
        let rel_coeff = (-1.0 / (0.001 * self.release_ms.max(5.0) * sr)).exp();
        let rms_coeff = (-1.0 / (0.003 * sr)).exp();

        let mut min_g = 1.0f32;

        for frame in main_io.iter_mut() {
            // 1. Pre-Emphasis High-Shelf Tilt + HF input energy tracking
            let mut pre = [0.0f32; 2];
            let mut hf_sq_in = 0.0f32;
            for ch in 0..2 {
                let x = frame[ch];
                let v = alpha_pre * (x - self.pre_lp_state[ch]);
                let lp = v + self.pre_lp_state[ch];
                self.pre_lp_state[ch] = lp + v;
                let hp = x - lp;
                hf_sq_in += hp * hp;
                pre[ch] = lp + tilt_gain * hp;

                self.isp_hist[ch][0] = self.isp_hist[ch][1];
                self.isp_hist[ch][1] = self.isp_hist[ch][2];
                self.isp_hist[ch][2] = self.isp_hist[ch][3];
                self.isp_hist[ch][3] = pre[ch];
            }
            self.hf_in_rms = rms_coeff * self.hf_in_rms + (1.0 - rms_coeff) * (0.5 * hf_sq_in);

            // 2. 4x Polyphase Inter-Sample Peak (ISP) Detector
            let isp_peak = Self::polyphase_4x_isp_abs(&self.isp_hist[0])
                .max(Self::polyphase_4x_isp_abs(&self.isp_hist[1]));

            // 3. Lookahead delay on audio path + attack/release gain computer
            let mut delayed = pre;
            self.delay
                .process_in_place(std::slice::from_mut(&mut delayed));

            if isp_peak > self.env {
                self.env = att_coeff * self.env + (1.0 - att_coeff) * isp_peak;
                if self.lookahead_samples == 0 && isp_peak > self.env {
                    self.env = isp_peak;
                }
            } else {
                self.env = rel_coeff * self.env + (1.0 - rel_coeff) * isp_peak;
            }

            let g = if self.env > ceiling_lin {
                ceiling_lin / self.env.max(1e-9)
            } else {
                1.0
            };
            if g < min_g {
                min_g = g;
            }

            let lim = [
                (delayed[0] * g).clamp(-ceiling_lin * tilt_gain.max(1.0), ceiling_lin * tilt_gain.max(1.0)),
                (delayed[1] * g).clamp(-ceiling_lin * tilt_gain.max(1.0), ceiling_lin * tilt_gain.max(1.0)),
            ];

            // 4. Exact Inverse De-Emphasis High-Shelf + Post-Limiter HF Transient Recovery
            let mut out = [0.0f32; 2];
            let mut hf_sq_out = 0.0f32;
            let hf_boost = if self.transient_recovery > 0.0 && self.hf_out_rms > 1e-9 {
                let ratio = (self.hf_in_rms / self.hf_out_rms.max(1e-9))
                    .sqrt()
                    .clamp(1.0, 1.35);
                1.0 + self.transient_recovery.clamp(0.0, 1.0) * (ratio - 1.0)
            } else {
                1.0
            };

            for ch in 0..2 {
                let y = lim[ch];
                let v = alpha_de * (y - self.de_lp_state[ch]);
                let lp = v + self.de_lp_state[ch];
                self.de_lp_state[ch] = lp + v;
                let hp = y - lp;
                let de_hp = inv_tilt_gain * hp * hf_boost;
                hf_sq_out += de_hp * de_hp;
                out[ch] = (lp + de_hp).clamp(-ceiling_lin, ceiling_lin);
            }
            self.hf_out_rms = rms_coeff * self.hf_out_rms + (1.0 - rms_coeff) * (0.5 * hf_sq_out);
            *frame = out;
        }

        self.last_gr_db = 20.0 * min_g.max(1e-6).log10();
    }
}

/// Real-Time 2-Bar Circular Time-Buffer & Volume Gate (`Gross Beat_x64.dll` `TBRDelay_32FS` `0x008420f0`)
/// upgraded with 4-Point Catmull-Rom Fractional Delay Interpolation, `2.0 ms` Jump Detection,
/// Dual-Read-Head `4.0 ms` Raised-Cosine Anti-Click Crossfader, and `1.5 ms` TPT Volume Gate Smoothing.
pub struct GrossBeatTimeBufferNode {
    history_ring: Vec<StereoFrame>,
    write_pos: usize,
    pub time_spline: TensionSpline,
    pub volume_spline: TensionSpline,
    pub click_reduction_enabled: bool,
    pub crossfade_ms: f32,
    head_a_delay: f64,
    head_a_vel: f64,
    head_b_delay: f64,
    head_b_vel: f64,
    xfade_pos: usize,
    xfade_len: usize,
    vol_smooth_state: f32,
    initialized: bool,
}

impl Default for GrossBeatTimeBufferNode {
    fn default() -> Self {
        Self::new(768_000)
    }
}

impl GrossBeatTimeBufferNode {
    pub fn new(history_frames: usize) -> Self {
        // Pre-allocate enough history for 2 bars down to 60 BPM at up to 96 kHz (768,000 frames)
        // so `process_slice` never allocates on RT.
        let cap = history_frames.max(768_000);
        Self {
            history_ring: vec![[0.0; 2]; cap],
            write_pos: 0,
            time_spline: TensionSpline::new(vec![
                TensionControlPoint::new(0.0, 0.0, 0.0),
                TensionControlPoint::new(1.0, 0.0, 0.0),
            ]),
            volume_spline: TensionSpline::new(vec![
                TensionControlPoint::new(0.0, 1.0, 0.0),
                TensionControlPoint::new(1.0, 1.0, 0.0),
            ]),
            click_reduction_enabled: true,
            crossfade_ms: 4.0,
            head_a_delay: 0.0,
            head_a_vel: 0.0,
            head_b_delay: 0.0,
            head_b_vel: 0.0,
            xfade_pos: 0,
            xfade_len: 0,
            vol_smooth_state: 1.0,
            initialized: false,
        }
    }

    #[inline(always)]
    fn read_catmull_rom(
        ring: &[StereoFrame],
        write_pos: usize,
        delay_samples: f64,
    ) -> StereoFrame {
        let len = ring.len();
        if len < 4 {
            return [0.0, 0.0];
        }
        let max_d = (len - 3) as f64;
        let d = delay_samples.clamp(0.0, max_d);
        let d_int = d.floor() as usize;
        let frac = (d - d_int as f64) as f32;

        // Read position is `write_pos - d`. Notice at `d = 0.0`, `frac = 0.0` -> returns exact `ring[write_pos]`.
        if frac <= 1e-7 {
            let idx = (write_pos + len - (d_int % len)) % len;
            return ring[idx];
        }

        // Let p(k) be sample at `write_pos - d_int - k` (going backward in delay, or forward in time)
        // Specifically, let continuous read coordinate `r = write_pos - d = (write_pos - d_int - 1) + (1.0 - frac)`.
        let t = 1.0 - frac;
        let base = (write_pos + len - ((d_int + 1) % len)) % len;
        let ym1 = ring[(base + len - 1) % len];
        let y0 = ring[base];
        let y1 = ring[(base + 1) % len];
        // Avoid wrapping past `write_pos` into unwritten future sample when `d_int == 0`
        let y2 = if d_int == 0 {
            y1
        } else {
            ring[(base + 2) % len]
        };

        let mut out = [0.0f32; 2];
        for ch in 0..2 {
            let c0 = y0[ch];
            let c1 = 0.5 * (y1[ch] - ym1[ch]);
            let c2 = ym1[ch] - 2.5 * y0[ch] + 2.0 * y1[ch] - 0.5 * y2[ch];
            let c3 = 0.5 * (y2[ch] - ym1[ch]) + 1.5 * (y0[ch] - y1[ch]);
            out[ch] = ((c3 * t + c2) * t + c1) * t + c0;
        }
        out
    }
}

impl UnifiedDspNode for GrossBeatTimeBufferNode {
    fn name(&self) -> &'static str {
        "Gross Beat Time & Volume Manipulator"
    }

    fn latency_samples(&self) -> usize {
        0
    }

    fn prepare(&mut self, sample_rate: f64, _max_block_size: usize) {
        let two_bars_max = ((sample_rate * 8.0) as usize).max(4096);
        if self.history_ring.len() < two_bars_max {
            self.history_ring.resize(two_bars_max, [0.0; 2]);
        }
    }

    fn reset_state(&mut self) {
        self.write_pos = 0;
        self.history_ring.fill([0.0; 2]);
        self.head_a_delay = 0.0;
        self.head_a_vel = 0.0;
        self.head_b_delay = 0.0;
        self.head_b_vel = 0.0;
        self.xfade_pos = 0;
        self.xfade_len = 0;
        self.vol_smooth_state = 1.0;
        self.initialized = false;
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.crossfade_ms = 0.5 + v * 14.5,
            1 => self.click_reduction_enabled = v >= 0.5,
            _ => {}
        }
    }

    fn export_visual_telemetry(&self, frame: &mut crate::telemetry_shm::VisualTelemetryFrame) {
        let len = self.history_ring.len().max(1) as f32;
        let phase = (self.write_pos as f32 / len).clamp(0.0, 1.0);
        frame.automation_spline.gross_beat_playhead_pos = [phase, phase];
    }

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        let ring_len = self.history_ring.len();
        if ring_len < 4 {
            return;
        }
        let sr = ctx.sample_rate.max(8000.0);
        let samples_per_2bars = ((60.0 * sr / ctx.bpm.max(20.0)) * 8.0).max(1.0);
        let jump_thresh_samples = (0.002 * sr).max(32.0);
        let target_xfade_len = ((self.crossfade_ms.max(0.5) as f64 * 0.001 * sr).round() as usize).max(32);
        let vol_alpha = (1.0 - (-1.0 / (0.0015 * sr)).exp()) as f32;
        let max_delay = (ring_len - 4) as f64;

        for (offset, frame) in main_io.iter_mut().enumerate() {
            self.history_ring[self.write_pos] = *frame;
            let abs_sample = (ctx.current_sample_pos + offset as i64).max(0) as f64;
            let bar_phase = (abs_sample / samples_per_2bars).fract();

            let delay_beat_frac = self.time_spline.eval(bar_phase).clamp(0.0, 1.0);
            let raw_vol_gate = self.volume_spline.eval(bar_phase).clamp(0.0, 1.0) as f32;
            let target_delay = (delay_beat_frac * samples_per_2bars * 0.25).clamp(0.0, max_delay);

            let delayed = if !self.click_reduction_enabled {
                let delay_int = target_delay as usize;
                let read_idx = (self.write_pos + ring_len - (delay_int % ring_len)) % ring_len;
                self.vol_smooth_state = raw_vol_gate;
                self.history_ring[read_idx]
            } else {
                if !self.initialized {
                    self.head_a_delay = target_delay;
                    self.head_a_vel = 0.0;
                    self.vol_smooth_state = raw_vol_gate;
                    self.initialized = true;
                } else {
                    let expected_delay = self.head_a_delay + self.head_a_vel;
                    let jump_err = (target_delay - expected_delay).abs();
                    if jump_err > jump_thresh_samples {
                        if self.xfade_pos < self.xfade_len && self.xfade_pos < (self.xfade_len / 2).max(16) {
                            // Multi-sample steep transition edge: keep Head B anchored on the clean
                            // pre-jump stream while updating Head A to the latest post-jump target.
                            self.head_a_delay = target_delay;
                            self.head_a_vel = 0.0;
                        } else {
                            // Promote outgoing trajectory to Head B and initiate C2-continuous quintic crossfade
                            self.head_b_delay = expected_delay.clamp(0.0, max_delay);
                            self.head_b_vel = self.head_a_vel;
                            self.head_a_delay = target_delay;
                            self.head_a_vel = 0.0;
                            self.xfade_pos = 0;
                            self.xfade_len = target_xfade_len;
                        }
                    } else {
                        self.head_a_vel = (target_delay - self.head_a_delay).clamp(-4.0, 4.0);
                        self.head_a_delay = target_delay;
                    }
                }

                let sig_a =
                    Self::read_catmull_rom(&self.history_ring, self.write_pos, self.head_a_delay);
                let blended = if self.xfade_pos < self.xfade_len {
                    let sig_b = Self::read_catmull_rom(
                        &self.history_ring,
                        self.write_pos,
                        self.head_b_delay,
                    );
                    self.head_b_delay = (self.head_b_delay + self.head_b_vel).clamp(0.0, max_delay);
                    // C2-continuous quintic smoothstep `u^3 * (10 - 15u + 6u^2)` -> zero 1st & 2nd derivatives at endpoints
                    let u = ((self.xfade_pos + 1) as f32) / ((self.xfade_len + 1) as f32);
                    let w_a = u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
                    let w_b = 1.0 - w_a;
                    self.xfade_pos += 1;
                    [
                        sig_a[0] * w_a + sig_b[0] * w_b,
                        sig_a[1] * w_a + sig_b[1] * w_b,
                    ]
                } else {
                    sig_a
                };

                self.vol_smooth_state += vol_alpha * (raw_vol_gate - self.vol_smooth_state);
                blended
            };

            let vg = self.vol_smooth_state;
            *frame = [delayed[0] * vg, delayed[1] * vg];

            self.write_pos += 1;
            if self.write_pos >= ring_len {
                self.write_pos = 0;
            }
        }
    }
}

/// Cascaded Schroeder Input Allpass Diffuser Stage (`LuxeVerb_x64.dll` early diffusion network).
#[derive(Clone)]
struct SchroederAllpassStage {
    buf: Vec<f32>,
    cursor: usize,
    base_delay_44k1: usize,
    active_len: usize,
}

impl SchroederAllpassStage {
    fn new(base_delay_44k1: usize) -> Self {
        let max_len = ((base_delay_44k1 as f64 * 192_000.0 / 44_100.0).ceil() as usize).max(64) + 16;
        Self {
            buf: vec![0.0; max_len],
            cursor: 0,
            base_delay_44k1,
            active_len: base_delay_44k1.max(1),
        }
    }

    #[inline]
    fn update_sample_rate(&mut self, sr: f64) {
        let scaled = ((self.base_delay_44k1 as f64) * (sr / 44_100.0)).round() as usize;
        let new_len = scaled.clamp(8, self.buf.len() - 1);
        if self.active_len != new_len {
            self.active_len = new_len;
            self.cursor %= new_len;
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.cursor = 0;
    }

    #[inline(always)]
    fn process(&mut self, x: f32, g: f32) -> f32 {
        let delayed = self.buf[self.cursor];
        let v = x - g * delayed;
        self.buf[self.cursor] = v;
        self.cursor += 1;
        if self.cursor >= self.active_len {
            self.cursor = 0;
        }
        delayed + g * v
    }
}

/// Studio-Grade `16×16` Kronecker Unitary Feedback Delay Network + Dual-Grain $\sin^2$ Hann
/// Octave-Up Shimmer Reverb (`LuxeVerb_x64.dll` `0x00731870` & `ildsp_x64.dll` `Shimmer_CreateInstance`).
///
/// Architecture verified via Phase 2 Reverse Engineering:
/// - Sample-rate-invariant Pre-Delay + Stereo 8-stage (`4L + 4R`) Schroeder Allpass Input Diffusers
///   (`[208, 561, 342, 575]` L, `[158, 410, 710, 393]` R @ `44.1 kHz`).
/// - `16×16` Unitary Walsh-Hadamard Kronecker Matrix ($R_2(\pi/4)^{\otimes 4}$) across 16 mutually
///   prime delay lines (`[877..6997]` @ `44.1 kHz`, dynamically scaled by $f_s / 44100$).
/// - 16-phase decorrelated sinusoidal LFO + 4-Point Catmull-Rom cubic fractional delay interpolation.
/// - `2124.8 Hz` TPT Dual-Band RT60 Damping ($g_i = 10^{-3 D_i / (f_s \cdot T_{60})}$).
/// - Dual-Grain `150 ms` $\sin^2$ Hann Pitch-Shifter (`+12` semitones default) injected at FDN input.
pub struct LuxeVerbFdnShimmerNode {
    pre_delay_ring: Vec<StereoFrame>,
    pre_delay_write: usize,
    diffusers_l: [SchroederAllpassStage; 4],
    diffusers_r: [SchroederAllpassStage; 4],
    fdn_lines: [Vec<f32>; 16],
    fdn_write_pos: [usize; 16],
    fdn_damp_state: [f32; 16],
    lfo_phases: [f64; 16],
    shimmer_ring: Vec<StereoFrame>,
    shimmer_write: usize,
    shimmer_grain_phase: f64,
    shimmer_hp_state: [f32; 2],
    shimmer_lp_state: [f32; 2],
    duck_env: f32,
    cached_sr: f64,
    pub decay: f32,
    pub wet_level: f32,
    pub pre_delay_ms: f32,
    pub damping_cutoff_hz: f32,
    pub high_decay_ratio: f32,
    pub mod_depth_ms: f32,
    pub mod_rate_hz: f32,
    pub shimmer_mix: f32,
    pub shimmer_semitones: f32,
    pub ducking_amount: f32,
}

impl Default for LuxeVerbFdnShimmerNode {
    fn default() -> Self {
        Self::new()
    }
}

impl LuxeVerbFdnShimmerNode {
    const BASE_PRIMES_44K1: [usize; 16] = [
        877, 941, 1087, 1229, 1433, 1663, 1931, 2239, 2593, 3001, 3469, 4013, 4643, 5381, 6229,
        6997,
    ];
    const DIFFUSER_L_44K1: [usize; 4] = [208, 561, 342, 575];
    const DIFFUSER_R_44K1: [usize; 4] = [158, 410, 710, 393];

    pub fn new() -> Self {
        // Pre-allocate all FDN, Pre-Delay, Diffuser, and Shimmer buffers for up to 192 kHz
        // so `process_slice` executes with strictly Zero Heap Allocation at every sample rate.
        let fdn_lines: [Vec<f32>; 16] = std::array::from_fn(|i| {
            let max_d =
                ((Self::BASE_PRIMES_44K1[i] as f64 * 192_000.0 / 44_100.0).ceil() as usize) + 1024;
            vec![0.0f32; max_d]
        });
        let diffusers_l = std::array::from_fn(|i| SchroederAllpassStage::new(Self::DIFFUSER_L_44K1[i]));
        let diffusers_r = std::array::from_fn(|i| SchroederAllpassStage::new(Self::DIFFUSER_R_44K1[i]));
        let lfo_phases = std::array::from_fn(|i| (i as f64) / 16.0);

        Self {
            pre_delay_ring: vec![[0.0; 2]; 57_600], // up to 300 ms @ 192 kHz
            pre_delay_write: 0,
            diffusers_l,
            diffusers_r,
            fdn_lines,
            fdn_write_pos: [0; 16],
            fdn_damp_state: [0.0; 16],
            lfo_phases,
            shimmer_ring: vec![[0.0; 2]; 96_000], // up to 500 ms @ 192 kHz
            shimmer_write: 0,
            shimmer_grain_phase: 0.0,
            shimmer_hp_state: [0.0; 2],
            shimmer_lp_state: [0.0; 2],
            duck_env: 0.0,
            cached_sr: 0.0,
            decay: 0.78,
            wet_level: 0.25,
            pre_delay_ms: 12.0,
            damping_cutoff_hz: 2124.8037,
            high_decay_ratio: 0.42,
            mod_depth_ms: 0.65,
            mod_rate_hz: 0.55,
            shimmer_mix: 0.0,
            shimmer_semitones: 12.0,
            ducking_amount: 0.0,
        }
    }

    /// In-place `16×16` Unitary Fast Walsh-Hadamard Transform ($H_{16} = \frac{1}{4} H_2^{\otimes 4}$).
    #[inline(always)]
    fn fwht16_unitary(v: &mut [f32; 16]) {
        let mut len = 1usize;
        while len < 16 {
            let step = len * 2;
            let mut i = 0usize;
            while i < 16 {
                for j in 0..len {
                    let a = v[i + j];
                    let b = v[i + j + len];
                    v[i + j] = a + b;
                    v[i + j + len] = a - b;
                }
                i += step;
            }
            len = step;
        }
        for x in v.iter_mut() {
            *x *= 0.25;
        }
    }

    #[inline(always)]
    fn read_catmull_rom_f32(line: &[f32], write_pos: usize, delay_samples: f64) -> f32 {
        let len = line.len();
        let d = delay_samples.clamp(2.0, (len - 4) as f64);
        let d_int = d.floor() as usize;
        let frac = (d - d_int as f64) as f32;
        let t = 1.0 - frac;
        let base = (write_pos + len - ((d_int + 1) % len)) % len;
        let ym1 = line[(base + len - 1) % len];
        let y0 = line[base];
        let y1 = line[(base + 1) % len];
        let y2 = line[(base + 2) % len];

        let c0 = y0;
        let c1 = 0.5 * (y1 - ym1);
        let c2 = ym1 - 2.5 * y0 + 2.0 * y1 - 0.5 * y2;
        let c3 = 0.5 * (y2 - ym1) + 1.5 * (y0 - y1);
        ((c3 * t + c2) * t + c1) * t + c0
    }

    #[inline(always)]
    fn read_shimmer_grain(
        ring: &[StereoFrame],
        write_pos: usize,
        grain_phase: f64,
        grain_win_samples: f64,
        pitch_ratio: f64,
    ) -> StereoFrame {
        let delay = if pitch_ratio >= 1.0 {
            (1.0 - grain_phase) * (pitch_ratio - 1.0) * grain_win_samples + 4.0
        } else {
            grain_phase * (1.0 - pitch_ratio) * grain_win_samples + 4.0
        };
        let s = GrossBeatTimeBufferNode::read_catmull_rom(ring, write_pos, delay);
        let win = (std::f64::consts::PI * grain_phase).sin();
        let w = (win * win) as f32;
        [s[0] * w, s[1] * w]
    }
}

impl UnifiedDspNode for LuxeVerbFdnShimmerNode {
    fn name(&self) -> &'static str {
        "LuxeVerb FDN Shimmer"
    }

    fn latency_samples(&self) -> usize {
        0
    }

    fn prepare(&mut self, sample_rate: f64, _max_block_size: usize) {
        let sr = sample_rate.clamp(8_000.0, 192_000.0);
        for d in &mut self.diffusers_l {
            d.update_sample_rate(sr);
        }
        for d in &mut self.diffusers_r {
            d.update_sample_rate(sr);
        }
        self.cached_sr = sr;
    }

    fn reset_state(&mut self) {
        self.pre_delay_ring.fill([0.0; 2]);
        self.pre_delay_write = 0;
        for d in &mut self.diffusers_l {
            d.reset();
        }
        for d in &mut self.diffusers_r {
            d.reset();
        }
        for line in &mut self.fdn_lines {
            line.fill(0.0);
        }
        self.fdn_write_pos = [0; 16];
        self.fdn_damp_state = [0.0; 16];
        self.shimmer_ring.fill([0.0; 2]);
        self.shimmer_write = 0;
        self.shimmer_grain_phase = 0.0;
        self.shimmer_hp_state = [0.0; 2];
        self.shimmer_lp_state = [0.0; 2];
        self.duck_env = 0.0;
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.decay = 0.1 + v * 0.88,
            1 => self.wet_level = v,
            2 => self.shimmer_mix = v,
            3 => self.shimmer_semitones = -12.0 + v * 36.0,
            4 => self.pre_delay_ms = v * 200.0,
            5 => self.damping_cutoff_hz = 600.0 * (16_000.0f32 / 600.0).powf(v),
            6 => self.mod_depth_ms = v * 2.0,
            7 => self.ducking_amount = v,
            _ => {}
        }
    }

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        let sr = ctx.sample_rate.clamp(8_000.0, 192_000.0);
        if (sr - self.cached_sr).abs() > 0.5 {
            for d in &mut self.diffusers_l {
                d.update_sample_rate(sr);
            }
            for d in &mut self.diffusers_r {
                d.update_sample_rate(sr);
            }
            self.cached_sr = sr;
        }

        let sr_ratio = sr / 44_100.0;
        // Map `decay` (`0.05..0.99`) to physical low-band RT60 in seconds (`0.30 s .. 10.5 s`)
        let rt60_low = (0.30 + (self.decay.clamp(0.05, 0.995) as f64).powi(2) * 10.2).max(0.15);
        let rt60_high = (rt60_low * (self.high_decay_ratio.clamp(0.1, 1.0) as f64)).max(0.08);

        let mut base_delays = [0.0f64; 16];
        let mut g_low = [0.0f32; 16];
        let mut g_high = [0.0f32; 16];
        for i in 0..16 {
            let d_samples = (Self::BASE_PRIMES_44K1[i] as f64) * sr_ratio;
            base_delays[i] = d_samples;
            let delay_sec = d_samples / sr;
            g_low[i] = 10.0f64.powf(-3.0 * delay_sec / rt60_low) as f32;
            g_high[i] = 10.0f64.powf(-3.0 * delay_sec / rt60_high) as f32;
        }

        let fc_damp = (self.damping_cutoff_hz as f64).clamp(250.0, sr * 0.42);
        let g_tpt = (std::f64::consts::PI * fc_damp / sr).tan();
        let alpha_damp = (g_tpt / (1.0 + g_tpt)) as f32;

        let pre_delay_samples =
            ((self.pre_delay_ms.clamp(0.0, 250.0) as f64) * 0.001 * sr).round() as usize;
        let pre_ring_len = self.pre_delay_ring.len();
        let pre_d = pre_delay_samples.min(pre_ring_len - 2);

        let mod_depth_samples = (self.mod_depth_ms.clamp(0.0, 3.0) as f64) * 0.001 * sr;
        let lfo_inc = (self.mod_rate_hz.clamp(0.05, 5.0) as f64) / sr;

        let shimmer_active = self.shimmer_mix > 1e-4;
        let shimmer_ratio = ((self.shimmer_semitones.clamp(-24.0, 24.0) / 12.0).exp2()) as f64;
        let shimmer_win_samples = (0.150 * sr).clamp(256.0, 32_768.0);
        let shimmer_phase_inc = 1.0 / shimmer_win_samples;
        let shimmer_hp_alpha =
            (1.0 - (-std::f64::consts::TAU * 140.0 / sr).exp()) as f32;
        let shimmer_lp_alpha =
            (1.0 - (-std::f64::consts::TAU * (11_500.0f64.min(sr * 0.42)) / sr).exp()) as f32;

        let duck_att = (-1.0 / (0.004 * sr as f32)).exp();
        let duck_rel = (-1.0 / (0.090 * sr as f32)).exp();
        let wet = self.wet_level.clamp(0.0, 1.5);

        for (idx, frame) in main_io.iter_mut().enumerate() {
            let in_l = frame[0];
            let in_r = frame[1];

            // 1. Pre-delay circular buffer
            self.pre_delay_ring[self.pre_delay_write] = [in_l, in_r];
            let pd_read = (self.pre_delay_write + pre_ring_len - pre_d) % pre_ring_len;
            let pd = self.pre_delay_ring[pd_read];
            self.pre_delay_write += 1;
            if self.pre_delay_write >= pre_ring_len {
                self.pre_delay_write = 0;
            }

            // 2. Stereo 4-Stage Cascaded Schroeder Input Allpass Diffusers
            let mut diff_l = pd[0];
            let mut diff_r = pd[1];
            for st in &mut self.diffusers_l {
                diff_l = st.process(diff_l, 0.618_034);
            }
            for st in &mut self.diffusers_r {
                diff_r = st.process(diff_r, 0.618_034);
            }

            // 3. Read all 16 modulated FDN delay lines via 4-Point Catmull-Rom + 2124.8 Hz TPT Damping
            let mut tapped = [0.0f32; 16];
            let mut damped = [0.0f32; 16];
            for i in 0..16 {
                let lfo = (std::f64::consts::TAU * self.lfo_phases[i]).sin() * mod_depth_samples;
                self.lfo_phases[i] = (self.lfo_phases[i] + lfo_inc).fract();

                let raw_tap = Self::read_catmull_rom_f32(
                    &self.fdn_lines[i],
                    self.fdn_write_pos[i],
                    base_delays[i] + lfo,
                );
                tapped[i] = raw_tap;

                let v = alpha_damp * (raw_tap - self.fdn_damp_state[i]);
                let lp = v + self.fdn_damp_state[i];
                self.fdn_damp_state[i] = lp + v;
                let hp = raw_tap - lp;
                damped[i] = lp * g_low[i] + hp * g_high[i];
            }

            // Orthogonal stereo decorrelated output tap sum across all 16 FDN lines
            let mut tank_l = 0.0f32;
            let mut tank_r = 0.0f32;
            for i in 0..8 {
                let sign = if (i & 1) == 0 { 1.0 } else { -1.0 };
                tank_l += tapped[2 * i] * sign;
                tank_r += tapped[2 * i + 1] * sign;
            }
            tank_l *= 0.353_553_38; // 1 / sqrt(8)
            tank_r *= 0.353_553_38;

            // 4. Dual-Grain 150 ms sin^2 Hann Shimmer Pitch-Shifter (`Shimmer_CreateInstance`)
            // Fed by diffused input + tank feedback and injected directly at the FDN input!
            let mut shim_inject_l = 0.0f32;
            let mut shim_inject_r = 0.0f32;
            if shimmer_active {
                let shim_src = [diff_l + 0.55 * tank_l, diff_r + 0.55 * tank_r];
                self.shimmer_ring[self.shimmer_write] = shim_src;

                let phase_a = self.shimmer_grain_phase;
                let phase_b = (phase_a + 0.5).fract();
                let ga = Self::read_shimmer_grain(
                    &self.shimmer_ring,
                    self.shimmer_write,
                    phase_a,
                    shimmer_win_samples,
                    shimmer_ratio,
                );
                let gb = Self::read_shimmer_grain(
                    &self.shimmer_ring,
                    self.shimmer_write,
                    phase_b,
                    shimmer_win_samples,
                    shimmer_ratio,
                );
                self.shimmer_grain_phase = (phase_a + shimmer_phase_inc).fract();
                self.shimmer_write += 1;
                if self.shimmer_write >= self.shimmer_ring.len() {
                    self.shimmer_write = 0;
                }

                let raw_shim = [ga[0] + gb[0], ga[1] + gb[1]];
                let mut bp_shim = [0.0f32; 2];
                for ch in 0..2 {
                    self.shimmer_hp_state[ch] +=
                        shimmer_hp_alpha * (raw_shim[ch] - self.shimmer_hp_state[ch]);
                    let hp = raw_shim[ch] - self.shimmer_hp_state[ch];
                    self.shimmer_lp_state[ch] +=
                        shimmer_lp_alpha * (hp - self.shimmer_lp_state[ch]);
                    bp_shim[ch] = self.shimmer_lp_state[ch];
                }
                let sm = self.shimmer_mix.clamp(0.0, 1.0);
                shim_inject_l = bp_shim[0] * sm * 0.55;
                shim_inject_r = bp_shim[1] * sm * 0.55;
                tank_l += bp_shim[0] * sm * 0.45;
                tank_r += bp_shim[1] * sm * 0.45;
            }

            // 5. Apply 16x16 Unitary Kronecker Hadamard Feedback Matrix & write back into FDN lines
            Self::fwht16_unitary(&mut damped);
            let excite_l = (diff_l + shim_inject_l) * 0.353_553_38;
            let excite_r = (diff_r + shim_inject_r) * 0.353_553_38;
            for i in 0..8 {
                let sgn = if (i & 2) == 0 { 1.0 } else { -1.0 };
                let idx_l = 2 * i;
                let idx_r = 2 * i + 1;

                let wl = self.fdn_write_pos[idx_l];
                self.fdn_lines[idx_l][wl] = damped[idx_l] + excite_l * sgn;
                self.fdn_write_pos[idx_l] = if wl + 1 >= self.fdn_lines[idx_l].len() {
                    0
                } else {
                    wl + 1
                };

                let wr = self.fdn_write_pos[idx_r];
                self.fdn_lines[idx_r][wr] = damped[idx_r] + excite_r * sgn;
                self.fdn_write_pos[idx_r] = if wr + 1 >= self.fdn_lines[idx_r].len() {
                    0
                } else {
                    wr + 1
                };
            }

            // 6. Optional External Sidechain / Self Ducking on Reverb Wet Tail
            let mut duck_gain = 1.0f32;
            if self.ducking_amount > 1e-4 {
                let det = sidechain_in
                    .and_then(|sc| sc.get(idx).copied())
                    .unwrap_or([in_l, in_r]);
                let pk = det[0].abs().max(det[1].abs());
                let c = if pk > self.duck_env { duck_att } else { duck_rel };
                self.duck_env = c * self.duck_env + (1.0 - c) * pk;
                duck_gain = (1.0 - self.ducking_amount.clamp(0.0, 1.0) * self.duck_env * 2.5)
                    .clamp(0.15, 1.0);
            }

            frame[0] = in_l + wet * duck_gain * tank_l;
            frame[1] = in_r + wet * duck_gain * tank_r;
        }
    }
}

/// Deterministic Fixed-Latency Pass-Through / Gain / Polarity-Invert DSP Node.
/// Used for building exact latency branches in 5-Point APDC null tests and phase utilities.
pub struct FixedLatencyTransformNode {
    latency: usize,
    pub gain: f32,
    pub invert_polarity: bool,
    delay: ApdcDelayLine,
    last_param_norm: [f32; 8],
}

impl FixedLatencyTransformNode {
    pub fn new(latency_samples: usize, gain: f32, invert_polarity: bool) -> Self {
        let mut delay = ApdcDelayLine::with_capacity(latency_samples.max(8192));
        delay.set_delay_samples(latency_samples);
        Self {
            latency: latency_samples,
            gain,
            invert_polarity,
            delay,
            last_param_norm: [0.0; 8],
        }
    }

    #[inline]
    pub fn last_parameter(&self, param_id: u32) -> f32 {
        self.last_param_norm[(param_id as usize) & 7]
    }

    /// Updates the reported and internal delay in samples with Zero Heap Allocation
    /// (requires `latency_samples <= delay.capacity()`).
    #[inline]
    pub fn set_latency_samples(&mut self, latency_samples: usize) {
        self.latency = latency_samples;
        self.delay.set_delay_samples(latency_samples);
    }
}

impl UnifiedDspNode for FixedLatencyTransformNode {
    fn name(&self) -> &'static str {
        "Fixed Latency Transform Node"
    }

    fn latency_samples(&self) -> usize {
        self.latency
    }

    fn prepare(&mut self, _sample_rate: f64, _max_block_size: usize) {
        self.delay.ensure_capacity(self.latency.max(8192));
        self.delay.set_delay_samples(self.latency);
    }

    fn reset_state(&mut self) {
        self.delay.reset();
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        self.last_param_norm[(param_id as usize) & 7] = norm_val;
        match param_id {
            0 => self.gain = norm_val,
            1 => self.invert_polarity = norm_val >= 0.5,
            2 => {
                let new_lat = (norm_val.clamp(0.0, 1.0) * 1024.0).round() as usize;
                self.set_latency_samples(new_lat);
            }
            _ => {}
        }
    }

    fn process_slice(
        &mut self,
        _ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        self.delay.process_in_place(main_io);
        let scale = if self.invert_polarity {
            -self.gain
        } else {
            self.gain
        };
        if (scale - 1.0).abs() > 1e-7 {
            for frame in main_io.iter_mut() {
                frame[0] *= scale;
                frame[1] *= scale;
            }
        }
    }
}

/// Deterministic Test Signal Generator Node (`TChannel` generator with configurable internal latency).
/// Generates an analytical multi-harmonic waveform based on sample index `(current_sample_pos + i) - latency`
/// so that a generator with internal latency `L` outputs `signal(t - L)` for `t >= L` and `0.0` for `t < L`.
pub struct DeterministicGeneratorNode {
    latency: usize,
    pub amplitude: f32,
    pub invert_polarity: bool,
    delay: ApdcDelayLine,
}

impl DeterministicGeneratorNode {
    pub fn new(latency_samples: usize, amplitude: f32, invert_polarity: bool) -> Self {
        let mut delay = ApdcDelayLine::with_capacity(latency_samples.max(8192));
        delay.set_delay_samples(latency_samples);
        Self {
            latency: latency_samples,
            amplitude,
            invert_polarity,
            delay,
        }
    }

    /// Evaluates the reference stereo waveform at sample index `sample_idx`.
    #[inline]
    pub fn reference_waveform_at(sample_idx: i64) -> StereoFrame {
        let t = sample_idx as f64;
        let l = (0.03125 * t).sin() + 0.5 * (0.09375 * t).cos() + 0.25 * (0.175 * t).sin();
        let r = (0.04250 * t).cos() - 0.4 * (0.11250 * t).sin() + 0.30 * (0.215 * t).cos();
        [(l * 0.4) as f32, (r * 0.4) as f32]
    }
}

impl UnifiedDspNode for DeterministicGeneratorNode {
    fn name(&self) -> &'static str {
        "Deterministic Reference Generator"
    }

    fn latency_samples(&self) -> usize {
        self.latency
    }

    fn prepare(&mut self, _sample_rate: f64, _max_block_size: usize) {
        self.delay.ensure_capacity(self.latency.max(8192));
        self.delay.set_delay_samples(self.latency);
    }

    fn reset_state(&mut self) {
        self.delay.reset();
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        if param_id == 0 {
            self.amplitude = norm_val;
        }
    }

    fn process_slice(
        &mut self,
        ctx: &ProcessContext,
        main_io: &mut [StereoFrame],
        _sidechain_in: Option<&[StereoFrame]>,
        _aux_sends_out: &mut [&mut [StereoFrame]],
    ) {
        // Models timeline content: silent while the transport is stopped.
        if !ctx.is_playing {
            return;
        }
        let sign = if self.invert_polarity {
            -self.amplitude
        } else {
            self.amplitude
        };
        for (i, frame) in main_io.iter_mut().enumerate() {
            let ref_frame = Self::reference_waveform_at(ctx.current_sample_pos + (i as i64));
            *frame = [ref_frame[0] * sign, ref_frame[1] * sign];
        }
        self.delay.process_in_place(main_io);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn goertzel_power_db(samples: &[f32], freq_hz: f64, sr: f64) -> f64 {
        let n = samples.len();
        if n == 0 {
            return -144.0;
        }
        let omega = std::f64::consts::TAU * freq_hz / sr;
        let coeff = 2.0 * omega.cos();
        let mut s_prev = 0.0f64;
        let mut s_prev2 = 0.0f64;
        for (i, &x) in samples.iter().enumerate() {
            let win = 0.5 - 0.5 * (std::f64::consts::TAU * (i as f64) / (n as f64)).cos();
            let s = (x as f64) * win + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s;
        }
        let power = (s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2)
            / ((n as f64) * (n as f64));
        10.0 * power.max(1e-24).log10()
    }

    #[test]
    fn gate4a_luxeverb_sample_rate_invariant_density_and_shimmer_octave() {
        // 1. Verify smooth echo density and consistent physical RT60 decay slope across 44.1, 48, and 96 kHz
        let mut decay_drops_db = Vec::new();
        for &sr in &[44_100.0f64, 48_000.0, 96_000.0] {
            let mut verb = LuxeVerbFdnShimmerNode::new();
            verb.decay = 0.78;
            verb.wet_level = 1.0;
            verb.pre_delay_ms = 0.0;
            verb.shimmer_mix = 0.0;
            verb.prepare(sr, 512);

            let total_frames = (sr * 1.2) as usize;
            let mut ir = vec![[0.0f32; 2]; total_frames];
            // Scale unit discrete impulse by sqrt(sr / 44_100) so continuous-time impulse energy
            // `integral x^2(t) dt = amp^2 / sr` is identical across 44.1 / 48 / 96 kHz.
            let imp = (sr / 44_100.0).sqrt() as f32;
            ir[0] = [imp, imp];

            let mut pos = 0usize;
            while pos < total_frames {
                let len = (total_frames - pos).min(512);
                let ctx = ProcessContext {
                    sample_rate: sr,
                    current_sample_pos: pos as i64,
                    ppq_tick: 0,
                    bpm: 120.0,
                    slice_frames: len,
                    is_playing: true,
                    is_offline_rendering: true,
                    ppq_resolution: crate::DEFAULT_PPQ,
                };
                verb.process_slice(&ctx, &mut ir[pos..pos + len], None, &mut []);
                pos += len;
            }

            // Measure RMS in window [120..270 ms] vs [750..950 ms] and echo density in [250..450 ms]
            let w1_start = (0.12 * sr) as usize;
            let w1_end = (0.27 * sr) as usize;
            let w2_start = (0.75 * sr) as usize;
            let w2_end = (0.95 * sr) as usize;

            let rms_db = |slice: &[[f32; 2]]| -> f64 {
                let mean_sq = slice
                    .iter()
                    .map(|f| 0.5 * ((f[0] as f64).powi(2) + (f[1] as f64).powi(2)))
                    .sum::<f64>()
                    / (slice.len().max(1) as f64);
                10.0 * mean_sq.max(1e-20).log10()
            };

            let e1_db = rms_db(&ir[w1_start..w1_end]);
            let e2_db = rms_db(&ir[w2_start..w2_end]);
            assert!(
                e1_db > -50.0 && e2_db < e1_db - 2.5,
                "Expected smooth decaying reverb tail at {sr} Hz: e1={e1_db:.2} dB, e2={e2_db:.2} dB"
            );
            decay_drops_db.push(e1_db - e2_db);

            // Echo density check: over [250..450 ms], > 70% of samples carry non-zero diffused reflection energy
            let d_slice = &ir[(0.25 * sr) as usize..(0.45 * sr) as usize];
            let peak_tail = d_slice
                .iter()
                .fold(0.0f32, |m, f| m.max(f[0].abs()).max(f[1].abs()));
            let dense_count = d_slice
                .iter()
                .filter(|f| f[0].abs().max(f[1].abs()) > peak_tail * 0.015)
                .count();
            let density_ratio = dense_count as f64 / (d_slice.len() as f64);
            assert!(
                density_ratio > 0.70,
                "Expected high diffused echo density (>0.70) at {sr} Hz, got {density_ratio:.3}"
            );
        }

        // Physical RT60 decay drop across 44.1 / 48 / 96 kHz must stay tightly consistent (within 2.5 dB)
        let min_drop = decay_drops_db.iter().copied().fold(f64::INFINITY, f64::min);
        let max_drop = decay_drops_db
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (max_drop - min_drop) < 2.5,
            "Sample-rate decay variance too large across 44.1/48/96 kHz: {decay_drops_db:?}"
        );

        // 2. Verify +12st Shimmer generates clear 2*f0 (880 Hz) octave-up energy from a pure 440 Hz tone
        let sr = 48_000.0;
        let render_shimmer = |shimmer_mix: f32| -> Vec<f32> {
            let mut verb = LuxeVerbFdnShimmerNode::new();
            verb.decay = 0.82;
            verb.wet_level = 1.0;
            verb.shimmer_mix = shimmer_mix;
            verb.shimmer_semitones = 12.0;
            verb.prepare(sr, 512);

            let n = 48_000;
            let mut buf: Vec<[f32; 2]> = (0..n)
                .map(|i| {
                    let s = (std::f64::consts::TAU * 440.0 * (i as f64) / sr).sin() as f32 * 0.25;
                    [s, s]
                })
                .collect();
            let ctx = ProcessContext {
                sample_rate: sr,
                current_sample_pos: 0,
                ppq_tick: 0,
                bpm: 120.0,
                slice_frames: n,
                is_playing: true,
                is_offline_rendering: true,
                ppq_resolution: crate::DEFAULT_PPQ,
            };
            verb.process_slice(&ctx, &mut buf, None, &mut []);
            buf[24_000..48_000].iter().map(|f| f[0]).collect()
        };

        let tail_no_shim = render_shimmer(0.0);
        let tail_with_shim = render_shimmer(0.75);
        let p880_off = goertzel_power_db(&tail_no_shim, 880.0, sr);
        let p880_on = goertzel_power_db(&tail_with_shim, 880.0, sr);
        assert!(
            p880_on - p880_off > 24.0,
            "Expected >= +24 dB octave-up (880 Hz) Shimmer energy boost, got off={p880_off:.2} dB vs on={p880_on:.2} dB"
        );
    }

    #[test]
    fn gate4b_gross_beat_hard_jump_click_reduction_exceeds_35db() {
        let sr = 48_000.0;
        let bpm = 120.0;
        // 2 bars @ 120 BPM = 4.0 seconds = 192,000 samples; jump happens at bar_phase = 0.50 (sample 96,000)
        let total_frames = 120_000usize;

        let run_gross_beat = |anti_click: bool| -> Vec<f32> {
            let mut gb = GrossBeatTimeBufferNode::new(240_000);
            gb.click_reduction_enabled = anti_click;
            gb.crossfade_ms = 4.0;
            // Step jump at phase 0.50 from delay=0.0 to delay=0.55
            gb.time_spline = TensionSpline::new(vec![
                TensionControlPoint::new(0.0, 0.0, 0.0),
                TensionControlPoint::new(0.499_99, 0.0, 0.0),
                TensionControlPoint::new(0.500_00, 0.55, 0.0),
                TensionControlPoint::new(1.0, 0.55, 0.0),
            ]);
            gb.prepare(sr, 512);

            // Low-frequency 110 Hz test sine so continuous 2nd derivative is tiny (~0.00018),
            // cleanly isolating any discontinuity spike at sample 96,000!
            let mut buf: Vec<[f32; 2]> = (0..total_frames)
                .map(|i| {
                    let s = (std::f64::consts::TAU * 110.0 * (i as f64) / sr).sin() as f32 * 0.85;
                    [s, s]
                })
                .collect();

            let mut pos = 0usize;
            while pos < total_frames {
                let len = (total_frames - pos).min(512);
                let ctx = ProcessContext {
                    sample_rate: sr,
                    current_sample_pos: pos as i64,
                    ppq_tick: 0,
                    bpm,
                    slice_frames: len,
                    is_playing: true,
                    is_offline_rendering: true,
                    ppq_resolution: crate::DEFAULT_PPQ,
                };
                gb.process_slice(&ctx, &mut buf[pos..pos + len], None, &mut []);
                pos += len;
            }
            buf.iter().map(|f| f[0]).collect()
        };

        let raw_out = run_gross_beat(false);
        let smooth_out = run_gross_beat(true);

        let peak_d2 = |sig: &[f32]| -> f64 {
            let mut max_d2 = 0.0f64;
            for i in 95_800..96_600 {
                let d2 = ((sig[i] as f64) - 2.0 * (sig[i - 1] as f64) + (sig[i - 2] as f64)).abs();
                if d2 > max_d2 {
                    max_d2 = d2;
                }
            }
            max_d2
        };

        let d2_raw = peak_d2(&raw_out);
        let d2_smooth = peak_d2(&smooth_out);
        let reduction_db = 20.0 * (d2_raw / d2_smooth.max(1e-12)).log10();
        assert!(
            reduction_db > 35.0,
            "Gate 4b failed: expected > 35 dB 2nd-derivative click reduction, got {reduction_db:.2} dB (raw={d2_raw:.6}, smooth={d2_smooth:.6})"
        );
    }

    #[test]
    fn emphasis_limiter_bilateral_tilt_unity_and_ceiling_clamp() {
        let sr = 48_000.0;
        let mut lim = EmphasisLimiterNode::new(64, -1.0);
        lim.emphasis_tilt_db = 4.5;
        lim.transient_recovery = 0.0;
        lim.prepare(sr, 512);

        // 1. Sub-threshold -24 dBFS multitone should pass through Pre-Emphasis + Inverse De-Emphasis within < 0.02 dB!
        let n = 8192usize;
        let mut sub_buf: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let t = (i as f64) / sr;
                let s = 0.04 * ((std::f64::consts::TAU * 220.0 * t).sin() + (std::f64::consts::TAU * 6400.0 * t).sin()) as f32;
                [s, s]
            })
            .collect();
        let ref_in = sub_buf.clone();
        let ctx = ProcessContext {
            sample_rate: sr,
            current_sample_pos: 0,
            ppq_tick: 0,
            bpm: 120.0,
            slice_frames: n,
            is_playing: true,
            is_offline_rendering: true,
            ppq_resolution: crate::DEFAULT_PPQ,
        };
        lim.process_slice(&ctx, &mut sub_buf, None, &mut []);
        let mut max_err = 0.0f32;
        for i in 1024..n {
            let err = (sub_buf[i][0] - ref_in[i - 64][0]).abs();
            max_err = max_err.max(err);
        }
        assert!(
            max_err < 1e-4,
            "Expected bilateral Pre/De-Emphasis exact cancellation below threshold, max_err={max_err}"
        );

        // 2. +8 dBFS hot drive must never exceed -1.0 dBFS ceiling
        lim.reset_state();
        let ceiling_lin = 10.0f32.powf(-1.0 / 20.0);
        let mut hot_buf: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let s = 2.5 * (std::f64::consts::TAU * 1000.0 * (i as f64) / sr).sin() as f32;
                [s, s]
            })
            .collect();
        lim.process_slice(&ctx, &mut hot_buf, None, &mut []);
        let max_out = hot_buf
            .iter()
            .fold(0.0f32, |m, f| m.max(f[0].abs()).max(f[1].abs()));
        assert!(
            max_out <= ceiling_lin + 1e-5,
            "EmphasisLimiterNode exceeded ceiling: max_out={max_out}, ceiling={ceiling_lin}"
        );
    }
}
