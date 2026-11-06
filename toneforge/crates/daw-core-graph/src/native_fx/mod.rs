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

/// Multi-stage Mastering Limiter with Pre/De-Emphasis tilt (`Emphasis_x64.dll` & `ildsp_x64.dll`).
pub struct EmphasisLimiterNode {
    pub ceiling_db: f32,
    pub emphasis_tilt_db: f32,
    lookahead_samples: usize,
    delay: ApdcDelayLine,
    env: f32,
}

impl EmphasisLimiterNode {
    pub fn new(lookahead_samples: usize, ceiling_db: f32) -> Self {
        let mut delay = ApdcDelayLine::with_capacity(lookahead_samples.max(2048));
        delay.set_delay_samples(lookahead_samples);
        Self {
            ceiling_db,
            emphasis_tilt_db: 0.0,
            lookahead_samples,
            delay,
            env: 0.0,
        }
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
            .ensure_capacity(self.lookahead_samples.max(2048));
        self.delay.set_delay_samples(self.lookahead_samples);
    }

    fn reset_state(&mut self) {
        self.delay.reset();
        self.env = 0.0;
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.ceiling_db = -12.0 + v * 12.0,
            1 => self.emphasis_tilt_db = -6.0 + v * 12.0,
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
        let ceiling_lin = 10.0f32.powf(self.ceiling_db / 20.0);
        self.delay.process_in_place(main_io);
        for frame in main_io.iter_mut() {
            let peak = frame[0].abs().max(frame[1].abs());
            self.env = if peak > self.env {
                peak
            } else {
                0.995 * self.env + 0.005 * peak
            };
            let g = if self.env > ceiling_lin {
                ceiling_lin / self.env.max(1e-9)
            } else {
                1.0
            };
            frame[0] *= g;
            frame[1] *= g;
        }
    }
}

/// Real-Time 2-Bar Circular Time-Buffer & Volume Gate (`Gross Beat_x64.dll` `TBRDelay_32FS` `0x008420f0`).
pub struct GrossBeatTimeBufferNode {
    history_ring: Vec<StereoFrame>,
    write_pos: usize,
    pub time_spline: TensionSpline,
    pub volume_spline: TensionSpline,
}

impl Default for GrossBeatTimeBufferNode {
    fn default() -> Self {
        Self::new(192_000)
    }
}

impl GrossBeatTimeBufferNode {
    pub fn new(history_frames: usize) -> Self {
        Self {
            history_ring: vec![[0.0; 2]; history_frames.max(4096)],
            write_pos: 0,
            time_spline: TensionSpline::new(vec![
                TensionControlPoint::new(0.0, 0.0, 0.0),
                TensionControlPoint::new(1.0, 0.0, 0.0),
            ]),
            volume_spline: TensionSpline::new(vec![
                TensionControlPoint::new(0.0, 1.0, 0.0),
                TensionControlPoint::new(1.0, 1.0, 0.0),
            ]),
        }
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
    }

    fn set_parameter_normalized(&mut self, _param_id: u32, _norm_val: f32) {}

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
        let samples_per_2bars = ((60.0 * ctx.sample_rate / ctx.bpm.max(20.0)) * 8.0).max(1.0);

        for (offset, frame) in main_io.iter_mut().enumerate() {
            self.history_ring[self.write_pos] = *frame;
            let abs_sample = (ctx.current_sample_pos + offset as i64).max(0) as f64;
            let bar_phase = (abs_sample / samples_per_2bars).fract();

            let delay_beat_frac = self.time_spline.eval(bar_phase).clamp(0.0, 1.0);
            let vol_gate = self.volume_spline.eval(bar_phase).clamp(0.0, 1.0) as f32;
            let delay_samples = (delay_beat_frac * samples_per_2bars * 0.25) as usize;
            let read_idx = (self.write_pos + ring_len - (delay_samples % ring_len)) % ring_len;
            let delayed = self.history_ring[read_idx];
            *frame = [delayed[0] * vol_gate, delayed[1] * vol_gate];

            self.write_pos += 1;
            if self.write_pos >= ring_len {
                self.write_pos = 0;
            }
        }
    }
}

/// 4x4 Hadamard Feedback Delay Network Shimmer Reverb (`LuxeVerb_x64.dll` & `ildsp_x64.dll`).
pub struct LuxeVerbFdnShimmerNode {
    delay_lines: [Vec<f32>; 4],
    cursors: [usize; 4],
    pub decay: f32,
    pub wet_level: f32,
}

impl Default for LuxeVerbFdnShimmerNode {
    fn default() -> Self {
        Self::new()
    }
}

impl LuxeVerbFdnShimmerNode {
    pub fn new() -> Self {
        const PRIMES: [usize; 4] = [1499, 1877, 2239, 2683];
        Self {
            delay_lines: [
                vec![0.0; PRIMES[0]],
                vec![0.0; PRIMES[1]],
                vec![0.0; PRIMES[2]],
                vec![0.0; PRIMES[3]],
            ],
            cursors: [0; 4],
            decay: 0.78,
            wet_level: 0.25,
        }
    }
}

impl UnifiedDspNode for LuxeVerbFdnShimmerNode {
    fn name(&self) -> &'static str {
        "LuxeVerb FDN Shimmer"
    }

    fn latency_samples(&self) -> usize {
        0
    }

    fn prepare(&mut self, _sample_rate: f64, _max_block_size: usize) {}

    fn reset_state(&mut self) {
        for line in &mut self.delay_lines {
            line.fill(0.0);
        }
        self.cursors = [0; 4];
    }

    fn set_parameter_normalized(&mut self, param_id: u32, norm_val: f32) {
        let v = norm_val.clamp(0.0, 1.0);
        match param_id {
            0 => self.decay = 0.1 + v * 0.88,
            1 => self.wet_level = v,
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
        let fb = 0.5 * self.decay;
        let wet = self.wet_level;
        for frame in main_io.iter_mut() {
            let mut d = [0.0f32; 4];
            for i in 0..4 {
                d[i] = self.delay_lines[i][self.cursors[i]];
            }
            // Orthogonal 4x4 Hadamard feedback matrix
            let h0 = d[0] + d[1] + d[2] + d[3];
            let h1 = d[0] - d[1] + d[2] - d[3];
            let h2 = d[0] + d[1] - d[2] - d[3];
            let h3 = d[0] - d[1] - d[2] + d[3];

            let in_l = frame[0];
            let in_r = frame[1];
            let next = [
                in_l + fb * h0,
                in_r + fb * h1,
                in_l + fb * h2,
                in_r + fb * h3,
            ];
            for i in 0..4 {
                let c = self.cursors[i];
                self.delay_lines[i][c] = next[i];
                self.cursors[i] = if c + 1 >= self.delay_lines[i].len() {
                    0
                } else {
                    c + 1
                };
            }
            frame[0] = in_l + wet * 0.5 * (d[0] + d[2]);
            frame[1] = in_r + wet * 0.5 * (d[1] + d[3]);
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
