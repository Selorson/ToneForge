//! Real ONNX Runtime GPU/CPU Inference for iZotope Ozone 12 Unlimiter (`delimit_exp009_ep080_fixed2.onnx`,
//! `5,580,442` B, `DSP::OzoneUnlimiter` `0x1834DCB90` / `0x1834EB540` / `0x1864D6DD0`).
//!
//! Exact Reverse-Engineered Tensor & DSP Contract (`re_q1_ozone12.md` Section 1):
//! - Internal model sample rate: `48,000 Hz` (automatic stateful resampling when `sample_rate != 48_000.0`).
//! - STFT geometry: `fftSize = 2048` (`1025` frequency bins), `hopSize = 1024` (`50%` overlap).
//! - Window: Half-sample shifted Hann window
//!   $$w[n] = 0.5 - 0.5 \cos\!\left(\frac{2\pi (n + 0.5)}{2048}\right), \quad n \in [0, 2047]$$
//!   with exact COLA WOLA synthesis normalization $w_{\text{syn}}[n] = \frac{w[n] / 2048}{w[n]^2 + w[(n + 1024) \bmod 2048]^2}$.
//! - Unscaled forward RealFFT -> `input_spectrogram: float32[1, 1, 1025, 2, 2]` (`4,100` floats,
//!   threshold-normalized mono spectrum duplicated across `Ch0` and `Ch1`, `Im[0] = Im[1024] = 0`)
//!   + ping-ponged `input_states: float32[1, 2, 2, 1, 13, 128]` (`6,656` floats)
//!   -> `output_spectrogram: float32[1, 1, 1, 1025, 1, 2]` (`2,050` floats)
//!   + `output_states: float32[1, 2, 2, 1, 13, 128]` (`6,656` floats)
//!   -> inverse RealFFT + shifted-Hann WOLA -> residual envelope $r[n]$.
//! - Provides both:
//!   1. Synchronous `OzoneDelimitOnnxProcessor` (implementing `NeuralUnlimiterBackend` for offline
//!      mastering and deterministic sample-accurate verification), and
//!   2. Lock-free `RealtimeDelimitOnnxWorker` (implementing `NeuralUnlimiterBackend` via pre-allocated
//!      `rtrb::RingBuffer<f32>` SPSC queues to a dedicated background thread with **0 heap allocations
//!      and 0 locks** on the real-time audio thread).

use crate::onnx_gpu_runtime::{
    GpuExecutionTier, GpuRuntimeManager, resolve_ozone_delimit_onnx_path,
};
use daw_core_graph::StereoFrame;
use dsp_ozone12_mastering::NeuralUnlimiterBackend;
use ort::{session::Session, value::TensorRef};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rtrb::{Consumer, Producer, RingBuffer};
use rustfft::num_complex::Complex32;
use std::f64::consts::PI;
use std::path::Path;
use std::sync::{
    Arc, Barrier,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread::JoinHandle;

/// Internal sample rate expected by `delimit_exp009_ep080_fixed2.onnx` (`48,000 Hz`).
pub const DELIMIT_INTERNAL_SR_HZ: f64 = 48_000.0;
/// STFT FFT size (`2,048` samples -> `1,025` positive frequency bins).
pub const DELIMIT_FFT_SIZE: usize = 2048;
/// Positive real-FFT bin count (`2048 / 2 + 1 = 1025`).
pub const DELIMIT_FREQ_BINS: usize = DELIMIT_FFT_SIZE / 2 + 1;
/// STFT hop size (`1,024` samples = `21.333 ms` @ `48 kHz`, `50%` overlap).
pub const DELIMIT_HOP_SIZE: usize = 1024;
/// Exact causal streaming STFT + WOLA lookahead latency at `48,000 Hz` (`2,048` samples).
pub const DELIMIT_LATENCY_SAMPLES_48K: usize = 2048;

/// Exact element counts for `delimit_exp009_ep080_fixed2.onnx`.
pub const OZONE_DELIMIT_SPEC_IN_LEN: usize = 1 * 1 * DELIMIT_FREQ_BINS * 2 * 2; // 4,100
pub const OZONE_DELIMIT_SPEC_OUT_LEN: usize = 1 * 1 * 1 * DELIMIT_FREQ_BINS * 1 * 2; // 2,050
pub const OZONE_DELIMIT_STATES_LEN: usize = 1 * 2 * 2 * 1 * 13 * 128; // 6,656

const RESAMPLE_FIFO_CAP: usize = 4096;
const RT_RING_CAPACITY: usize = 65_536;
/// Fixed FIFO cushion between the real-time thread and the ONNX worker (`2048` samples): the RT
/// side starts consuming residuals only after this many samples, so the neural residual stays
/// sample-aligned with the dry signal at a constant, reported latency regardless of worker jitter.
pub const DELIMIT_RT_CUSHION_SAMPLES: usize = 2048;

/// Zero-allocation 4-point Catmull-Rom streaming resampler for `sample_rate != 48_000.0`.
#[derive(Clone, Debug)]
struct StreamingCatmullResampler {
    hist: [f32; 4],
    phase: f64,
}

impl StreamingCatmullResampler {
    #[inline]
    fn reset(&mut self) {
        self.hist = [0.0; 4];
        self.phase = 0.0;
    }

    #[inline]
    fn eval_catmull(y: [f32; 4], t: f32) -> f32 {
        let a = -0.5 * y[0] + 1.5 * y[1] - 1.5 * y[2] + 0.5 * y[3];
        let b = y[0] - 2.5 * y[1] + 2.0 * y[2] - 0.5 * y[3];
        let c = -0.5 * y[0] + 0.5 * y[2];
        let d = y[1];
        ((a * t + b) * t + c) * t + d
    }

    /// Pushes one input sample at `src_sr` and emits `0..N` interpolated samples at `dst_sr`
    /// into `emit_fn`. Zero heap allocations.
    #[inline]
    fn push_sample<F: FnMut(f32)>(&mut self, x: f32, src_sr: f64, dst_sr: f64, mut emit_fn: F) {
        self.hist[0] = self.hist[1];
        self.hist[1] = self.hist[2];
        self.hist[2] = self.hist[3];
        self.hist[3] = x;

        let step = (src_sr / dst_sr).clamp(0.05, 20.0);
        while self.phase < 1.0 {
            let y = Self::eval_catmull(self.hist, self.phase as f32);
            emit_fn(y);
            self.phase += step;
        }
        self.phase -= 1.0;
    }
}

struct DelimitStreamingWorkspace {
    analysis_win: [f32; DELIMIT_FFT_SIZE],
    synthesis_win: [f32; DELIMIT_FFT_SIZE],
    in_ring: [f32; DELIMIT_FFT_SIZE],
    in_stage: [f32; DELIMIT_HOP_SIZE],
    out_fifo: [f32; DELIMIT_HOP_SIZE],
    stage_pos: usize,
    ola_ring: [f32; DELIMIT_FFT_SIZE],
    time_buf: [f32; DELIMIT_FFT_SIZE],
    spec_in: [Complex32; DELIMIT_FREQ_BINS],
    spec_out: [Complex32; DELIMIT_FREQ_BINS],

    resamp_active_sr: f64,
    resamp_to_48k: StreamingCatmullResampler,
    resamp_from_48k: StreamingCatmullResampler,
    host_out_ring: [f32; RESAMPLE_FIFO_CAP],
    host_out_read: usize,
    host_out_write: usize,
    host_out_len: usize,
}

impl DelimitStreamingWorkspace {
    fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            let mut ws = Box::from_raw(ptr);

            // Half-sample shifted Hann window: w[n] = 0.5 - 0.5 * cos(2*pi*(n + 0.5) / 2048)
            let mut w_f64 = [0.0_f64; DELIMIT_FFT_SIZE];
            for n in 0..DELIMIT_FFT_SIZE {
                let w = 0.5 - 0.5 * ((2.0 * PI * ((n as f64) + 0.5)) / (DELIMIT_FFT_SIZE as f64)).cos();
                w_f64[n] = w;
                ws.analysis_win[n] = w as f32;
            }
            // Exact 2x COLA WOLA synthesis normalization (including 1/2048 for unscaled RealFFT inverse)
            let inv_fft = 1.0 / (DELIMIT_FFT_SIZE as f64);
            for n in 0..DELIMIT_FFT_SIZE {
                let n_pair = (n + DELIMIT_HOP_SIZE) % DELIMIT_FFT_SIZE;
                let denom = (w_f64[n] * w_f64[n] + w_f64[n_pair] * w_f64[n_pair]).max(1.0e-12);
                ws.synthesis_win[n] = ((w_f64[n] * inv_fft) / denom) as f32;
            }

            ws.resamp_active_sr = DELIMIT_INTERNAL_SR_HZ;
            ws.reset_resampler_cushion();
            ws
        }
    }

    fn reset_resampler_cushion(&mut self) {
        self.resamp_to_48k.reset();
        self.resamp_from_48k.reset();
        self.host_out_ring.fill(0.0);
        self.host_out_read = 0;
        // 2-sample cushion absorbs +-1 sample fractional step jitter without underflow
        self.host_out_write = 2;
        self.host_out_len = 2;
    }

    fn reset(&mut self) {
        self.in_ring.fill(0.0);
        self.in_stage.fill(0.0);
        self.out_fifo.fill(0.0);
        self.stage_pos = 0;
        self.ola_ring.fill(0.0);
        self.time_buf.fill(0.0);
        self.spec_in.fill(Complex32::new(0.0, 0.0));
        self.spec_out.fill(Complex32::new(0.0, 0.0));
        self.reset_resampler_cushion();
    }

    #[inline]
    fn push_host_out(&mut self, val: f32) {
        if self.host_out_len < RESAMPLE_FIFO_CAP {
            self.host_out_ring[self.host_out_write] = val;
            self.host_out_write = (self.host_out_write + 1) & (RESAMPLE_FIFO_CAP - 1);
            self.host_out_len += 1;
        }
    }

    #[inline]
    fn pop_host_out(&mut self) -> f32 {
        if self.host_out_len == 0 {
            return 0.0;
        }
        let v = self.host_out_ring[self.host_out_read];
        self.host_out_read = (self.host_out_read + 1) & (RESAMPLE_FIFO_CAP - 1);
        self.host_out_len -= 1;
        v
    }
}

/// Real ONNX Runtime GPU/CPU De-Limiter / Unlimiter executing `delimit_exp009_ep080_fixed2.onnx`.
///
/// Implements `dsp_ozone12_mastering::NeuralUnlimiterBackend` directly for deterministic
/// offline mastering and verification tests, and powers `RealtimeDelimitOnnxWorker` on a
/// dedicated background worker thread for lock-free zero-allocation real-time playback.
pub struct OzoneDelimitOnnxProcessor {
    session: Session,
    active_tier: GpuExecutionTier,
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
    fft_scratch_fwd: Vec<Complex32>,
    fft_scratch_inv: Vec<Complex32>,
    ws: Box<DelimitStreamingWorkspace>,

    input_spec_buf: Vec<f32>,
    states_stream: Vec<f32>,
    states_next: Vec<f32>,
    states_l: Vec<f32>,
    states_r: Vec<f32>,
    mask_out_buf: Vec<f32>,
}

// SAFETY: `ort::Session` and all owned buffers are thread-safe when accessed via `&mut self` or `&self`.
unsafe impl Send for OzoneDelimitOnnxProcessor {}
unsafe impl Sync for OzoneDelimitOnnxProcessor {}

impl OzoneDelimitOnnxProcessor {
    /// Loads `delimit_exp009_ep080_fixed2.onnx` from the default workspace/standalone asset path
    /// via `GpuRuntimeManager` (`CUDA -> DirectML -> CPU`).
    pub fn from_default_assets() -> Result<Self, String> {
        let model_path = resolve_ozone_delimit_onnx_path().ok_or_else(|| {
            "Could not locate delimit_exp009_ep080_fixed2.onnx in assets/ozone12/models".to_string()
        })?;
        Self::from_model_path(&model_path)
    }

    /// Loads `delimit_exp009_ep080_fixed2.onnx` from an explicit path via `GpuRuntimeManager`.
    pub fn from_model_path(model_path: &Path) -> Result<Self, String> {
        let rt = GpuRuntimeManager::global();
        let (session, active_tier) = rt.create_session(model_path)?;

        let mut planner = RealFftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(DELIMIT_FFT_SIZE);
        let fft_inverse = planner.plan_fft_inverse(DELIMIT_FFT_SIZE);
        let fft_scratch_fwd = fft_forward.make_scratch_vec();
        let fft_scratch_inv = fft_inverse.make_scratch_vec();

        Ok(Self {
            session,
            active_tier,
            fft_forward,
            fft_inverse,
            fft_scratch_fwd,
            fft_scratch_inv,
            ws: DelimitStreamingWorkspace::new_boxed(),
            input_spec_buf: vec![0.0f32; OZONE_DELIMIT_SPEC_IN_LEN],
            states_stream: vec![0.0f32; OZONE_DELIMIT_STATES_LEN],
            states_next: vec![0.0f32; OZONE_DELIMIT_STATES_LEN],
            states_l: vec![0.0f32; OZONE_DELIMIT_STATES_LEN],
            states_r: vec![0.0f32; OZONE_DELIMIT_STATES_LEN],
            mask_out_buf: vec![0.0f32; OZONE_DELIMIT_SPEC_OUT_LEN],
        })
    }

    #[inline]
    pub fn active_tier(&self) -> GpuExecutionTier {
        self.active_tier
    }

    /// Clears all recurrent LSTM states (`[1, 2, 2, 1, 13, 128]`) and STFT WOLA buffers.
    pub fn reset(&mut self) {
        self.states_stream.fill(0.0);
        self.states_next.fill(0.0);
        self.states_l.fill(0.0);
        self.states_r.fill(0.0);
        self.ws.reset();
    }

    /// Exact `0x1834EB540` single-hop inference on a mono `1025`-bin unscaled spectrum `mono_spec`:
    /// - Packs `input_spectrogram: float32[1, 1, 1025, 2, 2]` with `mono_spec` duplicated across
    ///   `Ch0` and `Ch1` (`Im[0] = Im[1024] = 0.0`).
    /// - Ping-pongs `states_stream <-> states_next` (`float32[1, 2, 2, 1, 13, 128]`).
    /// - Unpacks `output_spectrogram: float32[1, 1, 1, 1025, 1, 2]` into `out_spec`.
    pub fn infer_mono_delimit_frame(
        &mut self,
        mono_spec: &[Complex32; DELIMIT_FREQ_BINS],
        out_spec: &mut [Complex32; DELIMIT_FREQ_BINS],
    ) -> Result<(), String> {
        for k in 0..DELIMIT_FREQ_BINS {
            let re = mono_spec[k].re;
            let im = if k == 0 || k == DELIMIT_FREQ_BINS - 1 {
                0.0
            } else {
                mono_spec[k].im
            };
            let base = k * 4;
            // Mono duplicated across Ch0 and Ch1: [1, 1, 1025, 2, 2]
            self.input_spec_buf[base] = re;
            self.input_spec_buf[base + 1] = im;
            self.input_spec_buf[base + 2] = re;
            self.input_spec_buf[base + 3] = im;
        }

        {
            let t_spec = TensorRef::from_array_view((
                [1usize, 1, DELIMIT_FREQ_BINS, 2, 2],
                self.input_spec_buf.as_slice(),
            ))
            .map_err(|e| e.to_string())?;
            let t_states = TensorRef::from_array_view((
                [1usize, 2, 2, 1, 13, 128],
                self.states_stream.as_slice(),
            ))
            .map_err(|e| e.to_string())?;

            let outputs = self
                .session
                .run(ort::inputs![
                    "input_spectrogram" => t_spec,
                    "input_states" => t_states,
                ])
                .map_err(|e| format!("Delimit ONNX run failed: {e}"))?;

            let (_, spec_slice) = outputs["output_spectrogram"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            let (_, next_states) = outputs["output_states"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;

            self.mask_out_buf.copy_from_slice(spec_slice);
            self.states_next.copy_from_slice(next_states);
        }
        // Ping-pong recurrent states without reallocating
        std::mem::swap(&mut self.states_stream, &mut self.states_next);

        for k in 0..DELIMIT_FREQ_BINS {
            let re = self.mask_out_buf[k * 2];
            let im = if k == 0 || k == DELIMIT_FREQ_BINS - 1 {
                0.0
            } else {
                self.mask_out_buf[k * 2 + 1]
            };
            out_spec[k] = Complex32::new(re, im);
        }

        Ok(())
    }

    /// Executes one `1,024`-sample STFT + ONNX + shifted-Hann WOLA hop at `48,000 Hz`.
    fn run_streaming_hop_48k(&mut self) -> Result<(), String> {
        // 1. Shift `in_ring` left by `1024` and append `in_stage[0..1024]`
        self.ws
            .in_ring
            .copy_within(DELIMIT_HOP_SIZE..DELIMIT_FFT_SIZE, 0);
        self.ws.in_ring[DELIMIT_HOP_SIZE..DELIMIT_FFT_SIZE].copy_from_slice(&self.ws.in_stage);

        // 2. Apply half-sample shifted Hann analysis window & unscaled forward RealFFT
        for n in 0..DELIMIT_FFT_SIZE {
            self.ws.time_buf[n] = self.ws.in_ring[n] * self.ws.analysis_win[n];
        }
        self.fft_forward
            .process_with_scratch(
                &mut self.ws.time_buf,
                &mut self.ws.spec_in,
                &mut self.fft_scratch_fwd,
            )
            .map_err(|e| e.to_string())?;
        self.ws.spec_in[0].im = 0.0;
        self.ws.spec_in[DELIMIT_FREQ_BINS - 1].im = 0.0;

        // 3. Run ONNX Delimit inference (`[1,1,1025,2,2]` -> `[1,1,1,1025,1,2]`)
        let spec_in_copy = self.ws.spec_in;
        let mut spec_out_local = [Complex32::new(0.0, 0.0); DELIMIT_FREQ_BINS];
        self.infer_mono_delimit_frame(&spec_in_copy, &mut spec_out_local)?;
        self.ws.spec_out = spec_out_local;

        // 4. Inverse RealFFT + shifted-Hann COLA WOLA synthesis
        self.fft_inverse
            .process_with_scratch(
                &mut self.ws.spec_out,
                &mut self.ws.time_buf,
                &mut self.fft_scratch_inv,
            )
            .map_err(|e| e.to_string())?;

        for n in 0..DELIMIT_FFT_SIZE {
            self.ws.ola_ring[n] += self.ws.time_buf[n] * self.ws.synthesis_win[n];
        }

        // 5. Emit completed `1024`-sample hop into `out_fifo` and shift `ola_ring`
        self.ws
            .out_fifo
            .copy_from_slice(&self.ws.ola_ring[..DELIMIT_HOP_SIZE]);
        self.ws
            .ola_ring
            .copy_within(DELIMIT_HOP_SIZE..DELIMIT_FFT_SIZE, 0);
        self.ws.ola_ring[DELIMIT_HOP_SIZE..DELIMIT_FFT_SIZE].fill(0.0);

        Ok(())
    }

    #[inline]
    fn step_one_48k_sample(&mut self, m_norm_48k: f32) -> Result<f32, String> {
        let pos = self.ws.stage_pos;
        let y_48k = self.ws.out_fifo[pos];
        self.ws.in_stage[pos] = m_norm_48k;
        let next_pos = pos + 1;
        if next_pos == DELIMIT_HOP_SIZE {
            self.ws.stage_pos = 0;
            self.run_streaming_hop_48k()?;
        } else {
            self.ws.stage_pos = next_pos;
        }
        Ok(y_48k)
    }

    /// Backwards-compatible low-level dual-channel `[1, 1, 1025, 2, 2]` inference helper.
    pub fn infer_channel_complex_mask(
        &mut self,
        frames_2: &[[Complex32; DELIMIT_FREQ_BINS]; 2],
        is_right_channel: bool,
        out_mask: &mut [Complex32; DELIMIT_FREQ_BINS],
    ) -> Result<(), String> {
        for k in 0..DELIMIT_FREQ_BINS {
            let base = k * 4;
            let im0 = if k == 0 || k == DELIMIT_FREQ_BINS - 1 {
                0.0
            } else {
                frames_2[0][k].im
            };
            let im1 = if k == 0 || k == DELIMIT_FREQ_BINS - 1 {
                0.0
            } else {
                frames_2[1][k].im
            };
            self.input_spec_buf[base] = frames_2[0][k].re;
            self.input_spec_buf[base + 1] = im0;
            self.input_spec_buf[base + 2] = frames_2[1][k].re;
            self.input_spec_buf[base + 3] = im1;
        }

        let states_ref = if is_right_channel {
            &mut self.states_r
        } else {
            &mut self.states_l
        };

        {
            let t_spec = TensorRef::from_array_view((
                [1usize, 1, DELIMIT_FREQ_BINS, 2, 2],
                self.input_spec_buf.as_slice(),
            ))
            .map_err(|e| e.to_string())?;
            let t_states = TensorRef::from_array_view((
                [1usize, 2, 2, 1, 13, 128],
                states_ref.as_slice(),
            ))
            .map_err(|e| e.to_string())?;

            let outputs = self
                .session
                .run(ort::inputs![
                    "input_spectrogram" => t_spec,
                    "input_states" => t_states,
                ])
                .map_err(|e| format!("Delimit ONNX run failed: {e}"))?;

            let (_, mask_slice) = outputs["output_spectrogram"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            let (_, next_states) = outputs["output_states"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;

            self.mask_out_buf.copy_from_slice(mask_slice);
            states_ref.copy_from_slice(next_states);
        }

        for k in 0..DELIMIT_FREQ_BINS {
            out_mask[k] = Complex32::new(self.mask_out_buf[k * 2], self.mask_out_buf[k * 2 + 1]);
        }
        Ok(())
    }

    /// Restores clipped/brickwall-limited transients on an offline stereo signal (`48,000 Hz` default)
    /// using the exact `0x1834DCB90` + `0x1834EB540` shifted-Hann WOLA restoration contract
    /// with automatic lookahead delay compensation (`0` output offset).
    pub fn process_offline(
        &mut self,
        clipped: &[StereoFrame],
        amount: f32,
    ) -> Result<Vec<StereoFrame>, String> {
        self.process_offline_with_threshold(clipped, 48_000.0, 0.0, amount)
    }

    /// Offline stereo transient restoration at arbitrary `sample_rate` and `threshold_db` (`[-24.0, 0.0]`),
    /// compensating for the exact `latency_samples(sample_rate)` lookahead so output aligns at sample `0`.
    pub fn process_offline_with_threshold(
        &mut self,
        clipped: &[StereoFrame],
        sample_rate: f64,
        threshold_db: f32,
        amount: f32,
    ) -> Result<Vec<StereoFrame>, String> {
        let len = clipped.len();
        if len == 0 {
            return Ok(Vec::new());
        }
        let alpha = amount.clamp(0.0, 1.0);
        if alpha <= 1.0e-6 {
            return Ok(clipped.to_vec());
        }

        self.reset();
        let sr = sample_rate.max(8_000.0);
        let delay = self.latency_samples(sr);

        // Estimate ceiling/threshold if `threshold_db == 0.0` and signal is brickwalled below 0 dBFS
        let peak_abs = clipped
            .iter()
            .fold(0.0_f32, |acc, f| acc.max(f[0].abs()).max(f[1].abs()));
        let thresh_lin = if threshold_db < -0.01 {
            10.0_f32.powf(threshold_db.clamp(-24.0, 0.0) / 20.0)
        } else {
            peak_abs.clamp(0.1, 1.0)
        }
        .max(1.0e-6);
        let inv_thresh = 1.0 / thresh_lin;

        let total = len + delay;
        let mut mono_in = vec![0.0f32; total];
        for i in 0..len {
            mono_in[i] = 0.5 * (clipped[i][0] + clipped[i][1]) * inv_thresh;
        }
        let mut residual_all = vec![0.0f32; total];
        if !self.process_delimit_envelope(&mono_in, sr, &mut residual_all) {
            return Err("OzoneDelimitOnnxProcessor failed during offline envelope inference".to_string());
        }

        let mut out = vec![[0.0f32; 2]; len];
        for i in 0..len {
            let r_pos = residual_all[delay + i].max(0.0);
            let g = 1.0 + alpha * r_pos;
            out[i] = [clipped[i][0] * g, clipped[i][1] * g];
        }
        Ok(out)
    }
}

impl NeuralUnlimiterBackend for OzoneDelimitOnnxProcessor {
    fn process_delimit_envelope(
        &mut self,
        mono_norm: &[f32],
        sample_rate: f64,
        out_residual: &mut [f32],
    ) -> bool {
        let n = mono_norm.len().min(out_residual.len());
        if n == 0 {
            return true;
        }
        let sr = sample_rate.max(8_000.0);

        if (sr - DELIMIT_INTERNAL_SR_HZ).abs() < 1.0 {
            for i in 0..n {
                match self.step_one_48k_sample(mono_norm[i]) {
                    Ok(y) => out_residual[i] = y,
                    Err(_) => return false,
                }
            }
            return true;
        }

        // Resampled streaming path (`sr -> 48_000 Hz -> STFT/ONNX/WOLA -> sr`)
        if (self.ws.resamp_active_sr - sr).abs() >= 1.0 {
            self.ws.resamp_active_sr = sr;
            self.ws.reset_resampler_cushion();
        }

        let mut mid_48k_buf = [0.0f32; 64];
        for i in 0..n {
            let mut mid_count = 0usize;
            self.ws
                .resamp_to_48k
                .push_sample(mono_norm[i], sr, DELIMIT_INTERNAL_SR_HZ, |s48| {
                    if mid_count < mid_48k_buf.len() {
                        mid_48k_buf[mid_count] = s48;
                        mid_count += 1;
                    }
                });

            for m_idx in 0..mid_count {
                let y48 = match self.step_one_48k_sample(mid_48k_buf[m_idx]) {
                    Ok(v) => v,
                    Err(_) => return false,
                };
                let mut host_emitted = [0.0f32; 64];
                let mut h_cnt = 0usize;
                self.ws.resamp_from_48k.push_sample(
                    y48,
                    DELIMIT_INTERNAL_SR_HZ,
                    sr,
                    |y_host| {
                        if h_cnt < host_emitted.len() {
                            host_emitted[h_cnt] = y_host;
                            h_cnt += 1;
                        }
                    },
                );
                for h_idx in 0..h_cnt {
                    self.ws.push_host_out(host_emitted[h_idx]);
                }
            }

            out_residual[i] = self.ws.pop_host_out();
        }
        true
    }

    #[inline]
    fn latency_samples(&self, sample_rate: f64) -> usize {
        let sr = sample_rate.max(8_000.0);
        if (sr - DELIMIT_INTERNAL_SR_HZ).abs() < 1.0 {
            DELIMIT_LATENCY_SAMPLES_48K
        } else {
            ((DELIMIT_LATENCY_SAMPLES_48K as f64) * sr / DELIMIT_INTERNAL_SR_HZ).round() as usize
        }
    }

    fn reset(&mut self) {
        OzoneDelimitOnnxProcessor::reset(self);
    }
}

/// Lock-free Real-Time ONNX Unlimiter Worker (`RealtimeDelimitOnnxWorker`).
///
/// Communicates between the real-time audio thread (`UnlimiterModule` / `Ozone12MasterChain`)
/// and a dedicated background ONNX Runtime inference thread via pre-allocated `rtrb::RingBuffer<f32>`
/// SPSC queues:
/// - **0 heap allocations** and **0 mutex locks** on the real-time caller thread.
/// - Reports exact `2048`-sample `@ 48 kHz` scaled latency (`(2048 * sample_rate / 48000).round()`).
pub struct RealtimeDelimitOnnxWorker {
    in_tx: Producer<f32>,
    out_rx: Consumer<f32>,
    /// Real input samples pushed (reset markers excluded) and residuals consumed (incl. skips).
    pushed: u64,
    popped: u64,
    /// Residuals still to be skipped when they arrive (late after an underflow, or stale after reset).
    deficit: u64,
    cushion_remaining: usize,
    /// Samples for which no residual was ready in time (reported, never silently misaligned).
    pub underflow_samples: u64,
    sample_rate_bits: Arc<AtomicU64>,
    stop_flag: Arc<AtomicBool>,
    active_tier: GpuExecutionTier,
    join_handle: Option<JoinHandle<()>>,
}

// SAFETY: `rtrb::Producer<f32>` and `rtrb::Consumer<f32>` are `Send`, and `&mut self` serializes access.
unsafe impl Send for RealtimeDelimitOnnxWorker {}
unsafe impl Sync for RealtimeDelimitOnnxWorker {}

impl RealtimeDelimitOnnxWorker {
    /// Spawns the lock-free real-time worker loading `delimit_exp009_ep080_fixed2.onnx` from default assets.
    pub fn from_default_assets(initial_sample_rate: f64) -> Result<Self, String> {
        let processor = OzoneDelimitOnnxProcessor::from_default_assets()?;
        Self::from_processor(processor, initial_sample_rate)
    }

    /// Spawns the lock-free real-time worker loading `delimit_exp009_ep080_fixed2.onnx` from `model_path`.
    pub fn from_model_path(model_path: &Path, initial_sample_rate: f64) -> Result<Self, String> {
        let processor = OzoneDelimitOnnxProcessor::from_model_path(model_path)?;
        Self::from_processor(processor, initial_sample_rate)
    }

    /// Spawns the lock-free real-time worker taking ownership of an initialized `OzoneDelimitOnnxProcessor`.
    pub fn from_processor(
        mut processor: OzoneDelimitOnnxProcessor,
        initial_sample_rate: f64,
    ) -> Result<Self, String> {
        let sr = initial_sample_rate.max(8_000.0);
        let active_tier = processor.active_tier();

        let (in_tx, mut in_rx) = RingBuffer::<f32>::new(RT_RING_CAPACITY);
        let (mut out_tx, out_rx) = RingBuffer::<f32>::new(RT_RING_CAPACITY);

        let sample_rate_bits = Arc::new(AtomicU64::new(sr.to_bits()));
        let reset_epoch = Arc::new(AtomicU64::new(1));
        let stop_flag = Arc::new(AtomicBool::new(false));
        let ready_barrier = Arc::new(Barrier::new(2));

        let sr_clone = Arc::clone(&sample_rate_bits);
        let epoch_clone = Arc::clone(&reset_epoch);
        let stop_clone = Arc::clone(&stop_flag);
        let ready_clone = Arc::clone(&ready_barrier);

        let join_handle = std::thread::Builder::new()
            .name("toneforge-delimit-onnx".to_string())
            .stack_size(4 * 1024 * 1024)
            .spawn(move || {
                // Pre-warm CUDA / DirectML / CPU kernels on this worker thread outside RT audio!
                let warm_in = [0.0f32; DELIMIT_HOP_SIZE];
                let mut warm_out = [0.0f32; DELIMIT_HOP_SIZE];
                let _ = processor.process_delimit_envelope(&warm_in, sr, &mut warm_out);
                processor.reset();
                ready_clone.wait();

                let mut local_epoch = 1u64;
                let mut in_chunk = [0.0f32; 512];
                let mut out_chunk = [0.0f32; 512];
                let mut idle_spins = 0u32;

                while !stop_clone.load(Ordering::Acquire) {
                    let target_epoch = epoch_clone.load(Ordering::Acquire);
                    if target_epoch != local_epoch {
                        while in_rx.pop().is_ok() {}
                        processor.reset();
                        local_epoch = target_epoch;
                    }

                    let avail = in_rx.slots().min(in_chunk.len());
                    if avail > 0 && out_tx.slots() >= avail {
                        idle_spins = 0;
                        let mut popped = 0usize;
                        while popped < avail {
                            if let Ok(v) = in_rx.pop() {
                                if v.is_nan() {
                                    // In-band reset marker: process what precedes it, then reset.
                                    if popped > 0 {
                                        let cur_sr = f64::from_bits(sr_clone.load(Ordering::Relaxed))
                                            .max(8_000.0);
                                        let ok = processor.process_delimit_envelope(
                                            &in_chunk[..popped],
                                            cur_sr,
                                            &mut out_chunk[..popped],
                                        );
                                        for &o in &out_chunk[..popped] {
                                            let _ = out_tx.push(if ok { o } else { 0.0 });
                                        }
                                        popped = 0;
                                    }
                                    processor.reset();
                                    continue;
                                }
                                in_chunk[popped] = v;
                                popped += 1;
                            } else {
                                break;
                            }
                        }
                        if popped > 0 {
                            let cur_sr = f64::from_bits(sr_clone.load(Ordering::Relaxed)).max(8_000.0);
                            let ok = processor.process_delimit_envelope(
                                &in_chunk[..popped],
                                cur_sr,
                                &mut out_chunk[..popped],
                            );
                            for &v in &out_chunk[..popped] {
                                let _ = out_tx.push(if ok { v } else { 0.0 });
                            }
                        }
                    } else {
                        idle_spins = idle_spins.saturating_add(1);
                        if idle_spins < 256 {
                            std::hint::spin_loop();
                        } else {
                            std::thread::sleep(std::time::Duration::from_micros(100));
                        }
                    }
                }
            })
            .map_err(|e| format!("Failed to spawn RealtimeDelimitOnnxWorker thread: {e}"))?;

        ready_barrier.wait();

        Ok(Self {
            in_tx,
            out_rx,
            pushed: 0,
            popped: 0,
            deficit: 0,
            cushion_remaining: DELIMIT_RT_CUSHION_SAMPLES,
            underflow_samples: 0,
            sample_rate_bits,
            
            stop_flag,
            active_tier,
            join_handle: Some(join_handle),
        })
    }

    #[inline]
    pub fn active_tier(&self) -> GpuExecutionTier {
        self.active_tier
    }

    /// Non-RT test synchronization helper: waits until the background worker has drained all
    /// currently queued input samples (up to `timeout`). Never called on the RT audio thread.
    pub fn wait_for_worker_drain(&self, timeout: std::time::Duration) {
        let start = std::time::Instant::now();
        while self.in_tx.slots() < RT_RING_CAPACITY && start.elapsed() < timeout {
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        // Allow one brief final slice for the worker to push `out_chunk`
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

impl NeuralUnlimiterBackend for RealtimeDelimitOnnxWorker {
    #[inline]
    fn process_delimit_envelope(
        &mut self,
        mono_norm: &[f32],
        sample_rate: f64,
        out_residual: &mut [f32],
    ) -> bool {
        let n = mono_norm.len().min(out_residual.len());
        if n == 0 {
            return true;
        }
        let sr_bits = sample_rate.max(8_000.0).to_bits();
        if self.sample_rate_bits.load(Ordering::Relaxed) != sr_bits {
            self.sample_rate_bits.store(sr_bits, Ordering::Relaxed);
        }

        for i in 0..n {
            if self.in_tx.push(mono_norm[i]).is_ok() {
                self.pushed += 1;
            } else {
                // Input ring full (worker stalled): this sample's residual will never exist.
                self.underflow_samples += 1;
                self.pushed += 1;
                self.popped += 1;
            }
        }
        for i in 0..n {
            if self.cushion_remaining > 0 {
                self.cushion_remaining -= 1;
                out_residual[i] = 0.0;
                continue;
            }
            out_residual[i] = loop {
                match self.out_rx.pop() {
                    Ok(v) => {
                        self.popped += 1;
                        if self.deficit > 0 {
                            self.deficit -= 1;
                            continue;
                        }
                        break v;
                    }
                    Err(_) => {
                        // Not ready in time: emit no residual for this sample and skip its late
                        // result so every later residual stays aligned.
                        self.underflow_samples += 1;
                        self.deficit += 1;
                        break 0.0;
                    }
                }
            };
        }
        true
    }

    #[inline]
    fn latency_samples(&self, sample_rate: f64) -> usize {
        let sr = sample_rate.max(8_000.0);
        let model = if (sr - DELIMIT_INTERNAL_SR_HZ).abs() < 1.0 {
            DELIMIT_LATENCY_SAMPLES_48K
        } else {
            ((DELIMIT_LATENCY_SAMPLES_48K as f64) * sr / DELIMIT_INTERNAL_SR_HZ).round() as usize
        };
        model + DELIMIT_RT_CUSHION_SAMPLES
    }

    fn reset(&mut self) {
        // Ordered in-band reset: residuals of everything pushed so far are stale.
        let _ = self.in_tx.push(f32::NAN);
        self.deficit = self.pushed - self.popped;
        self.cushion_remaining = DELIMIT_RT_CUSHION_SAMPLES;
    }
}

impl Drop for RealtimeDelimitOnnxWorker {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}
