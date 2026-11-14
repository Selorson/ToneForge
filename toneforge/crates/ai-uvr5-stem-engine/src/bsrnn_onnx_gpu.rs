//! Real ONNX Runtime GPU Inference Engines for iZotope Ozone 12 Neural Models (`bsrnn_onnx_gpu.rs`):
//!
//! 1. **`OzoneBsrnnOnnxSeparator`** (`ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx`, `7,230,143` B):
//!    - Exact ONNX Tensor Contract:
//!      * `input_spectrogram`: `f32[2, 1, 1025, 3, 2]` (`12,300` floats — `[L/R, 1, bins, 3 frames, Re/Im]`)
//!      * `input_states`: `f32[2, 2, 2, 1, 13, 128]` (`13,312` floats — recurrent LSTM `[h, c]` across 13 subbands)
//!      * `input_states_normalize`: `f32[2, 1]` (`2` floats — per-channel online EMA power normalizer)
//!      * `output_spectrogram`: `f32[2, 3, 1, 1025, 1, 2]` (`12,300` floats — complex-masked `Frame(0)` spectra
//!        ordered by iZotope `SourceType` `[0=Vocals, 1=Bass, 2=Drums]`, mapped to ToneForge `StemKind`
//!        `[0=Vocals, 1=Drums, 2=Bass, 3=Other]` with `Other = X_frame0 - (Vocals + Drums + Bass)`).
//!
//! 2. **`OzoneDelimitOnnxProcessor`** (`delimit_exp009_ep080_fixed2.onnx`, `5,580,442` B):
//!    - Exact ONNX Tensor Contract:
//!      * `input_spectrogram`: `f32[1, 1, 1025, 2, 2]` (`4,100` floats — 2-frame sliding STFT `[Frame(t), Frame(t+1)]`)
//!      * `input_states`: `f32[1, 2, 2, 1, 13, 128]` (`6,656` floats — 13-subband LSTM state)
//!      * `output_spectrogram`: `f32[1, 1, 1, 1025, 1, 2]` (`2,050` floats — complex GLU transient restoration mask)
//!      * `output_states`: `f32[1, 2, 2, 1, 13, 128]` (`6,656` floats).

use crate::bsrnn_streaming::{
    BSRNN_FREQ_BINS, BSRNN_HOP_SIZE, BSRNN_N_FFT, BSRNN_STEM_COUNT, BsrnnInputHop, BsrnnOutputHop,
};
pub use crate::delimit_onnx::{
    OZONE_DELIMIT_SPEC_IN_LEN, OZONE_DELIMIT_SPEC_OUT_LEN, OZONE_DELIMIT_STATES_LEN,
    OzoneDelimitOnnxProcessor, RealtimeDelimitOnnxWorker,
};
use crate::onnx_gpu_runtime::{
    GpuExecutionTier, GpuRuntimeManager, resolve_ozone_bsrnn_onnx_path,
};
use daw_core_graph::StereoFrame;
use ort::{session::Session, value::TensorRef};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex32;
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;

/// Exact element count of `input_spectrogram` / `output_spectrogram` (`2 * 1 * 1025 * 3 * 2 = 12,300`).
pub const OZONE_BSRNN_SPEC_LEN: usize = 2 * 1 * BSRNN_FREQ_BINS * 3 * 2;

/// Exact element count of `input_states` / `output_states` (`2 * 2 * 2 * 1 * 13 * 128 = 13,312`).
pub const OZONE_BSRNN_STATES_LEN: usize = 2 * 2 * 2 * 1 * 13 * 128;

/// Exact 2-hop future lookahead in samples (`2 * 512 = 1,024` samples) inside the 3-frame STFT window.
pub const OZONE_BSRNN_LOOKAHEAD_HOPS: usize = 2;
pub const OZONE_BSRNN_LOOKAHEAD_SAMPLES: usize = OZONE_BSRNN_LOOKAHEAD_HOPS * BSRNN_HOP_SIZE;

/// Real ONNX Runtime GPU/CPU 4-Stem Separator executing `ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx`.
struct BsrnnHeapBuffers {
    analysis_window: [f32; BSRNN_N_FFT],
    synthesis_window: [f32; BSRNN_N_FFT],
    in_ring_l: [f32; BSRNN_N_FFT],
    in_ring_r: [f32; BSRNN_N_FFT],
    spec_frames_l: [[Complex32; BSRNN_FREQ_BINS]; 3],
    spec_frames_r: [[Complex32; BSRNN_FREQ_BINS]; 3],
    stems_l: [[Complex32; BSRNN_FREQ_BINS]; BSRNN_STEM_COUNT],
    stems_r: [[Complex32; BSRNN_FREQ_BINS]; BSRNN_STEM_COUNT],
    ola_ring: [[[f32; BSRNN_N_FFT]; 2]; BSRNN_STEM_COUNT],
    time_buf: [f32; BSRNN_N_FFT],
    new_spec_l: [Complex32; BSRNN_FREQ_BINS],
    new_spec_r: [Complex32; BSRNN_FREQ_BINS],
    stem_spec_l: [Complex32; BSRNN_FREQ_BINS],
    stem_spec_r: [Complex32; BSRNN_FREQ_BINS],
}

impl BsrnnHeapBuffers {
    fn new_boxed() -> Box<Self> {
        // Zero-initialize directly on the heap via `alloc_zeroed` so `Self` (~200 KB)
        // is NEVER constructed on the thread stack even in unoptimized debug builds!
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            let mut boxed = Box::from_raw(ptr);
            let inv_n = 1.0 / (BSRNN_N_FFT as f32);
            for n in 0..BSRNN_N_FFT {
                let w = (PI * (n as f32) / (BSRNN_N_FFT as f32)).sin();
                boxed.analysis_window[n] = w;
                boxed.synthesis_window[n] = w * 0.5 * inv_n;
            }
            boxed
        }
    }
}

pub struct OzoneBsrnnOnnxSeparator {
    session: Session,
    active_tier: GpuExecutionTier,

    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
    fft_scratch_fwd: Vec<Complex32>,
    fft_scratch_inv: Vec<Complex32>,

    in_cursor: usize,
    ola_cursor: usize,
    ws: Box<BsrnnHeapBuffers>,

    /// Pre-allocated contiguous ONNX I/O buffers.
    input_spec_buf: Vec<f32>,
    states_buf: Vec<f32>,
    norm_buf: [f32; 2],
    output_spec_buf: Vec<f32>,
}

impl OzoneBsrnnOnnxSeparator {
    /// Loads `ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx` from the default `current_exe()`-anchored asset path.
    pub fn from_default_assets() -> Result<Self, String> {
        let model_path = resolve_ozone_bsrnn_onnx_path().ok_or_else(|| {
            "Could not locate ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx in assets/".to_string()
        })?;
        Self::from_model_path(&model_path)
    }

    /// Loads `ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx` from an explicit path on `GpuRuntimeManager`.
    pub fn from_model_path(model_path: &Path) -> Result<Self, String> {
        let rt = GpuRuntimeManager::global();
        let (session, active_tier) = rt.create_session(model_path)?;

        let mut planner = RealFftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(BSRNN_N_FFT);
        let fft_inverse = planner.plan_fft_inverse(BSRNN_N_FFT);
        let fft_scratch_fwd = fft_forward.make_scratch_vec();
        let fft_scratch_inv = fft_inverse.make_scratch_vec();

        Ok(Self {
            session,
            active_tier,
            fft_forward,
            fft_inverse,
            fft_scratch_fwd,
            fft_scratch_inv,
            in_cursor: 0,
            ola_cursor: 0,
            ws: BsrnnHeapBuffers::new_boxed(),
            input_spec_buf: vec![0.0f32; OZONE_BSRNN_SPEC_LEN],
            states_buf: vec![0.0f32; OZONE_BSRNN_STATES_LEN],
            norm_buf: [0.0f32; 2],
            output_spec_buf: vec![0.0f32; OZONE_BSRNN_SPEC_LEN],
        })
    }

    #[inline]
    pub fn active_tier(&self) -> GpuExecutionTier {
        self.active_tier
    }

    #[inline]
    pub fn states_slice(&self) -> &[f32] {
        &self.states_buf
    }

    #[inline]
    pub fn normalize_states(&self) -> [f32; 2] {
        self.norm_buf
    }

    pub fn reset(&mut self) {
        self.ws.in_ring_l.fill(0.0);
        self.ws.in_ring_r.fill(0.0);
        self.in_cursor = 0;
        for f in 0..3 {
            self.ws.spec_frames_l[f].fill(Complex32::new(0.0, 0.0));
            self.ws.spec_frames_r[f].fill(Complex32::new(0.0, 0.0));
        }
        for s in 0..BSRNN_STEM_COUNT {
            self.ws.ola_ring[s][0].fill(0.0);
            self.ws.ola_ring[s][1].fill(0.0);
        }
        self.ola_cursor = 0;
        self.states_buf.fill(0.0);
        self.norm_buf = [0.0; 2];
    }

    /// Runs a single spectral step on pre-computed 3-frame stereo complex spectrograms
    /// `[frame_0, frame_1, frame_2]` and returns the 4 separated complex stem spectra for `frame_0`:
    /// `[Vocals, Drums, Bass, Other]` where `Vocals + Drums + Bass + Other == frame_0` identically!
    pub fn infer_spectral_step(
        &mut self,
        frames_l: &[[Complex32; BSRNN_FREQ_BINS]; 3],
        frames_r: &[[Complex32; BSRNN_FREQ_BINS]; 3],
        out_stems_l: &mut [[Complex32; BSRNN_FREQ_BINS]; BSRNN_STEM_COUNT],
        out_stems_r: &mut [[Complex32; BSRNN_FREQ_BINS]; BSRNN_STEM_COUNT],
    ) -> Result<(), String> {
        // Pack [2, 1, 1025, 3, 2]
        for k in 0..BSRNN_FREQ_BINS {
            let base_l = (k * 3) * 2;
            let base_r = ((BSRNN_FREQ_BINS + k) * 3) * 2;
            for f in 0..3 {
                let cl = frames_l[f][k];
                let cr = frames_r[f][k];
                self.input_spec_buf[base_l + f * 2] = cl.re;
                self.input_spec_buf[base_l + f * 2 + 1] = cl.im;
                self.input_spec_buf[base_r + f * 2] = cr.re;
                self.input_spec_buf[base_r + f * 2 + 1] = cr.im;
            }
        }

        {
            let t_spec = TensorRef::from_array_view((
                [2usize, 1, BSRNN_FREQ_BINS, 3, 2],
                self.input_spec_buf.as_slice(),
            ))
            .map_err(|e| e.to_string())?;
            let t_states = TensorRef::from_array_view((
                [2usize, 2, 2, 1, 13, 128],
                self.states_buf.as_slice(),
            ))
            .map_err(|e| e.to_string())?;
            let t_norm =
                TensorRef::from_array_view(([2usize, 1], self.norm_buf.as_slice()))
                    .map_err(|e| e.to_string())?;

            let outputs = self
                .session
                .run(ort::inputs![
                    "input_spectrogram" => t_spec,
                    "input_states" => t_states,
                    "input_states_normalize" => t_norm,
                ])
                .map_err(|e| format!("BSRNN ONNX run failed: {}", e))?;

            let (_, spec_slice) = outputs["output_spectrogram"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            let (_, next_states) = outputs["output_states"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            let (_, next_norm) = outputs["output_states_normalize"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;

            self.output_spec_buf.copy_from_slice(spec_slice);
            self.states_buf.copy_from_slice(next_states);
            self.norm_buf.copy_from_slice(next_norm);
        }

        // Unpack [2, 3, 1, 1025, 1, 2]:
        // ONNX dim1 order in Ozone 12 Master Rebalance / StemEQ is `[0=Vocals, 1=Bass, 2=Drums]`.
        // Map to ToneForge `StemKind` `[0=Vocals, 1=Drums, 2=Bass, 3=Other]` and compute exact
        // complex residual `Other = X_frame0 - (Vocals + Drums + Bass)`.
        for k in 0..BSRNN_FREQ_BINS {
            let l_voc_off = (0 * BSRNN_FREQ_BINS + k) * 2;
            let l_bas_off = (1 * BSRNN_FREQ_BINS + k) * 2;
            let l_drm_off = (2 * BSRNN_FREQ_BINS + k) * 2;

            let r_voc_off = (3 * BSRNN_FREQ_BINS + k) * 2;
            let r_bas_off = (4 * BSRNN_FREQ_BINS + k) * 2;
            let r_drm_off = (5 * BSRNN_FREQ_BINS + k) * 2;

            let mut v_l = Complex32::new(self.output_spec_buf[l_voc_off], self.output_spec_buf[l_voc_off + 1]);
            let mut b_l = Complex32::new(self.output_spec_buf[l_bas_off], self.output_spec_buf[l_bas_off + 1]);
            let mut d_l = Complex32::new(self.output_spec_buf[l_drm_off], self.output_spec_buf[l_drm_off + 1]);

            let mut v_r = Complex32::new(self.output_spec_buf[r_voc_off], self.output_spec_buf[r_voc_off + 1]);
            let mut b_r = Complex32::new(self.output_spec_buf[r_bas_off], self.output_spec_buf[r_bas_off + 1]);
            let mut d_r = Complex32::new(self.output_spec_buf[r_drm_off], self.output_spec_buf[r_drm_off + 1]);

            if k == 0 || k == BSRNN_FREQ_BINS - 1 {
                v_l.im = 0.0;
                b_l.im = 0.0;
                d_l.im = 0.0;
                v_r.im = 0.0;
                b_r.im = 0.0;
                d_r.im = 0.0;
            }

            let mut x0_l = frames_l[0][k];
            let mut x0_r = frames_r[0][k];
            if k == 0 || k == BSRNN_FREQ_BINS - 1 {
                x0_l.im = 0.0;
                x0_r.im = 0.0;
            }

            let o_l = x0_l - (v_l + d_l + b_l);
            let o_r = x0_r - (v_r + d_r + b_r);

            out_stems_l[0][k] = v_l;
            out_stems_l[1][k] = d_l;
            out_stems_l[2][k] = b_l;
            out_stems_l[3][k] = o_l;

            out_stems_r[0][k] = v_r;
            out_stems_r[1][k] = d_r;
            out_stems_r[2][k] = b_r;
            out_stems_r[3][k] = o_r;
        }

        Ok(())
    }

    /// Processes a single `512`-sample stereo hop through the real ONNX BSRNN graph and writes
    /// 4 separated time-domain stems `[Vocals, Drums, Bass, Other]` into `out_hop`.
    ///
    /// Algorithmic causal latency: `2048` (STFT/OLA) + `1024` (2-hop lookahead) = `3,072` samples.
    pub fn process_hop(
        &mut self,
        in_hop: &BsrnnInputHop,
        out_hop: &mut BsrnnOutputHop,
    ) -> Result<(), String> {
        for i in 0..BSRNN_HOP_SIZE {
            let pos = (self.in_cursor + i) & (BSRNN_N_FFT - 1);
            self.ws.in_ring_l[pos] = in_hop.frames[i][0];
            self.ws.in_ring_r[pos] = in_hop.frames[i][1];
        }
        self.in_cursor = (self.in_cursor + BSRNN_HOP_SIZE) & (BSRNN_N_FFT - 1);

        for n in 0..BSRNN_N_FFT {
            let idx = (self.in_cursor + n) & (BSRNN_N_FFT - 1);
            self.ws.time_buf[n] = self.ws.in_ring_l[idx] * self.ws.analysis_window[n];
        }
        let _ = self.fft_forward.process_with_scratch(
            &mut self.ws.time_buf,
            &mut self.ws.new_spec_l,
            &mut self.fft_scratch_fwd,
        );

        for n in 0..BSRNN_N_FFT {
            let idx = (self.in_cursor + n) & (BSRNN_N_FFT - 1);
            self.ws.time_buf[n] = self.ws.in_ring_r[idx] * self.ws.analysis_window[n];
        }
        let _ = self.fft_forward.process_with_scratch(
            &mut self.ws.time_buf,
            &mut self.ws.new_spec_r,
            &mut self.fft_scratch_fwd,
        );

        // Shift 3-frame lookahead buffer: [0] <= [1], [1] <= [2], [2] <= newest
        self.ws.spec_frames_l[0] = self.ws.spec_frames_l[1];
        self.ws.spec_frames_l[1] = self.ws.spec_frames_l[2];
        self.ws.spec_frames_l[2] = self.ws.new_spec_l;

        self.ws.spec_frames_r[0] = self.ws.spec_frames_r[1];
        self.ws.spec_frames_r[1] = self.ws.spec_frames_r[2];
        self.ws.spec_frames_r[2] = self.ws.new_spec_r;

        // Pack & run ONNX inference directly from `self.ws.spec_frames_l/r` into `self.ws.stems_l/r`
        for k in 0..BSRNN_FREQ_BINS {
            let base_l = (k * 3) * 2;
            let base_r = ((BSRNN_FREQ_BINS + k) * 3) * 2;
            for f in 0..3 {
                let cl = self.ws.spec_frames_l[f][k];
                let cr = self.ws.spec_frames_r[f][k];
                self.input_spec_buf[base_l + f * 2] = cl.re;
                self.input_spec_buf[base_l + f * 2 + 1] = cl.im;
                self.input_spec_buf[base_r + f * 2] = cr.re;
                self.input_spec_buf[base_r + f * 2 + 1] = cr.im;
            }
        }

        {
            let t_spec = TensorRef::from_array_view((
                [2usize, 1, BSRNN_FREQ_BINS, 3, 2],
                self.input_spec_buf.as_slice(),
            ))
            .map_err(|e| e.to_string())?;
            let t_states = TensorRef::from_array_view((
                [2usize, 2, 2, 1, 13, 128],
                self.states_buf.as_slice(),
            ))
            .map_err(|e| e.to_string())?;
            let t_norm =
                TensorRef::from_array_view(([2usize, 1], self.norm_buf.as_slice()))
                    .map_err(|e| e.to_string())?;

            let outputs = self
                .session
                .run(ort::inputs![
                    "input_spectrogram" => t_spec,
                    "input_states" => t_states,
                    "input_states_normalize" => t_norm,
                ])
                .map_err(|e| format!("BSRNN ONNX run failed: {}", e))?;

            let (_, spec_slice) = outputs["output_spectrogram"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            let (_, next_states) = outputs["output_states"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            let (_, next_norm) = outputs["output_states_normalize"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;

            self.output_spec_buf.copy_from_slice(spec_slice);
            self.states_buf.copy_from_slice(next_states);
            self.norm_buf.copy_from_slice(next_norm);
        }

        for k in 0..BSRNN_FREQ_BINS {
            let l_voc_off = (0 * BSRNN_FREQ_BINS + k) * 2;
            let l_bas_off = (1 * BSRNN_FREQ_BINS + k) * 2;
            let l_drm_off = (2 * BSRNN_FREQ_BINS + k) * 2;

            let r_voc_off = (3 * BSRNN_FREQ_BINS + k) * 2;
            let r_bas_off = (4 * BSRNN_FREQ_BINS + k) * 2;
            let r_drm_off = (5 * BSRNN_FREQ_BINS + k) * 2;

            let mut v_l = Complex32::new(self.output_spec_buf[l_voc_off], self.output_spec_buf[l_voc_off + 1]);
            let mut b_l = Complex32::new(self.output_spec_buf[l_bas_off], self.output_spec_buf[l_bas_off + 1]);
            let mut d_l = Complex32::new(self.output_spec_buf[l_drm_off], self.output_spec_buf[l_drm_off + 1]);

            let mut v_r = Complex32::new(self.output_spec_buf[r_voc_off], self.output_spec_buf[r_voc_off + 1]);
            let mut b_r = Complex32::new(self.output_spec_buf[r_bas_off], self.output_spec_buf[r_bas_off + 1]);
            let mut d_r = Complex32::new(self.output_spec_buf[r_drm_off], self.output_spec_buf[r_drm_off + 1]);

            let mut x0_l = self.ws.spec_frames_l[0][k];
            let mut x0_r = self.ws.spec_frames_r[0][k];
            if k == 0 || k == BSRNN_FREQ_BINS - 1 {
                v_l.im = 0.0;
                b_l.im = 0.0;
                d_l.im = 0.0;
                v_r.im = 0.0;
                b_r.im = 0.0;
                d_r.im = 0.0;
                x0_l.im = 0.0;
                x0_r.im = 0.0;
            }

            let o_l = x0_l - (v_l + d_l + b_l);
            let o_r = x0_r - (v_r + d_r + b_r);

            self.ws.stems_l[0][k] = v_l;
            self.ws.stems_l[1][k] = d_l;
            self.ws.stems_l[2][k] = b_l;
            self.ws.stems_l[3][k] = o_l;

            self.ws.stems_r[0][k] = v_r;
            self.ws.stems_r[1][k] = d_r;
            self.ws.stems_r[2][k] = b_r;
            self.ws.stems_r[3][k] = o_r;
        }

        for s in 0..BSRNN_STEM_COUNT {
            self.ws.stem_spec_l = self.ws.stems_l[s];
            self.ws.stem_spec_r = self.ws.stems_r[s];

            let _ = self.fft_inverse.process_with_scratch(
                &mut self.ws.stem_spec_l,
                &mut self.ws.time_buf,
                &mut self.fft_scratch_inv,
            );
            for n in 0..BSRNN_N_FFT {
                let pos = (self.ola_cursor + n) & (BSRNN_N_FFT - 1);
                self.ws.ola_ring[s][0][pos] += self.ws.time_buf[n] * self.ws.synthesis_window[n];
            }

            let _ = self.fft_inverse.process_with_scratch(
                &mut self.ws.stem_spec_r,
                &mut self.ws.time_buf,
                &mut self.fft_scratch_inv,
            );
            for n in 0..BSRNN_N_FFT {
                let pos = (self.ola_cursor + n) & (BSRNN_N_FFT - 1);
                self.ws.ola_ring[s][1][pos] += self.ws.time_buf[n] * self.ws.synthesis_window[n];
            }

            for i in 0..BSRNN_HOP_SIZE {
                let pos = (self.ola_cursor + i) & (BSRNN_N_FFT - 1);
                out_hop.stems[s][i] = [self.ws.ola_ring[s][0][pos], self.ws.ola_ring[s][1][pos]];
                self.ws.ola_ring[s][0][pos] = 0.0;
                self.ws.ola_ring[s][1][pos] = 0.0;
            }
        }

        self.ola_cursor = (self.ola_cursor + BSRNN_HOP_SIZE) & (BSRNN_N_FFT - 1);
        Ok(())
    }

    /// Separates a complete offline stereo buffer into 4 phase-locked stems `[Vocals, Drums, Bass, Other]`
    /// with exact causal delay compensation so `Vocals + Drums + Bass + Other == mix` (`< 1e-5` residual).
    pub fn separate_4stems_offline(
        &mut self,
        mix: &[StereoFrame],
    ) -> Result<[Vec<StereoFrame>; BSRNN_STEM_COUNT], String> {
        let len = mix.len();
        if len == 0 {
            return Ok(std::array::from_fn(|_| Vec::new()));
        }

        self.reset();

        // Total causal delay through BSRNN hop processor:
        // - `in_ring` starts with 1536 zeros before the first 512-sample hop, and `ola_ring` outputs
        //   after 3 more hops => 1536 samples STFT/OLA delay + 2 * 512 (1024) 2-frame lookahead = 2,560 samples!
        // Let's verify exact sample delay:
        // At hop m=0 (samples 0..512 placed at in_ring[0..512], in_cursor=512):
        //   window reads in_ring[512..2048] (zeros) then in_ring[0..512] (samples 0..512 at window positions 1536..2048).
        //   That frame enters `spec_frames[2]` at m=0, moves to `spec_frames[1]` at m=1, and reaches `spec_frames[0]` at m=2!
        //   At m=2, iFFT places samples 0..512 at ola positions `ola_cursor + 1536 .. ola_cursor + 2048`.
        //   Since `ola_cursor` advances by 512 per hop, those samples are read out 3 hops later at m = 2 + 3 = 5!
        //   Hop m=5 outputs hop indices `5 * 512 .. 6 * 512` = `2560 .. 3072`!
        //   Therefore, input sample `0` emerges at output sample `2560` (`5 * 512`), and COLA ramp is full after 3 pre-hops!
        //   To give input sample `0` full 4-way COLA window coverage (`w^2` sum = 2.0 even at sample 0!),
        //   we prepend `BSRNN_N_FFT` (`2048`) zero samples before `mix[0]`, so `mix[0]` emerges at `2048 + 2560 = 4608`!
        let pre_pad = BSRNN_N_FFT;
        let algo_delay = 5 * BSRNN_HOP_SIZE; // 2,560 samples
        let total_offset = pre_pad + algo_delay; // 4,608 samples
        let needed_out = total_offset + len;
        let num_hops = needed_out.div_ceil(BSRNN_HOP_SIZE);

        let mut stems: [Vec<StereoFrame>; BSRNN_STEM_COUNT] =
            std::array::from_fn(|_| Vec::with_capacity(len));

        let mut in_hop = BsrnnInputHop::default();
        let mut out_hop = BsrnnOutputHop::default();

        for h in 0..num_hops {
            let base = h * BSRNN_HOP_SIZE;
            for i in 0..BSRNN_HOP_SIZE {
                let global_idx = base + i;
                if global_idx >= pre_pad && (global_idx - pre_pad) < len {
                    in_hop.frames[i] = mix[global_idx - pre_pad];
                } else {
                    in_hop.frames[i] = [0.0, 0.0];
                }
            }

            self.process_hop(&in_hop, &mut out_hop)?;

            for i in 0..BSRNN_HOP_SIZE {
                let out_idx = base + i;
                if out_idx >= total_offset && (out_idx - total_offset) < len {
                    for s in 0..BSRNN_STEM_COUNT {
                        stems[s].push(out_hop.stems[s][i]);
                    }
                }
            }
        }

        // Enforce exact sample-level residual on `Other` so V + D + B + O === mix to IEEE-754 precision
        for n in 0..len {
            for ch in 0..2 {
                stems[3][n][ch] =
                    mix[n][ch] - (stems[0][n][ch] + stems[1][n][ch] + stems[2][n][ch]);
            }
        }

        Ok(stems)
    }
}
