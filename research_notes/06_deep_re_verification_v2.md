# 06 — Deep Reverse-Engineering Verification v2 (`REA v6.8.0` + `Craft Anything v0.4.0 Deep Mode` + `Ghidra 12.1.4`)

## 0. Provenance & Clean-Room Verification Summary

| Target Binary / Symbol Archive | File Size | SHA-256 | ImageBase & Tooling |
|---|---:|---|---|
| `extracted_targets/ozone12/megacore_component/MEGACORE_WIN/iZOzone12Core.dll` | `579,805,696` B | `44a104fee42335b3ec2b37c30467172f8abcca5b8f86e83a45a270b8a1f08254` | `0x180000000` (`Craft Anything --deep` Evidence `ev_7b009f3773a2f8e46a51c807e97d05e799e098c33b5638313cc1a5a39604f3fa` + `.pdata` Ghidra `12.1.4`) |
| `extracted_targets/ozone12/.../Exciter Cache/exciter-*.dat` (`38` files) | `121,316,552` B | Verified `0x8BAD43FC` (`Triode`) & `0x9DFBCC23` (`DualTriode`) | `8000.000000.dat` .. `768000.000000.dat` |
| `extracted_targets/ffproq4/FabFilter Pro-Q 4.clap` | `9,340,688` B | `1a96c5239bc5fe4b5a52365490fffb08f373d4439633edf8ef95efba923f1d11` | `0x180000000` (`rea v6.8.0` MCP + Ghidra `12.1.4`) |
| `extracted_targets/soothe3/soothe3_symbols.map` & `DigiPageTable.xml` | `26,182,145` B | `6271164c9c4ccbaca448009878b91587d996d544f08ef98df68c325517a480f7` | `0x180000000` (`142,755` symbols, `DbgHelp!UnDecorateSymbolName`) |
| `extracted_targets/flstudio26/LuxeVerb_x64.dll` + `ildsp_x64.dll` | `4,341,248` B / `4,197,376` B | `498d8f8dffbfda3f...` / `b96354e1ee4f0962...` | `0x400000` / `0x180000000` (`Craft Anything --deep` + Ghidra `12.1.4`) |
| `extracted_targets/flstudio26/Gross Beat_x64.dll` | `5,850,112` B | `9025d93836796c80c392184998ee44cb49d2734739bb5dc211697eb9e0d3d4a0` | `0x400000` (Delphi VMT + Ghidra `12.1.4`) |
| `extracted_targets/flstudio26/Emphasis_x64.dll` | `4,142,080` B | `e64ca492e964e6c38dfcb245582552507d6d4ec1c568cb79d52e2dcff4f275c7` | `0x400000` -> `ildsp_x64.dll!Limiter_CreateInstance` (`0x1801AAFD0`) |

> **Strict Clean-Room Rule**: Only mathematical DSP topologies, filter pole/zero polynomials, spectral window formulations, and neural tensor contracts were extracted for clean-room re-implementation in Rust. All third-party licensing/DRM wrapper code was strictly excluded.

---

## 1. Question 1 — `iZOzone12Core.dll`: Neural Unlimiter, Master Assistant CNNs, DK-Triode Exciter & IRC V Maximizer

### 1.1 `delimit_exp009_ep080_fixed2.onnx` & `DSP::OzoneUnlimiter` (`VA 0x1834DCB90`, `0x1834E78C0`, `0x1834EB540`)
- **Xrefs & Vtables**:
  - `"models/delimit_exp009_ep080_fixed2"` @ `0x18568D3E0` (`16` RIP-relative `LEA` xrefs including static descriptor init `0x180BCE1B0` -> `0x1864D6DD0` and `OzoneUnlimiter` `0x1834DCE52` / `0x1834DD4E5`) + `".onnx"` @ `0x18572F98C`.
  - Embedded ONNX @ file offset `0x1D5249E4` (`5,580,442` B, ONNX IR v7, opset 17, 13-subband causal Band-Split RNN).
  - `DSP::OzoneUnlimiter` vtables @ `0x18510B7F8` (`93` methods) & `0x18568D6E0` (`75` methods); `SourceSeparationTwo::OnnxInference` vtable @ `0x18572F9D8` (`ORT_API_VERSION = 16`).
- **Exact STFT, Window, Hop & Sample-Rate Contract (`0x1864D6DD0`)**:
  - **Operating Sample Rate**: Fixed **`48,000.0 Hz`** (`0x473B8000` at `+0x4C`; resampled to/from host SR via `iZResampler` `0x182812020`).
  - **FFT / Hop**: **`fftSize = 2048`** (`1025` positive bins), **`hopSize = 1024`** (`50%` overlap, `21.333 ms` hop @ `48 kHz`), `midSideInput = false`, `midSideOutput = false`.
  - **Analysis & WOLA Synthesis Window (`0x182714660` & `0x182715380`)**: Half-sample shifted Hann window:
    $$w[n] = 0.5 - 0.5\cos\!\left(\frac{2\pi(n + 0.5)}{2048}\right), \qquad n \in \{0, \dots, 2047\}$$
    with exact COLA denominator normalization $\frac{1}{w[n]^2 + w[n + 1024]^2}$ on synthesis (`1/2048` inverse FFT scale applied in synthesis, **unscaled** forward real FFT fed into ONNX).
- **Exact Tensor Shapes & Semantics**:
  - `input_spectrogram`: `float32 [1, 1, 1025, 2, 2]` = `[Batch=1, Frames=1, Bins=1025, StereoCh=2, ComplexReIm=2]`. Mono-downmixed signal is passed twice (`{mono_ptr, mono_ptr}` in `0x1834DD340`), so `Ch0 == Ch1`, with `Im[0] = Im[1024] = 0.0f`.
  - `input_states` / `output_states`: `float32 [1, 2, 2, 1, 13, 128]` = `[Batch=1, Layers=2, LSTM_hc=2, CausalDir=1, Subbands=13, Hidden=128]` (`13` subband widths `[5, 6, 9, 12, 16, 22, 30, 41, 55, 74, 100, 135, 520]` summing to `1025`), zeroed on reset and copied `output_states -> input_states` (`6,656` floats) on every hop (`0x1834EB9C8`).
  - `output_spectrogram`: `float32 [1, 1, 1, 1025, 1, 2]` (`"DelimitedMixEnvelope"` @ `0x18568D1F0`).
- **Exact Transient Restoration Equation (`DSP::OzoneUnlimiter::processBlock` @ `0x1834DCB90`)**:
  $$\alpha = \frac{\text{Amount}_{\%}}{100}, \qquad T_{\text{lin}} = 10^{\text{Threshold}_{\text{dB}} / 20}, \qquad m_{\text{norm}}[n] = \frac{1}{C \, T_{\text{lin}}} \sum_{c=0}^{C-1} x_c[n]$$
  $$r[n] = \operatorname{iSTFT}_{48\text{k}\to f_s}\!\Big(\operatorname{BSRNN}_{\text{ONNX}}\big(\operatorname{STFT}_{2048,1024}(m_{\text{norm}})\big)\Big)[n]$$
  $$g[n] = 1.0 + \alpha \cdot \max(r[n],\, 0.0), \qquad y_c[n] = x_c[n - D_{\text{lat}}] \cdot g[n]$$

### 1.2 `gparamsBN8.0.npz` & `instrument_classifier_params.npz` & Master Assistant (`VA 0x18340CA00`..`0x183424140`)
- **Shared Mel-Spectrogram Front-End (`0x18340CA00`, `0x18340D5D0`, `0x18340DB70`)**:
  - Mono downmix resampled to **`32,000.0 Hz`** (`0x46FA0000`), `N_fft = 2048` (`1025` bins, `0..16 kHz`), `hop = 1024` (`32.0 ms`), periodic Hann $w[n] = 0.5 - 0.5\cos(2\pi n / 2048)$, unnormalized power $P[t,k] = \Re^2 + \Im^2$.
  - **80-Band Librosa Slaney Mel Filterbank (`0x18340D5D0`)**: `htk=False`, `norm='slaney'` (`0..16,000 Hz`, linear `200/3 Hz` below `1000 Hz`, log step `1.0711702875` above `1000 Hz`, area normalization $2 / (f_{m+2} - f_m)$).
  - Log-power & z-score: $M_{\text{dB}}[t,m] = 10\log_{10}(\max(S[t,m], 10^{-10}))$, standardized per Mel band as $\tilde{X}[0,t,m] = (M_{\text{dB}}[t,m] - \mu[m])/\sigma[m]$.
- **`gparamsBN8.0.npz` (`GenreClassifierNeuralNet::forward` @ `0x18340F340`)**:
  - **Exact Window**: **`T = 150` frames × `F = 80` mels** (`4.80 s`, `hop = 94` frames in `0x18340828B`).
  - **Critical Kernel & BN Semantics (`0x183422CC0` & `0x183421E50`)**:
    1. **Theano/Lasagne 2D-Flipped Valid Convolution**: Indexes `W[oc, ic, kH - 1 - kh, kW - 1 - kw]`.
    2. **BatchNorm Parameter Order**: Stored as **`[beta, gamma, mean, inv_std]`** where `inv_std = 1 / sqrt(var + eps)` is pre-inverted in `.npz`: $y = (x - \text{mean}) \cdot \text{inv\_std} \cdot \gamma + \beta$, applied *after* `MaxPool2D` and *before* `ReLU`!
  - **Exact Dimensions**:
    - `[1, 150, 80]` -> `Conv(3×3)` (`arr_0 [64,1,3,3]`) + `Pool(1×1)` + `BN` + `ReLU` -> `[64, 148, 78]`
    - -> `Conv(3×3)` (`arr_5 [32,64,3,3]`) + `Pool(3×3, s=3)` + `BN` + `ReLU` -> `[32, 48, 25]`
    - -> `Conv(3×3)` (`arr_10 [128,32,3,3]`) + `Pool(1×1)` + `BN` + `ReLU` -> `[128, 46, 23]`
    - -> `Conv(3×3)` (`arr_15 [64,128,3,3]`) + `Pool(3×3, s=3)` + `BN` + `ReLU` -> **`[64, 14, 7]`** -> NCHW flatten **`6,272`**
    - -> Dense `arr_20 [6272, 256]` + `BN` (`arr_21..24`) + `ReLU` -> Dense `arr_25 [256, 64]` + `BN` (`arr_26..29`) + `ReLU` -> Dense `arr_30 [64, 10]` + `arr_31 [10]` -> Clamped Softmax (`0x183424140`).
  - **10 Exact Genre Classes (`0x1855A7AB0`)**: `0: Cinematic (Classical), 1: Country, 2: EDM, 3: Folk, 4: Hip-hop/Rap, 5: Jazz, 6: Pop, 7: RnB/Soul, 8: Reggae, 9: Rock`.
- **`instrument_classifier_params.npz` (`InstrumentClassifierNeuralNet::forward` @ `0x183412400`)**:
  - **Exact Window**: **`T = 31` frames × `F = 80` mels** (`0.992 s`).
  - **Exact Dimensions**:
    - `[1, 31, 80]` -> Flipped valid `Conv2d(1×4)` (`layer0_W [16,1,1,4]`) -> `[16, 31, 77]` -> `MaxPool2d(1×4, s=(1,4))` + `layer0_b [16]` + `ReLU` -> `[16, 31, 19]`
    - -> Flipped valid `Conv2d(2×2)` (`layer1_W [32,16,2,2]`) -> `[32, 30, 18]` -> `MaxPool2d(2×2, s=(2,2))` + `layer1_b [32]` + `ReLU` -> **`[32, 15, 9]`** -> NCHW flatten **`4,320`**
    - -> Dense `layer2_W [4320, 1000]` + `b` + `ReLU` -> Dense `layer3_W [1000, 1000]` + `b` + `ReLU` -> Dense `layer4_W [1000, 10]` + `b` -> Softmax.
  - **10 Exact Instrument Classes (`0x186139A00`)**: `0: Vocals, 1: Guitar, 2: Drums, 3: Bass, 4: Piano, 5: Organ, 6: Brass, 7: Woodwind, 8: Strings, 9: Synth`.
- **Master Assistant (`IZAssistant::AnalysisComposer` @ `0x181841CD0`..`0x181843AC0`)**:
  - Computes amplitude-weighted window posteriors (`0x183408B10`), selects `DSP/AssistantPresets/<genre>-modern-pro.xml` + `DSP/Target Curves/<Genre>.json` (`0/250/2000/8000 Hz` macro bands), measures Integrated LUFS, True Peak, Crest Factor, and 4-band stereo width, and auto-calibrates EQ, Stabilizer, Impact, Imager, Unlimiter, and `IRC V` Gain/Threshold to hit target LUFS (`-8.5` to `-13.5 LUFS`).

### 1.3 `DSP::OzoneExciter` Multi-Rate `.dat` LUTs & `4x` Oversampling (`VA 0x181DC5C50`, `0x181DC88D0`, `0x181DF0580`, `0x181DB8BA0`)
- Uses `DSP::OversampleInt` (`0x182796500`, `4x` integer `128`-tap Kaiser $\beta=9.0$ polyphase FIR) and loads **`<prefix><effective_sr:.6f>.dat`** (`1.0 * hostSR` and `4.0 * hostSR`), explaining all `19` rates per circuit (`8000` .. `768000 = 4 × 192000 Hz`).
- **`Triode` (`0x8BAD43FC`, `2,074,532` B)** vs **`DualTriode` (`0x9DFBCC23`, `4,149,044` B, schema `v4`)**:
  - `DualTriode` header (`20` B: magic, version `4`, `f32 sr`, `dc1`, `dc2`) + `160` B DC operating points (`B+ = 286.0 V`) + **two** `2,074,432`-byte `KState<3,2,2,KorenFunction>` cascaded 12AX7 blocks (`8` B `f64 sr` + `776` B 14 DK state-space matrices $A_c, B_c, C_c, D_x, D_u, D_i, M_{\text{inv}}, P, Q, Q_{\text{prev}}, K, A_d, B_d, C_d$ + `48` B grid `[-314.6, +143.0] × [+228.8, +288.86]` + `2,073,600` B `f64[360][360][2]` Koren 12AX7 `(i_grid, i_plate)` LUT).

### 1.4 `DSP::OzoneMaximizer` (`IRC V`, `4x` Polynomial Soft-Clip & `9x` Cubic True-Peak) (`VA 0x183655880`..`0x183C2AEE0`)
- **Soft-Clipper (`0x183650889`, `0x18365B020`, `0x183685D10`, `0x1836839A0`)**:
  - Always-on **`4x` polyphase oversampling** + **`autoDrive = 1 + w`**, tracking threshold $T_{\text{lin}} = 10^{-\text{Gain}_{\text{dB}}/20}$, `mix = SoftClipMix / 100.0f`, with degree-10 $C^5$ maximally-flat polynomial:
    $$P_4(t) = t - \frac{35}{64}t^6 + \frac{7}{8}t^7 - \frac{135}{256}t^8 + \frac{35}{256}t^9 - \frac{7}{512}t^{10}, \qquad t \in [0, 2]$$
    where `Light` uses $w = 0.19$ ($G_{\text{drive}} = 1.19$), `Moderate` uses $w = 0.525$ ($G_{\text{drive}} = 1.525$), and `Heavy` uses $w = 1.0$ ($G_{\text{drive}} = 2.0$).
- **True-Peak & `IRC V` (`0x185618450`, `0x183C2AEE0`, `0x1835825F0`..`0x1835A0B10`)**:
  - `4x` 48-tap ITU-R BS.1770-4 FIR for metering + `9x` 289-tap Kaiser ($\beta=9.0$) polyphase FIR with analytic cubic extremum solver ($3at^2 + 2bt + c = 0$) and `1.02585f` (`+0.2217 dB`) safety margin inside the `IRC V` 4-band (`150 / 900 / 5000 Hz`) transient/sustain (`5 ms` / `50 ms`) lookahead limiter.

---

## 2. Question 2 — `FabFilter Pro-Q 4.clap`: Natural Phase Oversampling & Character Saturation

### 2.1 Natural Phase Jacobi-Elliptic Polyphase Allpass Half-Band IIR (`VA 0x18011d4e0`, `0x18011d830`, `0x18018d900`)
- **`sr::IIRConverterStyle` (`vftable @ 0x180221d58`)**:
  - For $f_s \le 48,001\text{ Hz}$ (`44.1 / 48 kHz`): Stage 1 (`1x <-> 2x`) uses **Order $N = 13$** (`6` first-order allpass sections split `3 + 3` across even/odd polyphase branches) with $\omega_s = 1.3823007\text{ rad} = 0.44\pi$, achieving **> 112 dB stopband rejection and < 0.00001 dB passband ripple** in `3` samples latency! Stage 2+ (`2x <-> 4x`) and native rates $f_s > 48,001\text{ Hz}$ use **Order $N = 7$** (`3` allpass poles) with $\omega_s = 1.0367256\text{ rad}$.
  - Exact pole formula (`0x18011d4e0`): $d_4 = \cot^2(\omega_s/2)$, $u_i = (\frac{2i+1}{N} + 1)K(1/d_4^2)$, $d_6 = \operatorname{sn}^2(u_i, 1/d_4)$, $d_7 = \sqrt{(d_4^2 - d_6)(1 - d_6)}$, $a_i = \frac{d_6 + d_4 - d_7}{d_6 + d_4 + d_7}$.
- **Analog Magnitude & Phase Matched Biquad (`0x180110dc0`, `0x180111550`, `0x18019afa0`)**:
  - Evaluates Laplace prototype $|H_a(j\Omega)|^2$ at DC, Nyquist ($\omega = \pi$), $\omega_0$, and bandwidth anchors $\omega_1, \omega_2$ inside the `2x`/`4x` oversampled rate and solves exact Orfanidis/Vicanek prescribed-Nyquist coefficients realized via Cytomic trapezoidal SVF (`< 0.01 dB` error `20 Hz – 20 kHz`).

### 2.2 Character Saturation (`Subtle` & `Warm`) (`VA 0x18000bce0`, `0x18000bd50`, `0x1801234e0`, `0x180123c40`)
- **`Subtle` (`WaveShaperVecLL<&vectorClipTanHyperbolicLLFast>`, `0x1801234e0`)**:
  - Linear low-level threshold $L \in [0, 1)$, $w = 1 - L$: $y(x) = x$ for $|x| \le L$ and $\operatorname{sgn}(x)(L + w\tanh((|x|-L)/w))$ for $|x| > L$, preserving exact unity small-signal slope $f'(0) = 1.0$ and $C^2$ continuity at $|x|=L$.
- **`Warm` (`WaveShaperVec<&vectorClipTubeFast>`, `0x180123c40`)**:
  - Asymmetric tube grid-conduction polynomial (`asym = 0.55f`, `-6.153 x^2 + 3.9375 x` around $0$, normalized by $g'(0) = 3.9375$ so $f'(0) = 1.0$), generating warm 2nd-order even harmonics + oversampled anti-aliasing + DC blocker.

---

## 3. Question 3 — `soothe3_symbols.map`: Dual-Mode Spectral Suppression & Sub-`120 Hz` Smoothness

### 3.1 Dual-Mode Multi-Resolution Topology (`dsp:processor.obj`, `449,632` B `.text`, `RVA 0x008538C0..0x008C07F0`)
- `dsp::Soothe3<16, float>` (`0x008586A0`) instantiates both:
  1. **`SplitProcessor<0, 16, float>` (`0x0085F6E0` — Buffered STFT WOLA)**: `Whopper::hop<1>` (`0x008A1E40`) + `vu::sig::minphase_spec` (`0x008AF040` — real cepstrum clamped at $\ln(10^{-60}) = -138.155106$ @ `0x0206F094` -> causal Hilbert fold -> `ippsExp_32f_A21` + `ippsSinCos_32f_A21`) or `linphase_spec` (`0x008AE070`) -> `irfft` -> `windows::hann` (`0x0086F570`) -> WOLA.
  2. **`SplitProcessor<1, 16, float>` (`0x0085FAC0` — Zero-Latency Hybrid Partitioned FIR)**: Analysis-only `Whopper::hop<0>` (`0x008A0960`) -> `vu::sig::minphase_kernel` (`0x008B9840`) -> `Convolution<16, float, 32, 256, 65536>::set_ir` (`0x008B9620`) -> **32-tap dual-IR crossfaded direct FIR head** (`DirectConvolutionFF::process_sample` `0x008A7E30`) + **dyadic partitioned FFT tail (`256..65536`, `ippsAddProduct_32fc` `0x000015E0`)**.
- **Five Sub-`120 Hz` Smoothness Mechanisms**:
  1. Multi-resolution decimated `lo_band` (`0x0206BF00`) + full-rate `hi_band` (`0x0206BFC0`) crossover (`update_xover` `0x008704A0`, 4-term Blackman-Harris FIR `0x00863AE0`).
  2. Frequency-tilted per-bin asymmetric ballistics (`update_speed` `0x00870D60`, `integrate_asym` `0x008B54F0`).
  3. Zero-phase forward-backward spectral smoothing (`apply_filtfilt` `0x008B8E20` + 4-stage constant-Q `sliding_onepole_pow2p1<4>` `0x008567D0`).
  4. Gaussian-bell perceptual baseline (`Filterbank::design_gbells` `0x00876DE0`).
  5. Causal minimum-phase FIR (`minphase_kernel`) + continuous dual-filter crossfade across hops.

---

## 4. Question 4 — FL Studio 26 Native FX (`LuxeVerb_x64.dll`, `Gross Beat_x64.dll`, `Emphasis_x64.dll`, `ildsp_x64.dll`)

### 4.1 `LuxeVerb_x64.dll` -> `ildsp_x64.dll!ShimmerReverbImpl` (`VA 0x1801AB810`, `0x1801849A0`, `0x18018735D`)
- **Unitary FDN**: `16×16` Kronecker rotation $M_{16}(\theta) = R_2(\theta)^{\otimes 4}$ (`0x180186030` `case 8`; normalized Hadamard at $\theta = 45^\circ$), **16 prime base delays @ `44.1 kHz`**: `[877, 1303, 1907, 2053, 2711, 2659, 3527, 3803, 4397, 4373, 4957, 5449, 5741, 6217, 6761, 6997]` snapped to a `2,000`-prime table (`0x1803DAC50`) and scaled by $f_s / 44100$.
- **Input Diffusion**: `172k`-smp 4-pt Lagrange pre-delay (`0x180183F40`) -> stereo **8-stage Schroeder allpass chain** (`[208, 158, 561, 410, 342, 710, 575, 393]` smp @ `44.1 kHz`, gains `[0.75, 0.625, 0.70, 0.50, 0.675, 0.712, 0.60, 0.583]`, `0x180184170`).
- **Modulation & Damping**: 16-phase ($\phi_i = i/16$) parabolic-sine LFO + 4-pt Catmull-Rom Cubic Hermite (`0x180188D10`) + per-line `2124.8 Hz` bilinear TPT 1-pole crossover RT60 damping + `Freeze` lossless bypass.
- **Shimmer Pitch-Shifter (`0x1801821F0`, `0x180182910`, `0x180183180`)**: Multi-grain `150 ms` $\sin^2$ Hann window (`3.5x` overlap, 4-pt cubic read, anti-alias LPF, AP + LP/HP biquads) injected directly at the FDN input right after the input allpass diffuser.

### 4.2 `Gross Beat_x64.dll` (`Eff_Render` @ `VA 0x8420F0`, `TResampler` @ `0x613A10`..`0x617D80`)
- **Circular Buffer (`TBRDelay_32FS` `0x7802D0`)**: `3`-bar stereo ring buffer with `16`-sample dual-end wrap mirrors.
- **Fractional Interpolation**: 6-pt 5th-order Niemitalo polynomial (`0x613A10`, `0x893808..0x89384C`) + dynamic 8-pt / 16-pt anti-aliased windowed sinc (`TBRResampler` `0x613CB0`/`0x613F50`, cutoff $\pi/r$ when $r > 1$).
- **Anti-Click Architecture**: (1) **`2.0 ms` (`0.002 * f_s`) jump detector** triggering zero-latency step-discontinuity latch $c \leftarrow (c + y_{\text{prev}}) - y_{\text{raw}}[n]$ with exponential decay (`1..5 ms`) + **dual-read-head crossfade** (`0x5B0` & `0x5B8`), plus 3-stage time-tension smoother and 2-stage gate smoother.

### 4.3 `Emphasis_x64.dll` -> `ildsp_x64.dll!LimiterImpl` (`VA 0x1801AAFD0`, `0x18016781A`)
- **Dual Oversampled True-Peak Architecture**: Cascaded elliptic polyphase allpass halfband IIR (`25th/13th/9th` order, `0x180170630`/`0x180170AA0`) for oversampled saturation & `GRStatic8` sub-sample limiting + `96*M + 1`-tap sine-windowed sinc FIR True-Peak safety detector (`0x180172570`), `GRAdaptiveRel` + `ExpSmoother8` transient/sustain split, and post-limiter HF RMS transient recovery ($\min(\text{RMS}_{\text{pre}}/\text{RMS}_{\text{post}}, 2.0)$, `0x180168B9B`).

---

## 5. Question 5 — Public Web Audio Surface of `https://music.motionstudio.cloud/`
- Completed safely in `< 4s` (`HTTP 200 OK`, zero CAPTCHA/Turnstile/login wall). Public Nuxt 3 SPA entry bundle (`/_nuxt/BYA7iLOX.js`, `155,533` B) contains only OpenGraph audio metadata (`ogAudio`, `ogAudioUrl`) and standard HTML5 media playback state with zero custom WebAudio/WASM mastering DSP, confirming ToneForge's architectural decision to bypass browser WebAudio DSP 100% in Tauri desktop mode.
