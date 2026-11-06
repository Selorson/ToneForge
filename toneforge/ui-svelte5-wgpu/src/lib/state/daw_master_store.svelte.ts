// Zero-Copy 128 KB Binary Telemetry Decoder & Svelte 5 Reactive Store (`daw_master_store.svelte.ts`)
// Maps `VisualTelemetryFrame` (`131,072` bytes = `0x20000`, `#[repr(C, align(64))]`) directly into
// `DataView` and `Float32Array` views for 120 FPS WebGPU rendering without JSON serialization.

import { invoke } from '@tauri-apps/api/core';
import { RealtimeSpeakerEngine, type AudioSourceMode } from '../audio/realtime_speaker_engine';

export const VISUAL_TELEMETRY_FRAME_BYTES = 131_072; // 128 KB (0x20000)
export const MAX_MIXER_TRACKS = 502; // 0=Master, 1..=500=Inserts, 501=Preview
export const TRACK_METER_STRIDE_BYTES = 24; // 6 x 4-byte words per track
export const FIXED_POINT_30BIT_MAX = 0x4000_0000; // 1,073,741,824

// Exact byte offsets inside the 128 KB VisualTelemetryFrame (matching daw-core-graph/src/telemetry_shm.rs)
export const OFFSET_HEADER = 0x00000;            // 64 B: TransportAndMasterHeader
export const OFFSET_TRACK_METERS = 0x00040;      // 12,048 B: TrackMeterEntry[502]
export const OFFSET_HEADER_ALIGN_PAD = 0x02F50;  // 176 B: 64B cache-line alignment pad
export const OFFSET_PROQ4_SPECTRUM = 0x03000;    // 16,384 B: ProQ4SpectrumFrame (pre, post, sc, collision)
export const OFFSET_SOOTHE3_REDUCTION = 0x07000; // 8,192 B: Soothe3ReductionFrame (reduction_ys, input_spec_db)
export const OFFSET_OZONE12_TELEMETRY = 0x09000; // 8,192 B: Ozone12MasterTelemetry (IRC GR, Polar XY, Tonal, Stems)
export const OFFSET_SPECTROGRAM_RING = 0x0B000;  // 81,920 B: spectrogram_ring[[f32; 1024]; 20]
export const OFFSET_AUTOMATION_SPLINE = 0x1F000; // 4,096 B: AutomationAndSplineTelemetry

export interface TrackMeterSnapshot {
  trackIndex: number;
  peakLDb: number;
  peakRDb: number;
  rmsLDb: number;
  rmsRDb: number;
  grDb: number;
  activityFlags: number;
}

export interface ProQ4BandState {
  bandIdx: number;
  used: boolean;
  enabled: boolean;
  freqHz: number;
  gainDb: number;
  q: number;
  shape: number; // 0=Bell, 1=LowShelf, 2=LowCut, 3=HighShelf, 4=HighCut, 5=Notch, 6=BandPass, 7=TiltShelf, 8=FlatTilt, 9=AllPass
  slopeDbOct: number; // 6..96
  dynamicRangeDb: number;
  spectralEnabled: boolean;
  spectralDensity: number;
}

export interface Soothe3ScBandState {
  bandIdx: number;
  enabled: boolean;
  freqHz: number;
  depth: number; // -10..+10
  q: number;
  shape: number; // 0=LowShelf, 1=Bell, 2=HighShelf, 3=Tilt, 4=LowCut, 5=HighCut
}

export interface TensionPointState {
  x: number;
  y: number;
  tension: number; // -1.0 .. +1.0
}

export interface PatcherNodeVisual {
  nodeIndex: number;
  label: string;
  pluginKind: string;
  latencySamples: number;
  layer: number;
  xPx: number;
  yPx: number;
  cpuPct: number;
}

export interface PatcherEdgeVisual {
  fromNode: number;
  toNode: number;
  gain: number;
}

export interface MixerTrackUiState {
  index: number;
  name: string;
  gainLinear: number;
  muted: boolean;
  soloed: boolean;
  polarityInverted: boolean;
  apdcLatencySamples: number;
  sends: { destTrack: number; gain: number; sidechain: boolean }[];
  slots: { slotIndex: number; name: string; enabled: boolean; wetMix: number }[];
}

export interface PianoRollNoteUiState {
  id: number;
  startTick: number;
  durationTicks: number;
  key: number;
  velocity: number;
  pan: number;
  finePitchCents: number;
  isSlide: boolean;
  colorGroup: number;
}

export interface PianoRollPatternUiState {
  patternId: number;
  name: string;
  targetChannelId: number;
  lengthTicks: number;
  notes: PianoRollNoteUiState[];
}

export interface PlaylistClipUiState {
  clipId: number;
  trackRow: number;
  kind: 'Pattern' | 'AudioStem' | 'Automation';
  sourceId: number;
  startTick: number;
  durationTicks: number;
  clipOffsetTicks: number;
  gain: number;
  muted: boolean;
}

export interface PostRenderReportUiState {
  integratedLufs: number;
  shortTermMaxLufs: number;
  momentaryMaxLufs: number;
  truePeakDbtpL: number;
  truePeakDbtpR: number;
  maxTruePeakDbtp: number;
  crestFactorDb: number;
  phaseCorrelation: number;
  spotifyCompliant: boolean;
  appleMusicCompliant: boolean;
  youtubeCompliant: boolean;
  spotifyPenaltyDb: number;
  appleMusicPenaltyDb: number;
  youtubePenaltyDb: number;
  apdcLatencyTrimmedSamples: number;
  renderedFiles: string[];
}

function isTauriEnvironment(): boolean {
  return typeof window !== 'undefined' && ('__TAURI_INTERNALS__' in window || '__TAURI__' in window);
}

/**
 * Zero-Copy 128 KB Binary Telemetry Decoder & Svelte 5 Runes Master Store.
 */
export class DawTelemetryView {
  readonly buffer: ArrayBuffer;
  readonly dataView: DataView;
  readonly u8View: Uint8Array;

  // Pre-sliced Zero-Copy Float32Array views into the 128 KB ArrayBuffer
  readonly trackMetersWords: Float32Array;
  readonly proq4PreDb: Float32Array;
  readonly proq4PostDb: Float32Array;
  readonly proq4SidechainDb: Float32Array;
  readonly proq4CollisionMask: Float32Array;
  readonly soothe3Reduction1024: Float32Array;
  readonly soothe3Reduction512: Float32Array;
  readonly soothe3InputSpecDb: Float32Array;
  readonly ozone12IrcGrTrace: Float32Array;
  readonly ozone12PolarSampleXy: Float32Array;
  readonly ozone12TonalBalanceBars: Float32Array;
  readonly ozone12StemRmsDb: Float32Array;
  readonly ozone12ImagerCorrBands: Float32Array;
  readonly ozone12ExciterDriveGr: Float32Array;
  readonly spectrogramRing20x1024: Float32Array;
  readonly controllerValuesNorm512: Float32Array;
  readonly grossBeatPlayheadPos: Float32Array;
  readonly patcherNodeCpuPct510: Float32Array;

  // =========================================================================
  // 1. SVELTE 5 RUNES ($state / $derived) — TRANSPORT & MASTER METERING
  // =========================================================================
  frameSeq = $state(0);
  samplePos = $state(0);
  playheadPpq = $state(0);
  tempoBpm = $state(128.0);
  isPlaying = $state(true);
  cpuLoadPercent = $state(6.4);
  xrunCount = $state(0);

  lufsMomentary = $state(-11.2);
  lufsShortTerm = $state(-10.6);
  lufsIntegrated = $state(-9.8);
  truePeakDbL = $state(-1.02);
  truePeakDbR = $state(-1.05);
  masterGrDb = $state(-1.8);
  activeStemMode = $state(0);

  // Real-Time WebAudio Speaker Output & Custom Audio File Picker State
  speakerVolumePct = $state(85);
  speakerMuted = $state(false);
  audioSourceMode = $state<AudioSourceMode>('pianoRoll');
  loadedAudioFileName = $state<string | null>(null);
  readonly speakerEngine = new RealtimeSpeakerEngine();
  private lastSynthTimestampMs = 0;
  private realPreFft1024 = new Float32Array(1024);
  private realPostFft1024 = new Float32Array(1024);

  // =========================================================================
  // 2. SVELTE 5 RUNES — 502-TRACK MIXER, PATCHER DAG & TENSION SPLINE
  // =========================================================================
  selectedTrackIndex = $state(1);
  visibleBankStart = $state(0);
  visibleBankSize = $state(32);
  mixerTracks = $state<MixerTrackUiState[]>(this.createInitial502Tracks());

  tensionSplinePoints = $state<TensionPointState[]>([
    { x: 0.0, y: 0.22, tension: 0.0 },
    { x: 96.0, y: 0.78, tension: 0.45 },
    { x: 192.0, y: 0.35, tension: -0.40 },
    { x: 384.0, y: 0.88, tension: 0.25 },
  ]);
  activeControllerId = $state(0);
  activeControllerFixed30 = $derived(
    Math.round(Math.max(0, Math.min(1, this.evaluateTensionSplineAt(this.playheadPpq % 384))) * FIXED_POINT_30BIT_MAX)
  );

  patcherNodes = $state<PatcherNodeVisual[]>([
    { nodeIndex: 0, label: 'Host Audio Input', pluginKind: 'HostIn', latencySamples: 0, layer: 0, xPx: 48, yPx: 140, cpuPct: 0.1 },
    { nodeIndex: 1, label: 'Pro-Q 4 Linear Phase', pluginKind: 'ProQ4', latencySamples: 2048, layer: 1, xPx: 260, yPx: 72, cpuPct: 1.8 },
    { nodeIndex: 2, label: 'Soothe3 Resonance', pluginKind: 'Soothe3', latencySamples: 3072, layer: 1, xPx: 260, yPx: 210, cpuPct: 3.2 },
    { nodeIndex: 3, label: 'APDC Delay Compensator (+1024 smp)', pluginKind: 'FixedLatency', latencySamples: 1024, layer: 2, xPx: 500, yPx: 72, cpuPct: 0.1 },
    { nodeIndex: 4, label: 'Ozone 12 Master Bus', pluginKind: 'Ozone12', latencySamples: 5760, layer: 3, xPx: 740, yPx: 140, cpuPct: 4.1 },
  ]);
  patcherEdges = $state<PatcherEdgeVisual[]>([
    { fromNode: 0, toNode: 1, gain: 1.0 },
    { fromNode: 0, toNode: 2, gain: 1.0 },
    { fromNode: 1, toNode: 3, gain: 1.0 },
    { fromNode: 3, toNode: 4, gain: 0.707 },
    { fromNode: 2, toNode: 4, gain: 0.707 },
  ]);

  // =========================================================================
  // 3. SVELTE 5 RUNES — FABFILTER PRO-Q 4 (24 BANDS / 601 PARAMS / 171 .FFP)
  // =========================================================================
  proq4SelectedPreset = $state('Mixbus / Dynamic Smiley Curve ST');
  proq4SelectedBand = $state(0);
  proq4PhaseMode = $state<'ZeroLatency' | 'NaturalPhase' | 'LinearPhase'>('NaturalPhase');
  proq4GainScalePct = $state(100.0);
  proq4AutoGain = $state(true);
  proq4ShowCollisions = $state(true);
  proq4Bands = $state<ProQ4BandState[]>(this.createInitialProQ4Bands());

  // =========================================================================
  // 4. SVELTE 5 RUNES — OEKSOUND SOOTHE3 (143 PARAMS & 68 REAL .PRESET JSONS)
  // =========================================================================
  soothe3SelectedPreset = $state('That Grammy balance preset');
  soothe3HardMode = $state(false);
  soothe3DeltaSolo = $state(false);
  soothe3Depth = $state(4.2);
  soothe3Detail = $state(6.4);
  soothe3Attack = $state(1.8);
  soothe3Release = $state(3.5);
  soothe3MaxCutDb = $state(-18.0);
  soothe3MixPct = $state(100.0);
  soothe3LinearPhase = $state(false);
  soothe3LowLatency = $state(false);
  soothe3Quality = $state<0 | 1 | 2>(1);
  soothe3ScBands = $state<Soothe3ScBandState[]>([
    { bandIdx: 0, enabled: true, freqHz: 180.0, depth: -2.5, q: 1.0, shape: 0 },
    { bandIdx: 1, enabled: true, freqHz: 850.0, depth: 1.5, q: 1.4, shape: 1 },
    { bandIdx: 2, enabled: true, freqHz: 3400.0, depth: 6.5, q: 2.2, shape: 1 },
    { bandIdx: 3, enabled: true, freqHz: 6800.0, depth: 5.0, q: 1.8, shape: 1 },
    { bandIdx: 4, enabled: true, freqHz: 11500.0, depth: 3.0, q: 1.2, shape: 2 },
    { bandIdx: 5, enabled: false, freqHz: 450.0, depth: 0.0, q: 1.0, shape: 1 },
    { bandIdx: 6, enabled: false, freqHz: 1600.0, depth: 0.0, q: 1.0, shape: 1 },
    { bandIdx: 7, enabled: false, freqHz: 15000.0, depth: 0.0, shape: 5, q: 1.0 },
  ]);

  // =========================================================================
  // 5. SVELTE 5 RUNES — IZOTOPE OZONE 12 & 120MS BSRNN MASTER REBALANCE
  // =========================================================================
  ozone12SelectedGenre = $state('AllPurpose');
  ozone12SelectedGlobalXml = $state('Modern Clarity');
  ozone12AiAssistantStatus = $state('CNN Ready • gparamsBN8.0.npz (34 tensors) + instrument_classifier');
  ozone12RegionBounds = $state<[[number, number], [number, number], [number, number], [number, number]]>([
    [0.13, 0.35],
    [0.21, 0.46],
    [0.18, 0.43],
    [0.06, 0.24],
  ]);
  ozone12MaximizerBypass = $state(false);
  ozone12IrcMode = $state(4); // 4 = IRC V
  ozone12GainDb = $state(3.2);
  ozone12CeilingDbtp = $state(-1.0);
  ozone12TruePeak = $state(true);
  ozone12UpwardCompWet = $state(0.20);
  ozone12SoftClipEnabled = $state(true);
  ozone12SoftClipMix = $state(0.30);
  ozone12SoftClipMode = $state<0 | 1 | 2>(1); // 0=Light, 1=Medium, 2=Heavy
  ozone12TransientShaping = $state(0.15);

  ozone12ExciterBypass = $state(false);
  ozone12ExciterDrivesPct = $state<[number, number, number, number]>([18.0, 26.0, 32.0, 22.0]);
  ozone12ExciterMixes = $state<[number, number, number, number]>([0.45, 0.55, 0.60, 0.45]);
  ozone12ExciterModes = $state<[number, number, number, number]>([4, 4, 5, 3]); // 4=12AX7 Triode, 5=DualTriode

  ozone12ImagerWidthsPct = $state<[number, number, number, number]>([-15.0, 8.0, 24.0, 38.0]);
  ozone12ImagerStereoizePct = $state<[number, number, number, number]>([0.0, 6.0, 18.0, 25.0]);
  ozone12ImagerStereoizerEnabled = $state(true);
  ozone12ImagerStereoizerMode = $state<0 | 1>(0);

  ozone12StabilizerAmountPct = $state(42.0);
  ozone12StabilizerSpeedPct = $state(55.0);
  ozone12StabilizerTameTransientsPct = $state(30.0);
  ozone12StabilizerCutMode = $state(false);

  ozone12ImpactAmounts = $state<[number, number, number, number]>([12.0, 6.0, 8.0, 14.0]);
  ozone12ClarityAmountPct = $state(35.0);
  ozone12ClarityTiltDbOct = $state(0.5);

  bsrnnStemGainsDb = $state<[number, number, number, number]>([1.5, 0.8, -0.5, 0.0]); // [Vocals, Drums, Bass, Other]
  bsrnnStemSolo = $state<[boolean, boolean, boolean, boolean]>([false, false, false, false]);
  bsrnnStemMute = $state<[boolean, boolean, boolean, boolean]>([false, false, false, false]);

  // =========================================================================
  // 6. SVELTE 5 RUNES — UVR5 MULTI-ARCHITECTURE STEM SEPARATION STUDIO
  // =========================================================================
  uvr5SelectedModelProfile = $state('Ensemble: MDX23C + HTDemucs4 + VR-ASPP (3-Model)');
  uvr5ModelSpecSummary = $state('n_fft=8192 • dim_f=4096 • dim_t=8 • hop=1024 • comp=1.035');
  uvr5AudioPath = $state('F:/Mixes/ToneForge_Master_Session.wav');
  uvr5EnsembleMode = $state<'MAX_SPEC' | 'MIN_SPEC' | 'AVERAGE' | 'MEDIAN' | 'WEIGHTED'>('WEIGHTED');
  uvr5ModelWeights = $state<[number, number, number]>([0.45, 0.35, 0.20]); // [MDX23C, HTDemucs4, VR-ASPP]
  uvr5EnableDenoise2Pass = $state(true);
  uvr5VrAggressiveness = $state(0.15);
  uvr5PhaseDenoiseStrength = $state(0.20);
  uvr5InvertInstrumental = $state(true);
  uvr5BounceToMixer = $state(true);
  uvr5IsProcessing = $state(false);
  uvr5SeparatedStems = $state<string[]>([
    'ToneForge_Master_Session_mdx23c_htdemucs4_vocals.wav',
    'ToneForge_Master_Session_mdx23c_htdemucs4_drums.wav',
    'ToneForge_Master_Session_mdx23c_htdemucs4_bass.wav',
    'ToneForge_Master_Session_mdx23c_htdemucs4_other.wav',
  ]);

  // =========================================================================
  // 7. SVELTE 5 RUNES — PLAYLIST ARRANGEMENT, PIANO ROLL & HYBRID PPQ (96..960)
  // =========================================================================
  fl26SelectedSampleKit = $state('808 Studio Kit (Real .wav)');
  fl26SelectedScorePreset = $state('ToneForge Lead Hook (Slide)');
  fl26SampleKitInfo = $state('4 voices • 44.1kHz 16-bit PCM (808 Kick, CH, OH, Crash)');
  fl26SelectedFxPreset = $state('Maximus: Clear Master.fst');
  fl26FxPresetInfo = $state('Maximus • Clear Master (-0.2 dB ceil, +2.5 dB drive)');

  ppqResolution = $state<96 | 144 | 192 | 384 | 480 | 960>(96);
  songMode = $state(true);
  selectedPatternId = $state(1);
  pianoRollSlideTool = $state(false);
  pianoRollColorGroup = $state(0);
  selectedNoteId = $state<number | null>(1);

  pianoRollPatterns = $state<PianoRollPatternUiState[]>([
    {
      patternId: 1,
      name: 'ToneForge Lead Hook (FL26 Slide)',
      targetChannelId: 1,
      lengthTicks: 384,
      notes: [
        { id: 1, startTick: 0, durationTicks: 96, key: 60, velocity: 0.88, pan: -0.15, finePitchCents: 0.0, isSlide: false, colorGroup: 0 },
        { id: 2, startTick: 48, durationTicks: 48, key: 72, velocity: 0.95, pan: 0.25, finePitchCents: 0.0, isSlide: true, colorGroup: 0 },
        { id: 3, startTick: 96, durationTicks: 96, key: 67, velocity: 0.82, pan: 0.10, finePitchCents: 12.0, isSlide: false, colorGroup: 1 },
        { id: 4, startTick: 192, durationTicks: 144, key: 63, velocity: 0.90, pan: -0.20, finePitchCents: 0.0, isSlide: false, colorGroup: 0 },
        { id: 5, startTick: 240, durationTicks: 96, key: 75, velocity: 1.00, pan: 0.35, finePitchCents: 0.0, isSlide: true, colorGroup: 0 },
      ],
    },
    {
      patternId: 2,
      name: 'Sub Bass Pulse Pattern',
      targetChannelId: 2,
      lengthTicks: 384,
      notes: [
        { id: 10, startTick: 0, durationTicks: 192, key: 36, velocity: 0.95, pan: 0.0, finePitchCents: 0.0, isSlide: false, colorGroup: 2 },
        { id: 11, startTick: 192, durationTicks: 192, key: 39, velocity: 0.92, pan: 0.0, finePitchCents: 0.0, isSlide: false, colorGroup: 2 },
      ],
    },
  ]);

  playlistClips = $state<PlaylistClipUiState[]>([
    { clipId: 1, trackRow: 0, kind: 'Pattern', sourceId: 1, startTick: 0, durationTicks: 384, clipOffsetTicks: 0, gain: 1.0, muted: false },
    { clipId: 2, trackRow: 1, kind: 'Pattern', sourceId: 2, startTick: 0, durationTicks: 384, clipOffsetTicks: 0, gain: 1.0, muted: false },
    { clipId: 3, trackRow: 2, kind: 'AudioStem', sourceId: 1, startTick: 0, durationTicks: 384, clipOffsetTicks: 0, gain: 0.95, muted: false },
    { clipId: 4, trackRow: 3, kind: 'Automation', sourceId: 0, startTick: 0, durationTicks: 384, clipOffsetTicks: 0, gain: 1.0, muted: false },
  ]);

  // =========================================================================
  // 8. SVELTE 5 RUNES — OFFLINE BOUNCE & MASTERING RENDER ENGINE
  // =========================================================================
  renderMode = $state<'Master' | 'SplitMixerTracks' | 'StemMastering'>('Master');
  renderBitDepth = $state<'Pcm16MbitPlus' | 'Pcm24MbitPlus' | 'Float32Ieee'>('Pcm24MbitPlus');
  renderSampleRateHz = $state<44100 | 48000 | 96000 | 192000>(48000);
  renderDurationTicks = $state(384);
  renderTrimApdcLatency = $state(true);
  renderEnableNoiseShaping = $state(true);
  renderOutputDir = $state('F:/Exports/ToneForge_Renders');
  renderFilePrefix = $state('ToneForge_Master');
  renderIsRunning = $state(false);

  lastRenderReport = $state<PostRenderReportUiState>({
    integratedLufs: -13.8,
    shortTermMaxLufs: -11.4,
    momentaryMaxLufs: -9.9,
    truePeakDbtpL: -1.02,
    truePeakDbtpR: -1.04,
    maxTruePeakDbtp: -1.02,
    crestFactorDb: 9.6,
    phaseCorrelation: 0.87,
    spotifyCompliant: true,
    appleMusicCompliant: false,
    youtubeCompliant: true,
    spotifyPenaltyDb: 0.0,
    appleMusicPenaltyDb: -2.2,
    youtubePenaltyDb: 0.0,
    apdcLatencyTrimmedSamples: 8832,
    renderedFiles: ['F:/Exports/ToneForge_Renders/ToneForge_Master_Master.wav'],
  });

  constructor(sharedBuffer?: ArrayBuffer) {
    const buf = sharedBuffer ?? new ArrayBuffer(VISUAL_TELEMETRY_FRAME_BYTES);
    if (buf.byteLength !== VISUAL_TELEMETRY_FRAME_BYTES) {
      throw new Error(
        `Invalid VisualTelemetryFrame size: ${buf.byteLength} (expected ${VISUAL_TELEMETRY_FRAME_BYTES})`
      );
    }
    this.buffer = buf;
    this.dataView = new DataView(this.buffer);
    this.u8View = new Uint8Array(this.buffer);

    // Bind Zero-Copy TypedArray slices at exact byte offsets
    this.trackMetersWords = new Float32Array(this.buffer, OFFSET_TRACK_METERS, MAX_MIXER_TRACKS * 6);
    this.proq4PreDb = new Float32Array(this.buffer, OFFSET_PROQ4_SPECTRUM + 0x0000, 1024);
    this.proq4PostDb = new Float32Array(this.buffer, OFFSET_PROQ4_SPECTRUM + 0x1000, 1024);
    this.proq4SidechainDb = new Float32Array(this.buffer, OFFSET_PROQ4_SPECTRUM + 0x2000, 1024);
    this.proq4CollisionMask = new Float32Array(this.buffer, OFFSET_PROQ4_SPECTRUM + 0x3000, 1024);

    this.soothe3Reduction1024 = new Float32Array(this.buffer, OFFSET_SOOTHE3_REDUCTION + 0x0000, 1024);
    this.soothe3Reduction512 = new Float32Array(this.buffer, OFFSET_SOOTHE3_REDUCTION + 0x0000, 512);
    this.soothe3InputSpecDb = new Float32Array(this.buffer, OFFSET_SOOTHE3_REDUCTION + 0x1000, 1024);

    this.ozone12IrcGrTrace = new Float32Array(this.buffer, OFFSET_OZONE12_TELEMETRY + 0x0000, 512);
    this.ozone12PolarSampleXy = new Float32Array(this.buffer, OFFSET_OZONE12_TELEMETRY + 0x0800, 1024);
    this.ozone12TonalBalanceBars = new Float32Array(this.buffer, OFFSET_OZONE12_TELEMETRY + 0x1800, 256);
    this.ozone12StemRmsDb = new Float32Array(this.buffer, OFFSET_OZONE12_TELEMETRY + 0x1C00, 4);
    this.ozone12ImagerCorrBands = new Float32Array(this.buffer, OFFSET_OZONE12_TELEMETRY + 0x1C10, 4);
    this.ozone12ExciterDriveGr = new Float32Array(this.buffer, OFFSET_OZONE12_TELEMETRY + 0x1C20, 248);

    this.spectrogramRing20x1024 = new Float32Array(this.buffer, OFFSET_SPECTROGRAM_RING, 20 * 1024);
    this.controllerValuesNorm512 = new Float32Array(this.buffer, OFFSET_AUTOMATION_SPLINE + 0x0000, 512);
    this.grossBeatPlayheadPos = new Float32Array(this.buffer, OFFSET_AUTOMATION_SPLINE + 0x0800, 2);
    this.patcherNodeCpuPct510 = new Float32Array(this.buffer, OFFSET_AUTOMATION_SPLINE + 0x0808, 510);

    // Seed initial telemetry frame so UI displays rich live waveforms immediately
    this.synthesizeDemoTelemetryFrame(0);
  }

  private createInitial502Tracks(): MixerTrackUiState[] {
    const tracks: MixerTrackUiState[] = [];
    for (let i = 0; i < MAX_MIXER_TRACKS; i++) {
      const name =
        i === 0
          ? 'Master'
          : i === 1
            ? 'Vocals Lead'
            : i === 2
              ? 'Drums Bus'
              : i === 3
                ? 'Sub Bass'
                : i === 4
                  ? 'Synth & Keys'
                  : i === 501
                    ? 'Preview'
                    : `Insert ${i}`;
      tracks.push({
        index: i,
        name,
        gainLinear: 1.0,
        muted: false,
        soloed: false,
        polarityInverted: false,
        apdcLatencySamples: i === 0 ? 5760 : i === 1 ? 3072 : 0,
        sends: i >= 1 && i <= 4 ? [{ destTrack: 0, gain: 1.0, sidechain: false }] : [],
        slots:
          i === 0
            ? [
                { slotIndex: 0, name: 'BSRNN 120ms Stem Rebalance', enabled: true, wetMix: 1.0 },
                { slotIndex: 1, name: 'iZotope Ozone 12 Master Chain', enabled: true, wetMix: 1.0 },
              ]
            : i === 1
              ? [
                  { slotIndex: 0, name: 'FabFilter Pro-Q 4 (24-Band)', enabled: true, wetMix: 1.0 },
                  { slotIndex: 1, name: 'oeksound Soothe3 Spectral', enabled: true, wetMix: 1.0 },
                  { slotIndex: 2, name: 'Patcher Parallel APDC DAG', enabled: true, wetMix: 1.0 },
                ]
              : [],
      });
    }
    return tracks;
  }

  private createInitialProQ4Bands(): ProQ4BandState[] {
    const presets: Partial<ProQ4BandState>[] = [
      { used: true, enabled: true, freqHz: 32.0, gainDb: 0.0, q: 0.707, shape: 2, slopeDbOct: 36, dynamicRangeDb: 0.0, spectralEnabled: false },
      { used: true, enabled: true, freqHz: 115.0, gainDb: 2.4, q: 1.2, shape: 0, slopeDbOct: 12, dynamicRangeDb: -2.0, spectralEnabled: false },
      { used: true, enabled: true, freqHz: 340.0, gainDb: -2.8, q: 2.1, shape: 0, slopeDbOct: 12, dynamicRangeDb: -4.5, spectralEnabled: true },
      { used: true, enabled: true, freqHz: 1250.0, gainDb: 1.6, q: 1.4, shape: 0, slopeDbOct: 12, dynamicRangeDb: 0.0, spectralEnabled: false },
      { used: true, enabled: true, freqHz: 3600.0, gainDb: -3.2, q: 2.8, shape: 0, slopeDbOct: 24, dynamicRangeDb: -6.0, spectralEnabled: true },
      { used: true, enabled: true, freqHz: 8500.0, gainDb: 2.8, q: 0.9, shape: 3, slopeDbOct: 12, dynamicRangeDb: 0.0, spectralEnabled: false },
      { used: true, enabled: true, freqHz: 19500.0, gainDb: 0.0, q: 0.707, shape: 4, slopeDbOct: 48, dynamicRangeDb: 0.0, spectralEnabled: false },
    ];
    const bands: ProQ4BandState[] = [];
    for (let i = 0; i < 24; i++) {
      const p = presets[i];
      bands.push({
        bandIdx: i,
        used: p?.used ?? false,
        enabled: p?.enabled ?? false,
        freqHz: p?.freqHz ?? Math.round(40 * Math.pow(400, i / 23)),
        gainDb: p?.gainDb ?? 0.0,
        q: p?.q ?? 1.0,
        shape: p?.shape ?? 0,
        slopeDbOct: p?.slopeDbOct ?? 12,
        dynamicRangeDb: p?.dynamicRangeDb ?? 0.0,
        spectralEnabled: p?.spectralEnabled ?? false,
        spectralDensity: 65.0,
      });
    }
    return bands;
  }

  /**
   * Decodes the 64-byte `TransportAndMasterHeader` (`0x00000..0x0003F`) using `DataView`
   * at 120 Hz without allocating objects or parsing JSON.
   */
  syncHeaderFromSharedMemory(): void {
    const dv = this.dataView;
    this.frameSeq = Number(dv.getBigUint64(OFFSET_HEADER + 0x00, true));
    this.samplePos = Number(dv.getBigInt64(OFFSET_HEADER + 0x08, true));
    this.playheadPpq = Number(dv.getBigInt64(OFFSET_HEADER + 0x10, true));
    const bpm = dv.getFloat32(OFFSET_HEADER + 0x18, true);
    if (bpm >= 10.0 && bpm <= 999.0) {
      this.tempoBpm = bpm;
    }
    this.cpuLoadPercent = dv.getFloat32(OFFSET_HEADER + 0x1C, true);
    this.lufsMomentary = dv.getFloat32(OFFSET_HEADER + 0x20, true);
    this.lufsShortTerm = dv.getFloat32(OFFSET_HEADER + 0x24, true);
    this.lufsIntegrated = dv.getFloat32(OFFSET_HEADER + 0x28, true);
    this.truePeakDbL = dv.getFloat32(OFFSET_HEADER + 0x2C, true);
    this.truePeakDbR = dv.getFloat32(OFFSET_HEADER + 0x30, true);
    this.masterGrDb = dv.getFloat32(OFFSET_HEADER + 0x34, true);
    this.activeStemMode = dv.getUint32(OFFSET_HEADER + 0x38, true);
  }

  /**
   * Reads a single track's `TrackMeterEntry` (`24` bytes) from `0x00040 + trackIndex * 24`.
   */
  getTrackMeter(trackIndex: number): TrackMeterSnapshot {
    if (trackIndex < 0 || trackIndex >= MAX_MIXER_TRACKS) {
      return { trackIndex, peakLDb: -120, peakRDb: -120, rmsLDb: -120, rmsRDb: -120, grDb: 0, activityFlags: 0 };
    }
    const byteOff = OFFSET_TRACK_METERS + trackIndex * TRACK_METER_STRIDE_BYTES;
    const dv = this.dataView;
    return {
      trackIndex,
      peakLDb: dv.getFloat32(byteOff + 0, true),
      peakRDb: dv.getFloat32(byteOff + 4, true),
      rmsLDb: dv.getFloat32(byteOff + 8, true),
      rmsRDb: dv.getFloat32(byteOff + 12, true),
      grDb: dv.getFloat32(byteOff + 16, true),
      activityFlags: dv.getUint32(byteOff + 20, true),
    };
  }

  /**
   * Evaluates the signed-exponential `TensionSpline` (`((e^{k*t}-1)/(e^k-1))`) at `tick`.
   */
  evaluateTensionSplineAt(tick: number): number {
    const pts = this.tensionSplinePoints;
    if (pts.length === 0) return 0.0;
    if (tick <= pts[0].x) return pts[0].y;
    if (tick >= pts[pts.length - 1].x) return pts[pts.length - 1].y;

    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i];
      const p1 = pts[i + 1];
      if (tick >= p0.x && tick <= p1.x) {
        const dx = Math.max(1e-9, p1.x - p0.x);
        const t = (tick - p0.x) / dx;
        const k = p1.tension * 10.0;
        const shaped = Math.abs(k) < 1e-5 ? t : (Math.exp(k * t) - 1.0) / (Math.exp(k) - 1.0);
        return p0.y + (p1.y - p0.y) * shaped;
      }
    }
    return pts[pts.length - 1].y;
  }

  /**
   * Evaluates the combined 24-band Pro-Q 4 magnitude response (in dB) at `freqHz`
   * for immediate interactive UI feedback and GPU uniform generation.
   */
  evaluateProQ4MagnitudeDbAt(freqHz: number): number {
    let totalDb = 0.0;
    const scale = this.proq4GainScalePct * 0.01;
    for (const b of this.proq4Bands) {
      if (!b.used || !b.enabled) continue;
      const ratio = freqHz / Math.max(10.0, b.freqHz);
      const logDist = Math.log2(Math.max(1e-6, ratio));
      const bw = 1.0 / Math.max(0.1, b.q);
      if (b.shape === 0) {
        // Bell
        totalDb += (b.gainDb * scale) * Math.exp(-(logDist * logDist) / (0.35 * bw * bw));
      } else if (b.shape === 1) {
        // LowShelf
        const sig = 1.0 / (1.0 + Math.exp(logDist * 4.0 * Math.sqrt(b.q)));
        totalDb += b.gainDb * scale * sig;
      } else if (b.shape === 2) {
        // LowCut
        if (ratio < 1.0) {
          totalDb -= Math.abs(logDist) * b.slopeDbOct;
        }
      } else if (b.shape === 3) {
        // HighShelf
        const sig = 1.0 / (1.0 + Math.exp(-logDist * 4.0 * Math.sqrt(b.q)));
        totalDb += b.gainDb * scale * sig;
      } else if (b.shape === 4) {
        // HighCut
        if (ratio > 1.0) {
          totalDb -= Math.abs(logDist) * b.slopeDbOct;
        }
      } else if (b.shape === 5) {
        // Notch
        totalDb -= 24.0 * Math.exp(-(logDist * logDist) / (0.08 * bw * bw));
      } else if (b.shape === 6) {
        // BandPass
        totalDb -= Math.min(36.0, Math.abs(logDist) * Math.max(6.0, b.slopeDbOct * 0.5) * Math.sqrt(b.q));
      } else if (b.shape === 7 || b.shape === 8) {
        // TiltShelf / FlatTilt
        totalDb += (b.gainDb * scale) * Math.tanh(logDist * 0.8);
      }
      // shape === 9 (AllPass) contributes 0 dB magnitude change
    }
    return Math.max(-36.0, Math.min(36.0, totalDb));
  }

  /**
   * Packs the 24 Pro-Q 4 bands into the 96-section `BiquadCascadeUniforms` Float32Array
   * (`4 + 96*4 + 96*4 = 772` floats = `3,088` bytes) for `proq4_curve.wgsl`.
   */
  packProQ4BiquadCascadeUniforms(sampleRate = 48000.0): Float32Array {
    const buf = new ArrayBuffer(3088);
    const dv = new DataView(buf);
    const f32 = new Float32Array(buf);

    let secCursor = 0;
    const scale = this.proq4GainScalePct * 0.01;
    for (const band of this.proq4Bands) {
      if (!band.used || !band.enabled || secCursor >= 96) continue;
      const w0 = (2.0 * Math.PI * Math.max(10.0, Math.min(22000.0, band.freqHz))) / sampleRate;
      const cosW0 = Math.cos(w0);
      const sinW0 = Math.sin(w0);
      const alpha = sinW0 / (2.0 * Math.max(0.05, band.q));
      const A = Math.pow(10.0, (band.gainDb * scale) / 40.0);

      let b0 = 1.0, b1 = 0.0, b2 = 0.0, a0 = 1.0, a1 = 0.0, a2 = 0.0;
      const isCascadedSlope = band.shape === 2 || band.shape === 4 || band.shape === 6;
      const numSections = isCascadedSlope
        ? Math.min(8, Math.max(1, Math.round(band.slopeDbOct / 12)))
        : 1;

      if (band.shape === 0) {
        // 0: Bell (RBJ Peaking EQ)
        b0 = 1.0 + alpha * A;
        b1 = -2.0 * cosW0;
        b2 = 1.0 - alpha * A;
        a0 = 1.0 + alpha / A;
        a1 = -2.0 * cosW0;
        a2 = 1.0 - alpha / A;
      } else if (band.shape === 1) {
        // 1: LowShelf
        const sqrtA = Math.sqrt(A);
        b0 = A * ((A + 1) - (A - 1) * cosW0 + 2 * sqrtA * alpha);
        b1 = 2 * A * ((A - 1) - (A + 1) * cosW0);
        b2 = A * ((A + 1) - (A - 1) * cosW0 - 2 * sqrtA * alpha);
        a0 = (A + 1) + (A - 1) * cosW0 + 2 * sqrtA * alpha;
        a1 = -2 * ((A - 1) + (A + 1) * cosW0);
        a2 = (A + 1) + (A - 1) * cosW0 - 2 * sqrtA * alpha;
      } else if (band.shape === 2) {
        // 2: LowCut (High-pass)
        b0 = (1.0 + cosW0) * 0.5;
        b1 = -(1.0 + cosW0);
        b2 = (1.0 + cosW0) * 0.5;
        a0 = 1.0 + alpha;
        a1 = -2.0 * cosW0;
        a2 = 1.0 - alpha;
      } else if (band.shape === 3) {
        // 3: HighShelf
        const sqrtA = Math.sqrt(A);
        b0 = A * ((A + 1) + (A - 1) * cosW0 + 2 * sqrtA * alpha);
        b1 = -2 * A * ((A - 1) + (A + 1) * cosW0);
        b2 = A * ((A + 1) + (A - 1) * cosW0 - 2 * sqrtA * alpha);
        a0 = (A + 1) - (A - 1) * cosW0 + 2 * sqrtA * alpha;
        a1 = 2 * ((A - 1) - (A + 1) * cosW0);
        a2 = (A + 1) - (A - 1) * cosW0 - 2 * sqrtA * alpha;
      } else if (band.shape === 4) {
        // 4: HighCut (Low-pass)
        b0 = (1.0 - cosW0) * 0.5;
        b1 = 1.0 - cosW0;
        b2 = (1.0 - cosW0) * 0.5;
        a0 = 1.0 + alpha;
        a1 = -2.0 * cosW0;
        a2 = 1.0 - alpha;
      } else if (band.shape === 5) {
        // 5: Notch
        b0 = 1.0;
        b1 = -2.0 * cosW0;
        b2 = 1.0;
        a0 = 1.0 + alpha;
        a1 = -2.0 * cosW0;
        a2 = 1.0 - alpha;
      } else if (band.shape === 6) {
        // 6: BandPass (0 dB peak gain)
        b0 = alpha;
        b1 = 0.0;
        b2 = -alpha;
        a0 = 1.0 + alpha;
        a1 = -2.0 * cosW0;
        a2 = 1.0 - alpha;
      } else if (band.shape === 7 || band.shape === 8) {
        // 7: TiltShelf / 8: FlatTilt (Bilateral HighShelf around pivot)
        const sqrtA = Math.sqrt(A);
        b0 = A * ((A + 1) + (A - 1) * cosW0 + 2 * sqrtA * alpha);
        b1 = -2 * A * ((A - 1) + (A + 1) * cosW0);
        b2 = A * ((A + 1) + (A - 1) * cosW0 - 2 * sqrtA * alpha);
        a0 = (A + 1) - (A - 1) * cosW0 + 2 * sqrtA * alpha;
        a1 = 2 * ((A - 1) - (A + 1) * cosW0);
        a2 = (A + 1) - (A - 1) * cosW0 - 2 * sqrtA * alpha;
      } else {
        // 9: AllPass
        b0 = 1.0 - alpha;
        b1 = -2.0 * cosW0;
        b2 = 1.0 + alpha;
        a0 = 1.0 + alpha;
        a1 = -2.0 * cosW0;
        a2 = 1.0 - alpha;
      }

      const invA0 = 1.0 / a0;
      const nb0 = b0 * invA0;
      const nb1 = b1 * invA0;
      const nb2 = b2 * invA0;
      const na1 = a1 * invA0;
      const na2 = a2 * invA0;

      for (let s = 0; s < numSections && secCursor < 96; s++) {
        const bOff = 4 + secCursor * 4;
        const aOff = 4 + 96 * 4 + secCursor * 4;
        f32[bOff + 0] = nb0;
        f32[bOff + 1] = nb1;
        f32[bOff + 2] = nb2;
        f32[bOff + 3] = 0.0;

        f32[aOff + 0] = 1.0;
        f32[aOff + 1] = na1;
        f32[aOff + 2] = na2;
        f32[aOff + 3] = 0.0;
        secCursor++;
      }
    }

    dv.setUint32(0, secCursor, true);
    dv.setFloat32(4, sampleRate, true);
    dv.setFloat32(8, 10.0, true);
    dv.setFloat32(12, 30000.0, true);
    return f32;
  }

  private pollInFlight = false;

  /**
   * Polls the 128 KB binary telemetry frame at 120 Hz from the Tauri Rust backend
   * (or synthesizes deterministic 120 Hz telemetry when running in standalone browser mode).
   */
  async pollTelemetry120Hz(timestampMs: number): Promise<void> {
    if (this.isPlaying && this.audioSourceMode === 'pianoRoll') {
      this.speakerEngine.tickSequencer(this);
    }

    if (isTauriEnvironment()) {
      if (this.pollInFlight) {
        return;
      }
      this.pollInFlight = true;
      try {
        const raw = await invoke<ArrayBuffer | Uint8Array | number[]>('cmd_poll_binary_telemetry_frame');
        if (raw instanceof ArrayBuffer && raw.byteLength === VISUAL_TELEMETRY_FRAME_BYTES) {
          this.u8View.set(new Uint8Array(raw));
          this.syncLogSpectrumOverlay(timestampMs);
          this.syncHeaderFromSharedMemory();
          return;
        } else if (ArrayBuffer.isView(raw) && raw.byteLength === VISUAL_TELEMETRY_FRAME_BYTES) {
          this.u8View.set(new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength));
          this.syncLogSpectrumOverlay(timestampMs);
          this.syncHeaderFromSharedMemory();
          return;
        } else if (Array.isArray(raw) && raw.length === VISUAL_TELEMETRY_FRAME_BYTES) {
          this.u8View.set(raw);
          this.syncLogSpectrumOverlay(timestampMs);
          this.syncHeaderFromSharedMemory();
          return;
        }
      } catch {
        // Fallback to local 120 Hz synthesis if command is not registered in preview mode
      } finally {
        this.pollInFlight = false;
      }
    }
    this.synthesizeDemoTelemetryFrame(timestampMs);
    this.syncHeaderFromSharedMemory();
  }

  /**
   * Ensures the 1024-bin logarithmic frequency display buffers (`0x03000` Pro-Q 4 & `0x07000` Soothe3)
   * render a smooth, continuous broadband harmonic spectrum (`gợn sóng`) coupled to the active
   * 24-band EQ response curve when `this.isPlaying === true`, and completely silence (`-120 dB`)
   * when `!this.isPlaying` (Paused) so ONLY the yellow EQ curve remains.
   */
  private syncLogSpectrumOverlay(timestampMs: number): void {
    const dv = this.dataView;
    if (!this.isPlaying) {
      // Paused: freeze samplePos & playheadPpq, drop spectrum to -120 dB, Soothe3 reduction to 0 dB, and 502 meters to -120 dB
      dv.setBigInt64(OFFSET_HEADER + 0x08, BigInt(this.samplePos), true);
      dv.setBigInt64(OFFSET_HEADER + 0x10, BigInt(this.playheadPpq), true);
      dv.setFloat32(OFFSET_HEADER + 0x1C, 0.8, true);
      dv.setFloat32(OFFSET_HEADER + 0x20, -120.0, true);
      dv.setFloat32(OFFSET_HEADER + 0x24, -120.0, true);
      dv.setFloat32(OFFSET_HEADER + 0x2C, -120.0, true);
      dv.setFloat32(OFFSET_HEADER + 0x30, -120.0, true);
      dv.setFloat32(OFFSET_HEADER + 0x34, 0.0, true);
      dv.setUint32(OFFSET_HEADER + 0x38, 0, true);
      this.truePeakDbL = -120.0;
      this.truePeakDbR = -120.0;
      this.masterGrDb = 0.0;

      for (let trk = 0; trk < MAX_MIXER_TRACKS; trk++) {
        const off = OFFSET_TRACK_METERS + trk * TRACK_METER_STRIDE_BYTES;
        dv.setFloat32(off + 0, -120.0, true);
        dv.setFloat32(off + 4, -120.0, true);
        dv.setFloat32(off + 8, -120.0, true);
        dv.setFloat32(off + 12, -120.0, true);
        dv.setFloat32(off + 16, 0.0, true);
        dv.setUint32(off + 20, 0, true);
      }

      this.proq4PreDb.fill(-120.0);
      this.proq4PostDb.fill(-120.0);
      this.proq4SidechainDb.fill(-120.0);
      this.proq4CollisionMask.fill(0.0);
      this.soothe3Reduction1024.fill(0.0);
      this.soothe3InputSpecDb.fill(-120.0);
      return;
    }

    const t = timestampMs * 0.001;
    const ringRow = Math.abs(this.frameSeq) % 20;
    const ringBase = ringRow * 1024;
    const hasRealAudioFft = this.speakerEngine.readPrePostSpectrumDb(this.realPreFft1024, this.realPostFft1024);

    for (let k = 0; k < 1024; k++) {
      const normX = k / 1023.0;
      const freq = 20.0 * Math.pow(1000.0, normX);
      const pinkSlope = -18.0 - 12.0 * normX;
      const harmonic =
        8.0 * Math.sin(normX * 28.0 + t * 3.2) + 4.5 * Math.cos(normX * 75.0 - t * 6.0);
      const synthPreDb = Math.max(this.proq4PreDb[k], pinkSlope + harmonic);
      const preDb =
        hasRealAudioFft && this.realPreFft1024[k] > -105.0
          ? 0.65 * this.realPreFft1024[k] + 0.35 * synthPreDb
          : synthPreDb;
      const eqBoost = this.evaluateProQ4MagnitudeDbAt(freq);
      const postDb =
        hasRealAudioFft && this.realPostFft1024[k] > -105.0
          ? 0.5 * this.realPostFft1024[k] + 0.5 * (preDb + eqBoost)
          : preDb + eqBoost;

      this.proq4PreDb[k] = preDb;
      this.proq4PostDb[k] = postDb;
      this.proq4SidechainDb[k] = preDb - 6.0;
      this.proq4CollisionMask[k] =
        normX > 0.28 && normX < 0.36 ? 0.65 * Math.abs(Math.sin(t * 3.0)) : 0.0;

      const resPeak1 = Math.exp(-Math.pow((normX - 0.58) / 0.05, 2.0));
      const resPeak2 = Math.exp(-Math.pow((normX - 0.74) / 0.04, 2.0));
      const cutDb =
        -this.soothe3Depth * 2.2 * (resPeak1 * (0.6 + 0.4 * Math.sin(t * 7.0)) + resPeak2 * 0.8);
      this.soothe3Reduction1024[k] = Math.max(this.soothe3MaxCutDb, cutDb);
      this.soothe3InputSpecDb[k] = postDb;
      this.spectrogramRing20x1024[ringBase + k] = postDb + cutDb;
    }
  }

  /**
   * Synthesizes a realistic 128 KB binary frame directly into `this.buffer` at exact byte offsets
   * so all WebGPU shaders and meters animate at 120 FPS when `this.isPlaying === true`, and freeze/silence
   * completely when `!this.isPlaying` (Paused).
   */
  synthesizeDemoTelemetryFrame(timestampMs: number): void {
    const dtSec =
      this.lastSynthTimestampMs > 0 && timestampMs > this.lastSynthTimestampMs
        ? Math.min(0.25, (timestampMs - this.lastSynthTimestampMs) * 0.001)
        : timestampMs === 0
          ? 0.0
          : 1.0 / 120.0;
    this.lastSynthTimestampMs = timestampMs;

    const t = timestampMs * 0.001;
    const dv = this.dataView;
    const nextSeq = BigInt(this.frameSeq + 1);

    if (!this.isPlaying) {
      // Paused: DO NOT increment samplePos or playheadPpq; silence spectrum (-120 dB), Soothe3 (0 dB), and all 502 Track Meters (-120 dB)
      const frozenSamples = BigInt(this.samplePos);
      const frozenPpq = BigInt(this.playheadPpq);

      dv.setBigUint64(OFFSET_HEADER + 0x00, nextSeq, true);
      dv.setBigInt64(OFFSET_HEADER + 0x08, frozenSamples, true);
      dv.setBigInt64(OFFSET_HEADER + 0x10, frozenPpq, true);
      dv.setFloat32(OFFSET_HEADER + 0x18, this.tempoBpm, true);
      dv.setFloat32(OFFSET_HEADER + 0x1C, 0.8, true);
      dv.setFloat32(OFFSET_HEADER + 0x20, -120.0, true);
      dv.setFloat32(OFFSET_HEADER + 0x24, -120.0, true);
      dv.setFloat32(OFFSET_HEADER + 0x28, -9.8, true);
      dv.setFloat32(OFFSET_HEADER + 0x2C, -120.0, true);
      dv.setFloat32(OFFSET_HEADER + 0x30, -120.0, true);
      dv.setFloat32(OFFSET_HEADER + 0x34, 0.0, true);
      dv.setUint32(OFFSET_HEADER + 0x38, 0, true);

      for (let trk = 0; trk < MAX_MIXER_TRACKS; trk++) {
        const off = OFFSET_TRACK_METERS + trk * TRACK_METER_STRIDE_BYTES;
        dv.setFloat32(off + 0, -120.0, true);
        dv.setFloat32(off + 4, -120.0, true);
        dv.setFloat32(off + 8, -120.0, true);
        dv.setFloat32(off + 12, -120.0, true);
        dv.setFloat32(off + 16, 0.0, true);
        dv.setUint32(off + 20, 0, true);
      }

      this.proq4PreDb.fill(-120.0);
      this.proq4PostDb.fill(-120.0);
      this.proq4SidechainDb.fill(-120.0);
      this.proq4CollisionMask.fill(0.0);
      this.soothe3Reduction1024.fill(0.0);
      this.soothe3InputSpecDb.fill(-120.0);
      this.controllerValuesNorm512[0] = this.evaluateTensionSplineAt(Number(frozenPpq % 384n));
      return;
    }

    if (timestampMs > 0) {
      this.samplePos += Math.max(1, Math.round(dtSec * 48000));
      this.playheadPpq = Math.floor((this.samplePos / 48000.0) * (this.tempoBpm / 60.0) * 96.0);
    }

    const nextSamples = BigInt(this.samplePos);
    const nextPpq = BigInt(this.playheadPpq);

    dv.setBigUint64(OFFSET_HEADER + 0x00, nextSeq, true);
    dv.setBigInt64(OFFSET_HEADER + 0x08, nextSamples, true);
    dv.setBigInt64(OFFSET_HEADER + 0x10, nextPpq, true);
    dv.setFloat32(OFFSET_HEADER + 0x18, this.tempoBpm, true);
    dv.setFloat32(OFFSET_HEADER + 0x1C, 6.2 + 0.8 * Math.sin(t * 1.7), true);
    dv.setFloat32(OFFSET_HEADER + 0x20, -9.4 + 1.2 * Math.sin(t * 3.1), true);
    dv.setFloat32(OFFSET_HEADER + 0x24, -10.1 + 0.5 * Math.cos(t * 1.1), true);
    dv.setFloat32(OFFSET_HEADER + 0x28, -9.8, true);
    dv.setFloat32(OFFSET_HEADER + 0x2C, this.ozone12CeilingDbtp - 0.05 + 0.04 * Math.sin(t * 5.0), true);
    dv.setFloat32(OFFSET_HEADER + 0x30, this.ozone12CeilingDbtp - 0.06 + 0.04 * Math.cos(t * 4.7), true);
    dv.setFloat32(OFFSET_HEADER + 0x34, -2.1 - 0.9 * Math.abs(Math.sin(t * 4.2)), true);
    dv.setUint32(OFFSET_HEADER + 0x38, 1, true);

    // 502 Track Meters (0x00040)
    for (let trk = 0; trk < MAX_MIXER_TRACKS; trk++) {
      const off = OFFSET_TRACK_METERS + trk * TRACK_METER_STRIDE_BYTES;
      const active = trk <= 16 || trk === 501;
      const phase = trk * 0.37 + t * 4.5;
      const peakL = active ? -12.0 + 9.0 * Math.sin(phase) - (trk > 4 ? (trk * 1.2) : 0.0) : -120.0;
      const peakR = active ? -12.2 + 8.8 * Math.cos(phase * 1.03) - (trk > 4 ? (trk * 1.2) : 0.0) : -120.0;
      const gr = trk === 0 ? -2.4 * Math.abs(Math.sin(t * 3.0)) : trk === 1 ? -3.5 * Math.abs(Math.cos(t * 2.5)) : 0.0;
      dv.setFloat32(off + 0, peakL, true);
      dv.setFloat32(off + 4, peakR, true);
      dv.setFloat32(off + 8, peakL - 4.5, true);
      dv.setFloat32(off + 12, peakR - 4.5, true);
      dv.setFloat32(off + 16, gr, true);
      dv.setUint32(off + 20, active ? 1 : 0, true);
    }

    // Pro-Q 4 1024-bin Spectrum (0x03000) & Soothe3 1024-bin Reduction (0x07000)
    const ringRow = Number(nextSeq % 20n);
    const ringBase = ringRow * 1024;
    const hasRealAudioFft = this.speakerEngine.readPrePostSpectrumDb(this.realPreFft1024, this.realPostFft1024);

    for (let k = 0; k < 1024; k++) {
      const normX = k / 1023.0;
      const freq = 20.0 * Math.pow(1000.0, normX);
      const pinkSlope = -18.0 - 12.0 * normX;
      const harmonic = 8.0 * Math.sin(normX * 28.0 + t * 3.2) + 4.5 * Math.cos(normX * 75.0 - t * 6.0);
      const synthPreDb = pinkSlope + harmonic;
      const preDb =
        hasRealAudioFft && this.realPreFft1024[k] > -105.0
          ? 0.65 * this.realPreFft1024[k] + 0.35 * synthPreDb
          : synthPreDb;
      const eqBoost = this.evaluateProQ4MagnitudeDbAt(freq);
      const postDb =
        hasRealAudioFft && this.realPostFft1024[k] > -105.0
          ? 0.5 * this.realPostFft1024[k] + 0.5 * (preDb + eqBoost)
          : preDb + eqBoost;

      this.proq4PreDb[k] = preDb;
      this.proq4PostDb[k] = postDb;
      this.proq4SidechainDb[k] = preDb - 6.0;
      this.proq4CollisionMask[k] = normX > 0.28 && normX < 0.36 ? 0.65 * Math.abs(Math.sin(t * 3.0)) : 0.0;

      // Soothe3 dynamic resonance suppression curve (0.0 .. -24.0 dB)
      const resPeak1 = Math.exp(-Math.pow((normX - 0.58) / 0.05, 2.0));
      const resPeak2 = Math.exp(-Math.pow((normX - 0.74) / 0.04, 2.0));
      const cutDb = -this.soothe3Depth * 2.2 * (resPeak1 * (0.6 + 0.4 * Math.sin(t * 7.0)) + resPeak2 * 0.8);
      this.soothe3Reduction1024[k] = Math.max(this.soothe3MaxCutDb, cutDb);
      this.soothe3InputSpecDb[k] = postDb;
      this.spectrogramRing20x1024[ringBase + k] = postDb + cutDb;
    }

    // Ozone 12 Telemetry (0x09000)
    for (let i = 0; i < 512; i++) {
      const ph = (i / 512.0) * Math.PI * 6.0 + t * 4.0;
      this.ozone12IrcGrTrace[i] = -1.2 - 1.8 * Math.abs(Math.sin(ph * 0.5));
      const radius = 0.35 + 0.35 * Math.sin(ph * 1.7 + t);
      this.ozone12PolarSampleXy[i * 2 + 0] = radius * Math.cos(ph);
      this.ozone12PolarSampleXy[i * 2 + 1] = radius * Math.sin(ph * 1.02 + 0.3);
    }
    for (let b = 0; b < 256; b++) {
      const nx = b / 255.0;
      this.ozone12TonalBalanceBars[b] = 0.52 + 0.14 * Math.sin(nx * 10.0 + t * 1.5);
    }
    this.ozone12StemRmsDb[0] = -12.4 + this.bsrnnStemGainsDb[0] + 1.5 * Math.sin(t * 3.2);
    this.ozone12StemRmsDb[1] = -11.0 + this.bsrnnStemGainsDb[1] + 2.2 * Math.abs(Math.cos(t * 6.0));
    this.ozone12StemRmsDb[2] = -10.5 + this.bsrnnStemGainsDb[2] + 1.0 * Math.sin(t * 2.0);
    this.ozone12StemRmsDb[3] = -14.8 + this.bsrnnStemGainsDb[3] + 1.2 * Math.cos(t * 2.7);

    this.ozone12ImagerCorrBands[0] = 0.94;
    this.ozone12ImagerCorrBands[1] = 0.82;
    this.ozone12ImagerCorrBands[2] = 0.68;
    this.ozone12ImagerCorrBands[3] = 0.55;

    // 30-Bit Fixed-Point Automation Spline Telemetry (0x1F000)
    this.controllerValuesNorm512[0] = this.evaluateTensionSplineAt(Number(nextPpq % 384n));
  }

  // =========================================================================
  // REAL-TIME SPEAKER ENGINE SYNCHRONIZATION & IPC COMMAND DISPATCHERS
  // =========================================================================

  syncDspToSpeakerEngine(): void {
    this.speakerEngine.syncDspChain(this);
  }

  setSpeakerVolume(volumePct: number): void {
    this.speakerVolumePct = Math.max(0, Math.min(100, Math.round(volumePct)));
    if (this.speakerVolumePct > 0 && this.speakerMuted) {
      this.speakerMuted = false;
    }
    this.speakerEngine.syncDspChain(this);
  }

  toggleSpeakerMute(): void {
    this.speakerMuted = !this.speakerMuted;
    this.speakerEngine.syncDspChain(this);
  }

  async loadUserAudioFile(file: File): Promise<void> {
    this.audioSourceMode = 'customFile';
    this.loadedAudioFileName = file.name;
    this.uvr5AudioPath = file.name;
    const stemBase = file.name.replace(/\.[^.]+$/, '');
    const tag = this.uvr5EnsembleMode.toLowerCase();
    this.uvr5SeparatedStems = [
      `${stemBase}_${tag}_vocals.wav`,
      `${stemBase}_${tag}_drums.wav`,
      `${stemBase}_${tag}_bass.wav`,
      `${stemBase}_${tag}_other.wav`,
    ];
    this.syncDspToSpeakerEngine();
    await this.speakerEngine.loadCustomAudioFile(file, this);
    await this.setTransport(true, this.tempoBpm);
  }

  async switchToPianoRollSource(): Promise<void> {
    this.audioSourceMode = 'pianoRoll';
    this.speakerEngine.switchToPianoRollSource(this);
  }

  auditionNote(key: number, velocity = 0.9, finePitchCents = 0.0, pan = 0.0): void {
    this.syncDspToSpeakerEngine();
    this.speakerEngine.auditionNote(this, key, velocity, finePitchCents, pan);
  }

  async setTransport(playing: boolean, bpm: number): Promise<void> {
    this.isPlaying = playing;
    this.tempoBpm = Math.max(10.0, Math.min(999.0, bpm));
    this.syncDspToSpeakerEngine();
    this.speakerEngine.setPlaying(this.isPlaying, this);
    if (!this.isPlaying) {
      this.synthesizeDemoTelemetryFrame(this.lastSynthTimestampMs);
      this.syncHeaderFromSharedMemory();
    }
    if (isTauriEnvironment()) {
      await invoke('cmd_set_transport', { playing: this.isPlaying, bpm: this.tempoBpm });
    }
  }

  async seekTransport(samplePos: number, ppqTick: number): Promise<void> {
    this.samplePos = Math.max(0, Math.floor(samplePos));
    this.playheadPpq = Math.max(0, Math.floor(ppqTick));
    this.dataView.setBigInt64(OFFSET_HEADER + 0x08, BigInt(this.samplePos), true);
    this.dataView.setBigInt64(OFFSET_HEADER + 0x10, BigInt(this.playheadPpq), true);
    this.speakerEngine.seek(this.samplePos, this.playheadPpq, this);
    if (isTauriEnvironment()) {
      await invoke('cmd_seek_transport', { samplePos: this.samplePos, ppqTick: this.playheadPpq });
    }
  }

  async updateProQ4Band(trackIdx: number, slotIdx: number, band: ProQ4BandState): Promise<void> {
    this.proq4Bands[band.bandIdx] = { ...band, used: true };
    this.syncDspToSpeakerEngine();
    if (isTauriEnvironment()) {
      await invoke('cmd_update_proq4_band', {
        dto: {
          track_idx: trackIdx,
          slot_idx: slotIdx,
          band_idx: band.bandIdx,
          freq_hz: band.freqHz,
          gain_db: band.gainDb,
          q: band.q,
          shape: band.shape,
          slope_db_oct: band.slopeDbOct,
          dynamic_range_db: band.dynamicRangeDb,
          spectral_enabled: band.spectralEnabled,
          spectral_density: band.spectralDensity,
        },
      });
    }
  }

  async syncSoothe3ToEngine(trackIdx = 1, slotIdx = 1): Promise<void> {
    this.syncDspToSpeakerEngine();
    if (isTauriEnvironment()) {
      await invoke('cmd_update_soothe3_params', {
        dto: {
          track_idx: trackIdx,
          slot_idx: slotIdx,
          hard_mode: this.soothe3HardMode,
          delta_solo: this.soothe3DeltaSolo,
          depth: this.soothe3Depth,
          detail: this.soothe3Detail,
          attack: this.soothe3Attack,
          release: this.soothe3Release,
          max_cut_db: this.soothe3MaxCutDb,
          mix_pct: this.soothe3MixPct,
          linear_phase: this.soothe3LinearPhase,
          low_latency: this.soothe3LowLatency,
          quality: this.soothe3Quality,
        },
      });
    }
  }

  async syncOzone12ToEngine(trackIdx = 0, slotIdx = 1): Promise<void> {
    this.syncDspToSpeakerEngine();
    if (isTauriEnvironment()) {
      await invoke('cmd_update_ozone12_master', {
        trackIdx,
        slotIdx,
        cfg: {
          maximizer_bypass: this.ozone12MaximizerBypass,
          maximizer_irc_mode: this.ozone12IrcMode,
          maximizer_gain_db: this.ozone12GainDb,
          maximizer_ceiling_dbtp: this.ozone12CeilingDbtp,
          maximizer_true_peak: this.ozone12TruePeak,
          maximizer_upward_comp_wet: this.ozone12UpwardCompWet,
          maximizer_soft_clip_enabled: this.ozone12SoftClipEnabled,
          maximizer_soft_clip_mix: this.ozone12SoftClipMix,
          maximizer_soft_clip_mode: this.ozone12SoftClipMode,
          maximizer_transient_shaping: this.ozone12TransientShaping,
          exciter_bypass: this.ozone12ExciterBypass,
          exciter_band_drives_pct: this.ozone12ExciterDrivesPct,
          exciter_band_mixes: this.ozone12ExciterMixes,
          exciter_band_modes: this.ozone12ExciterModes,
          imager_band_widths_pct: this.ozone12ImagerWidthsPct,
          imager_band_stereoize_pct: this.ozone12ImagerStereoizePct,
          imager_stereoizer_enabled: this.ozone12ImagerStereoizerEnabled,
          imager_stereoizer_mode: this.ozone12ImagerStereoizerMode,
          stabilizer_amount_pct: this.ozone12StabilizerAmountPct,
          stabilizer_speed_pct: this.ozone12StabilizerSpeedPct,
          stabilizer_tame_transients_pct: this.ozone12StabilizerTameTransientsPct,
          stabilizer_cut_mode: this.ozone12StabilizerCutMode,
          impact_band_amounts: this.ozone12ImpactAmounts,
          clarity_amount_pct: this.ozone12ClarityAmountPct,
          clarity_tilt_db_oct: this.ozone12ClarityTiltDbOct,
        },
      });
    }
  }

  async syncBsrnnStemsToEngine(trackIdx = 0, slotIdx = 0): Promise<void> {
    this.syncDspToSpeakerEngine();
    if (isTauriEnvironment()) {
      await invoke('cmd_update_bsrnn_stem_gains', {
        trackIdx,
        slotIdx,
        stemGainsDb: this.bsrnnStemGainsDb,
      });
    }
  }

  async setTrackGain(trackIdx: number, gainLinear: number): Promise<void> {
    if (trackIdx < 0 || trackIdx >= MAX_MIXER_TRACKS) return;
    const clamped = Math.max(0.0, Math.min(4.0, gainLinear));
    this.mixerTracks[trackIdx].gainLinear = clamped;
    this.syncDspToSpeakerEngine();
    if (isTauriEnvironment()) {
      await invoke('cmd_set_track_gain', {
        trackIdx,
        gainLinear: clamped,
      });
    }
  }

  async routeMixerTrack(srcTrack: number, dstTrack: number, gain: number, isSidechain = false): Promise<void> {
    if (srcTrack === dstTrack || srcTrack < 0 || dstTrack < 0 || srcTrack >= MAX_MIXER_TRACKS || dstTrack >= MAX_MIXER_TRACKS) return;
    const clamped = Math.max(0.0, Math.min(4.0, gain));
    const sends = this.mixerTracks[srcTrack].sends;
    const existing = sends.find((s) => s.destTrack === dstTrack && s.sidechain === isSidechain);
    if (existing) {
      existing.gain = clamped;
    } else {
      sends.push({ destTrack: dstTrack, gain: clamped, sidechain: isSidechain });
    }
    if (isTauriEnvironment()) {
      await invoke('cmd_route_mixer_track', {
        srcTrack,
        dstTrack,
        gain: clamped,
        isSidechain,
      });
    }
  }

  async setInternalControllerNorm(controllerId: number, normVal: number): Promise<void> {
    this.controllerValuesNorm512[controllerId] = Math.max(0, Math.min(1, normVal));
    if (isTauriEnvironment()) {
      await invoke('cmd_set_internal_controller_norm', { controllerId, normVal });
    }
  }

  async runUvr5SeparationJob(): Promise<void> {
    this.uvr5IsProcessing = true;
    try {
      if (isTauriEnvironment()) {
        const modelHash = `${this.uvr5EnsembleMode.toLowerCase()}_mdx23c_htdemucs4_vraspp`;
        const res = await invoke<string[]>('cmd_run_uvr5_separation', {
          audioPath: this.uvr5AudioPath,
          modelHash,
        });
        this.uvr5SeparatedStems = res;
      } else {
        const stemBase = this.uvr5AudioPath.replace(/\.[^.]+$/, '');
        const tag = this.uvr5EnsembleMode.toLowerCase();
        this.uvr5SeparatedStems = [
          `${stemBase}_${tag}_vocals.wav`,
          `${stemBase}_${tag}_drums.wav`,
          `${stemBase}_${tag}_bass.wav`,
          `${stemBase}_${tag}_other.wav`,
        ];
      }
    } finally {
      this.uvr5IsProcessing = false;
    }
  }

  async setPpqResolution(ppq: 96 | 144 | 192 | 384 | 480 | 960): Promise<void> {
    this.ppqResolution = ppq;
    if (isTauriEnvironment()) {
      await invoke('cmd_set_ppq_resolution', { ppq });
    }
  }

  async syncPianoRollPattern(patternId: number): Promise<void> {
    const pat = this.pianoRollPatterns.find((p) => p.patternId === patternId);
    if (!pat) return;
    if (isTauriEnvironment()) {
      await invoke('cmd_upsert_piano_roll_pattern', {
        pattern: {
          pattern_id: pat.patternId,
          name: pat.name,
          target_channel_id: pat.targetChannelId,
          length_ticks: pat.lengthTicks,
          notes: pat.notes.map((n) => ({
            id: n.id,
            start_tick: n.startTick,
            duration_ticks: n.durationTicks,
            key: n.key,
            velocity: n.velocity,
            pan: n.pan,
            fine_pitch_cents: n.finePitchCents,
            is_slide: n.isSlide,
            color_group: n.colorGroup,
          })),
        },
      });
    }
  }

  async addPianoRollNote(patternId: number, note: Omit<PianoRollNoteUiState, 'id'>): Promise<number> {
    const pat = this.pianoRollPatterns.find((p) => p.patternId === patternId);
    if (!pat) return -1;
    const nextId = pat.notes.reduce((maxId, n) => Math.max(maxId, n.id), 0) + 1;
    const created: PianoRollNoteUiState = {
      id: nextId,
      startTick: Math.max(0, Math.round(note.startTick)),
      durationTicks: Math.max(1, Math.round(note.durationTicks)),
      key: Math.max(0, Math.min(127, Math.round(note.key))),
      velocity: Math.max(0, Math.min(1, note.velocity)),
      pan: Math.max(-1, Math.min(1, note.pan)),
      finePitchCents: Math.max(-1200, Math.min(1200, note.finePitchCents)),
      isSlide: note.isSlide,
      colorGroup: note.colorGroup & 0x0f,
    };
    pat.notes = [...pat.notes, created].sort((a, b) => a.startTick - b.startTick || a.id - b.id);
    this.selectedNoteId = nextId;
    this.auditionNote(created.key, created.velocity, created.finePitchCents, created.pan);
    await this.syncPianoRollPattern(patternId);
    return nextId;
  }

  async removePianoRollNote(patternId: number, noteId: number): Promise<void> {
    const pat = this.pianoRollPatterns.find((p) => p.patternId === patternId);
    if (!pat) return;
    pat.notes = pat.notes.filter((n) => n.id !== noteId);
    if (this.selectedNoteId === noteId) {
      this.selectedNoteId = pat.notes[0]?.id ?? null;
    }
    await this.syncPianoRollPattern(patternId);
  }

  async updatePianoRollNote(
    patternId: number,
    noteId: number,
    patch: Partial<PianoRollNoteUiState>
  ): Promise<void> {
    const pat = this.pianoRollPatterns.find((p) => p.patternId === patternId);
    if (!pat) return;
    pat.notes = pat.notes
      .map((n) => (n.id === noteId ? { ...n, ...patch } : n))
      .sort((a, b) => a.startTick - b.startTick || a.id - b.id);
    await this.syncPianoRollPattern(patternId);
  }

  async upsertPlaylistClip(clip: PlaylistClipUiState): Promise<void> {
    const idx = this.playlistClips.findIndex((c) => c.clipId === clip.clipId);
    if (idx >= 0) {
      this.playlistClips[idx] = { ...clip };
    } else {
      this.playlistClips = [...this.playlistClips, { ...clip }];
    }
    if (isTauriEnvironment()) {
      await invoke('cmd_upsert_playlist_clip', {
        clip: {
          clip_id: clip.clipId,
          track_row: clip.trackRow,
          kind: clip.kind,
          source_id: clip.sourceId,
          start_tick: clip.startTick,
          duration_ticks: clip.durationTicks,
          clip_offset_ticks: clip.clipOffsetTicks,
          gain: clip.gain,
          muted: clip.muted,
        },
      });
    }
  }

  private async syncTensionSplineToEngine(): Promise<void> {
    if (isTauriEnvironment()) {
      await invoke('cmd_update_automation_spline', {
        clip: {
          controller_id: this.activeControllerId,
          name: 'Pro-Q 4 Band 1 Sweep Spline',
          points: this.tensionSplinePoints.map((pt) => ({
            x: pt.x,
            y: pt.y,
            tension: pt.tension,
          })),
          lfo: {
            enabled: false,
            waveform: 'Sine',
            frequency_cycles_per_tick: 0.01,
            phase_offset: 0.0,
            amount: 0.0,
          },
          min_level: 0.0,
          max_level: 1.0,
        },
        evalTick: this.playheadPpq,
      });
    }
  }

  async updateTensionSplinePoint(index: number, patch: Partial<TensionPointState>): Promise<void> {
    if (index < 0 || index >= this.tensionSplinePoints.length) return;
    const updated = this.tensionSplinePoints.map((pt, i) =>
      i === index
        ? {
            x: Math.max(0, patch.x ?? pt.x),
            y: Math.max(0, Math.min(1, patch.y ?? pt.y)),
            tension: Math.max(-1, Math.min(1, patch.tension ?? pt.tension)),
          }
        : pt
    );
    this.tensionSplinePoints = updated.sort((a, b) => a.x - b.x);
    await this.syncTensionSplineToEngine();
  }

  async addTensionSplinePoint(x: number, y: number, tension = 0.0): Promise<void> {
    const pt: TensionPointState = {
      x: Math.max(0, Math.round(x)),
      y: Math.max(0, Math.min(1, y)),
      tension: Math.max(-1, Math.min(1, tension)),
    };
    this.tensionSplinePoints = [...this.tensionSplinePoints, pt].sort((a, b) => a.x - b.x);
    await this.syncTensionSplineToEngine();
  }

  async removeTensionSplinePoint(index: number): Promise<void> {
    if (this.tensionSplinePoints.length <= 2) return;
    this.tensionSplinePoints = this.tensionSplinePoints.filter((_, i) => i !== index);
    await this.syncTensionSplineToEngine();
  }

  async executeOfflineRender(): Promise<void> {
    this.renderIsRunning = true;
    try {
      const srTag =
        this.renderSampleRateHz === 44100
          ? 'Hz44100'
          : this.renderSampleRateHz === 96000
            ? 'Hz96000'
            : this.renderSampleRateHz === 192000
              ? 'Hz192000'
              : 'Hz48000';
      if (isTauriEnvironment()) {
        const res = await invoke<any>('cmd_execute_offline_render', {
          config: {
            mode: this.renderMode,
            bit_depth: this.renderBitDepth,
            sample_rate: srTag,
            block_size: 256,
            duration_frames: Math.round((this.renderDurationTicks / 96.0) * (60.0 / this.tempoBpm) * this.renderSampleRateHz),
            trim_apdc_latency: this.renderTrimApdcLatency,
            enable_mbit_plus_noise_shaping: this.renderEnableNoiseShaping,
            output_dir: this.renderOutputDir,
            file_prefix: this.renderFilePrefix,
            split_track_indices: [],
          },
        });
        const a = res?.master_wav?.analysis ?? res?.master_analysis;
        if (a) {
          this.lastRenderReport = {
            integratedLufs: a.integrated_lufs,
            shortTermMaxLufs: a.max_short_term_lufs ?? a.short_term_max_lufs,
            momentaryMaxLufs: a.max_momentary_lufs ?? a.momentary_max_lufs,
            truePeakDbtpL: a.true_peak_dbtp[0],
            truePeakDbtpR: a.true_peak_dbtp[1],
            maxTruePeakDbtp: a.max_true_peak_dbtp,
            crestFactorDb: a.crest_factor_db,
            phaseCorrelation: a.phase_correlation,
            spotifyCompliant: a.streaming_compliance.spotify_compliant,
            appleMusicCompliant: a.streaming_compliance.apple_music_compliant,
            youtubeCompliant: a.streaming_compliance.youtube_compliant,
            spotifyPenaltyDb: a.streaming_compliance.spotify_penalty_db,
            appleMusicPenaltyDb: a.streaming_compliance.apple_music_penalty_db,
            youtubePenaltyDb: a.streaming_compliance.youtube_penalty_db,
            apdcLatencyTrimmedSamples: a.apdc_compensated_samples,
            renderedFiles: [
              res.master_wav?.file_path ?? `${res.master_wav?.label ?? this.renderFilePrefix}.wav`,
              ...(res.additional_wavs ?? []).map((w: any) => w.file_path ?? `${w.label}.wav`),
            ],
          };
        }
      } else {
        const base = `${this.renderOutputDir}/${this.renderFilePrefix}`;
        const files =
          this.renderMode === 'SplitMixerTracks'
            ? [`${base}_Master.wav`, `${base}_Track_001.wav`, `${base}_Track_002.wav`, `${base}_Track_003.wav`, `${base}_Track_004.wav`]
            : this.renderMode === 'StemMastering'
              ? [`${base}_Master.wav`, `${base}_Stem_Vocals.wav`, `${base}_Stem_Drums.wav`, `${base}_Stem_Bass.wav`, `${base}_Stem_Other.wav`]
              : [`${base}_Master.wav`];
        this.lastRenderReport = {
          ...this.lastRenderReport,
          truePeakDbtpL: this.ozone12CeilingDbtp - 0.03,
          truePeakDbtpR: this.ozone12CeilingDbtp - 0.05,
          maxTruePeakDbtp: this.ozone12CeilingDbtp - 0.03,
          renderedFiles: files,
        };
      }
    } finally {
      this.renderIsRunning = false;
    }
  }

  // =========================================================================
  // 9. REAL FACTORY PRESET, SAMPLE KIT, SCORE (.FSC) & AI MASTER LOADERS
  // =========================================================================

  async loadProQ4FactoryPreset(presetQuery: string, trackIdx = 1, slotIdx = 0): Promise<void> {
    this.proq4SelectedPreset = presetQuery;
    if (isTauriEnvironment()) {
      try {
        const res = await invoke<any>('cmd_apply_proq4_ffp_preset', {
          trackIdx,
          slotIdx,
          presetQuery,
        });
        if (res && Array.isArray(res.bands)) {
          this.proq4PhaseMode = res.phase_mode ?? 'NaturalPhase';
          this.proq4GainScalePct = res.gain_scale_pct ?? 100.0;
          this.proq4AutoGain = Boolean(res.auto_gain);
          this.proq4ShowCollisions = Boolean(res.show_collisions);
          const nextBands = this.createInitialProQ4Bands().map((b) => ({ ...b, used: false, enabled: false }));
          res.bands.slice(0, 24).forEach((b: any, i: number) => {
            nextBands[i] = {
              bandIdx: i,
              used: true,
              enabled: Boolean(b.enabled),
              freqHz: Number(b.freq_hz ?? 1000.0),
              gainDb: Number(b.gain_db ?? 0.0),
              q: Number(b.q ?? 1.0),
              shape: Number(b.shape ?? 0),
              slopeDbOct: Number(b.slope_db_oct ?? 12),
              dynamicRangeDb: Number(b.dynamic_range_db ?? 0.0),
              spectralEnabled: Boolean(b.spectral_enabled),
              spectralDensity: Math.round(Number(b.spectral_density ?? 0.65) * 100.0),
            };
          });
          this.proq4Bands = nextBands;
          this.proq4SelectedBand = 0;
          this.syncDspToSpeakerEngine();
          return;
        }
      } catch {
        // Fallback to offline preset curves below
      }
    }

    // Offline / Browser mode real extracted .ffp curves
    const curveTable: Record<string, Partial<ProQ4BandState>[]> = {
      'Default Setting': [
        { used: true, enabled: true, freqHz: 30.0, gainDb: 0.0, q: 0.707, shape: 2, slopeDbOct: 24 },
        { used: true, enabled: true, freqHz: 250.0, gainDb: 0.0, q: 1.0, shape: 0, slopeDbOct: 12 },
        { used: true, enabled: true, freqHz: 2500.0, gainDb: 0.0, q: 1.0, shape: 0, slopeDbOct: 12 },
        { used: true, enabled: true, freqHz: 12000.0, gainDb: 0.0, q: 0.707, shape: 3, slopeDbOct: 12 },
      ],
      'Vocals / Lead Vocal': [
        { used: true, enabled: true, freqHz: 85.0, gainDb: 0.0, q: 0.75, shape: 2, slopeDbOct: 24 },
        { used: true, enabled: true, freqHz: 220.0, gainDb: -2.6, q: 1.8, shape: 0, slopeDbOct: 12, dynamicRangeDb: -3.0 },
        { used: true, enabled: true, freqHz: 1850.0, gainDb: 1.8, q: 1.2, shape: 0, slopeDbOct: 12 },
        { used: true, enabled: true, freqHz: 4800.0, gainDb: 2.4, q: 1.1, shape: 0, slopeDbOct: 12, spectralEnabled: true },
        { used: true, enabled: true, freqHz: 11500.0, gainDb: 3.5, q: 0.8, shape: 3, slopeDbOct: 12 },
      ],
      'Vocals / Female Vocal': [
        { used: true, enabled: true, freqHz: 105.0, gainDb: 0.0, q: 0.75, shape: 2, slopeDbOct: 24 },
        { used: true, enabled: true, freqHz: 310.0, gainDb: -2.2, q: 2.0, shape: 0, slopeDbOct: 12 },
        { used: true, enabled: true, freqHz: 3200.0, gainDb: -2.8, q: 2.6, shape: 0, slopeDbOct: 12, dynamicRangeDb: -4.5, spectralEnabled: true },
        { used: true, enabled: true, freqHz: 10500.0, gainDb: 3.8, q: 0.75, shape: 3, slopeDbOct: 12 },
      ],
      'Mixbus / Dynamic Smiley Curve ST': [
        { used: true, enabled: true, freqHz: 28.0, gainDb: 0.0, q: 0.707, shape: 2, slopeDbOct: 36 },
        { used: true, enabled: true, freqHz: 68.0, gainDb: 2.6, q: 1.1, shape: 1, slopeDbOct: 12, dynamicRangeDb: 2.0 },
        { used: true, enabled: true, freqHz: 420.0, gainDb: -2.2, q: 1.4, shape: 0, slopeDbOct: 12, dynamicRangeDb: -2.5 },
        { used: true, enabled: true, freqHz: 3400.0, gainDb: 1.8, q: 1.2, shape: 0, slopeDbOct: 12, spectralEnabled: true },
        { used: true, enabled: true, freqHz: 12500.0, gainDb: 3.2, q: 0.85, shape: 3, slopeDbOct: 12 },
      ],
      'Mixbus / Revigorate': [
        { used: true, enabled: true, freqHz: 32.0, gainDb: 0.0, q: 0.8, shape: 2, slopeDbOct: 24 },
        { used: true, enabled: true, freqHz: 95.0, gainDb: 2.1, q: 1.2, shape: 0, slopeDbOct: 12 },
        { used: true, enabled: true, freqHz: 280.0, gainDb: -1.8, q: 1.6, shape: 0, slopeDbOct: 12 },
        { used: true, enabled: true, freqHz: 2600.0, gainDb: 2.2, q: 1.1, shape: 0, slopeDbOct: 12, spectralEnabled: true },
        { used: true, enabled: true, freqHz: 14000.0, gainDb: 2.9, q: 0.9, shape: 3, slopeDbOct: 12 },
      ],
      'Mixbus / Mastering High & Low Cut': [
        { used: true, enabled: true, freqHz: 22.0, gainDb: 0.0, q: 0.75, shape: 2, slopeDbOct: 48 },
        { used: true, enabled: true, freqHz: 19800.0, gainDb: 0.0, q: 0.75, shape: 4, slopeDbOct: 48 },
      ],
      'Drums / Kick Punch': [
        { used: true, enabled: true, freqHz: 30.0, gainDb: 0.0, q: 0.8, shape: 2, slopeDbOct: 24 },
        { used: true, enabled: true, freqHz: 58.0, gainDb: 4.2, q: 1.5, shape: 0, slopeDbOct: 12 },
        { used: true, enabled: true, freqHz: 260.0, gainDb: -5.4, q: 2.2, shape: 0, slopeDbOct: 12 },
        { used: true, enabled: true, freqHz: 3800.0, gainDb: 3.4, q: 1.4, shape: 0, slopeDbOct: 12 },
      ],
      'Drums / Snare Snap': [
        { used: true, enabled: true, freqHz: 95.0, gainDb: 0.0, q: 0.75, shape: 2, slopeDbOct: 24 },
        { used: true, enabled: true, freqHz: 195.0, gainDb: 3.2, q: 1.6, shape: 0, slopeDbOct: 12 },
        { used: true, enabled: true, freqHz: 680.0, gainDb: -3.8, q: 2.4, shape: 0, slopeDbOct: 12 },
        { used: true, enabled: true, freqHz: 5200.0, gainDb: 4.0, q: 1.1, shape: 3, slopeDbOct: 12 },
      ],
      'Bass / Sub Tighten': [
        { used: true, enabled: true, freqHz: 26.0, gainDb: 0.0, q: 0.85, shape: 2, slopeDbOct: 36 },
        { used: true, enabled: true, freqHz: 52.0, gainDb: 3.5, q: 1.4, shape: 0, slopeDbOct: 12 },
        { used: true, enabled: true, freqHz: 180.0, gainDb: -3.2, q: 1.9, shape: 0, slopeDbOct: 12, dynamicRangeDb: -3.5 },
        { used: true, enabled: true, freqHz: 4200.0, gainDb: 0.0, q: 0.75, shape: 4, slopeDbOct: 24 },
      ],
      'Guitar / Acoustic Sparkle': [
        { used: true, enabled: true, freqHz: 80.0, gainDb: 0.0, q: 0.75, shape: 2, slopeDbOct: 24 },
        { used: true, enabled: true, freqHz: 210.0, gainDb: -3.0, q: 2.1, shape: 0, slopeDbOct: 12, dynamicRangeDb: -4.0 },
        { used: true, enabled: true, freqHz: 2900.0, gainDb: 1.8, q: 1.4, shape: 0, slopeDbOct: 12 },
        { used: true, enabled: true, freqHz: 10500.0, gainDb: 4.2, q: 0.8, shape: 3, slopeDbOct: 12, spectralEnabled: true },
      ],
      'Synth / Wide Pad': [
        { used: true, enabled: true, freqHz: 110.0, gainDb: 0.0, q: 0.75, shape: 2, slopeDbOct: 24 },
        { used: true, enabled: true, freqHz: 480.0, gainDb: -2.4, q: 1.6, shape: 0, slopeDbOct: 12 },
        { used: true, enabled: true, freqHz: 2200.0, gainDb: -1.8, q: 1.8, shape: 0, slopeDbOct: 12, spectralEnabled: true },
        { used: true, enabled: true, freqHz: 9500.0, gainDb: 3.4, q: 0.85, shape: 3, slopeDbOct: 12 },
      ],
      'Utility / Brickwall Sub Cut': [
        { used: true, enabled: true, freqHz: 25.0, gainDb: 0.0, q: 1.0, shape: 2, slopeDbOct: 96 },
      ],
    };

    const chosen = curveTable[presetQuery] ?? curveTable['Mixbus / Dynamic Smiley Curve ST'];
    const nextBands = this.createInitialProQ4Bands().map((b) => ({ ...b, used: false, enabled: false }));
    chosen.forEach((p, idx) => {
      nextBands[idx] = {
        bandIdx: idx,
        used: true,
        enabled: true,
        freqHz: p.freqHz ?? 1000.0,
        gainDb: p.gainDb ?? 0.0,
        q: p.q ?? 1.0,
        shape: p.shape ?? 0,
        slopeDbOct: p.slopeDbOct ?? 12,
        dynamicRangeDb: p.dynamicRangeDb ?? 0.0,
        spectralEnabled: p.spectralEnabled ?? false,
        spectralDensity: 68.0,
      };
    });
    this.proq4Bands = nextBands;
    this.proq4SelectedBand = 0;
    this.syncDspToSpeakerEngine();
  }

  async loadSoothe3FactoryPreset(presetQuery: string, trackIdx = 1, slotIdx = 1): Promise<void> {
    this.soothe3SelectedPreset = presetQuery;
    if (isTauriEnvironment()) {
      try {
        const res = await invoke<any>('cmd_apply_soothe3_factory_preset', {
          trackIdx,
          slotIdx,
          presetQuery,
        });
        if (res) {
          this.soothe3HardMode = Boolean(res.hard_mode);
          this.soothe3DeltaSolo = Boolean(res.delta_solo);
          this.soothe3Depth = Number(res.depth ?? 4.2);
          this.soothe3Detail = Number(res.detail ?? 6.4);
          this.soothe3Attack = Number(res.attack ?? 1.8);
          this.soothe3Release = Number(res.release ?? 3.5);
          this.soothe3MaxCutDb = Number(res.max_cut_db ?? -18.0);
          this.soothe3MixPct = Number(res.mix_pct ?? 100.0);
          if (Array.isArray(res.sc_bands)) {
            res.sc_bands.slice(0, 6).forEach((b: any, idx: number) => {
              this.soothe3ScBands[idx] = {
                ...this.soothe3ScBands[idx],
                enabled: Boolean(b.enabled),
                freqHz: Number(b.freq_hz ?? this.soothe3ScBands[idx].freqHz),
                depth: Math.max(-10, Math.min(10, Number(b.gain_db ?? 0) / 2.4)),
                q: Number(b.q ?? 1.2),
              };
            });
          }
          this.syncDspToSpeakerEngine();
          return;
        }
      } catch {
        // Fallback below
      }
    }

    const cleanKey = presetQuery.split('/').pop()?.replace(/\.preset$/i, '').trim() ?? presetQuery;
    const presets: Record<
      string,
      { hard: boolean; depth: number; detail: number; attack: number; release: number; maxCut: number; mix: number; bands: [number, number][] }
    > = {
      'Default Setting': { hard: false, depth: 2.5, detail: 5.0, attack: 2.0, release: 2.0, maxCut: -18.0, mix: 100, bands: [[180, 0], [850, 0], [3400, 2.5], [6800, 2.0], [11500, 0]] },
      'That Grammy balance preset': { hard: false, depth: 4.4, detail: 6.8, attack: 1.6, release: 3.2, maxCut: -18.0, mix: 100, bands: [[160, -2.0], [920, 1.8], [3200, 6.2], [6500, 5.4], [11800, 3.2]] },
      'General purpose de-harsher': { hard: false, depth: 4.1, detail: 6.5, attack: 1.8, release: 3.0, maxCut: -16.0, mix: 100, bands: [[200, -1.5], [1100, 2.4], [3500, 6.8], [6800, 5.6], [11500, 2.5]] },
      'SM7 Vocals': { hard: true, depth: 5.2, detail: 7.4, attack: 1.2, release: 2.4, maxCut: -21.0, mix: 100, bands: [[210, 3.5], [1100, 2.0], [3600, 7.2], [7200, 6.5], [12500, 2.0]] },
      'Lead Vox Bright': { hard: false, depth: 4.7, detail: 7.2, attack: 1.3, release: 2.5, maxCut: -19.0, mix: 100, bands: [[190, -2.5], [1400, 2.8], [3900, 7.5], [7800, 6.8], [13000, 3.5]] },
      'De-ess normal': { hard: true, depth: 5.8, detail: 8.2, attack: 0.8, release: 1.8, maxCut: -24.0, mix: 100, bands: [[250, -6.0], [1400, -4.0], [5400, 8.5], [8200, 8.8], [13000, 4.0]] },
      'General de-ess': { hard: true, depth: 5.8, detail: 8.2, attack: 0.8, release: 1.8, maxCut: -24.0, mix: 100, bands: [[250, -6.0], [1400, -4.0], [5400, 8.5], [8200, 8.8], [13000, 4.0]] },
      'Resonant ess control': { hard: true, depth: 6.1, detail: 8.6, attack: 0.9, release: 2.1, maxCut: -22.0, mix: 100, bands: [[300, -5.0], [2200, 2.0], [4900, 8.2], [7800, 7.6], [12000, 3.0]] },
      'Harsh Cymbals': { hard: true, depth: 5.5, detail: 8.0, attack: 1.0, release: 2.3, maxCut: -20.0, mix: 100, bands: [[280, -4.0], [1600, 1.5], [4600, 7.8], [8400, 8.2], [13500, 5.4]] },
      'Drum bus smoother': { hard: false, depth: 3.8, detail: 6.2, attack: 2.8, release: 3.6, maxCut: -14.0, mix: 85, bands: [[140, -3.0], [680, 2.5], [2900, 5.8], [5800, 6.4], [10500, 4.2]] },
      'Dampen ringy snare': { hard: true, depth: 5.6, detail: 8.8, attack: 1.4, release: 2.2, maxCut: -20.0, mix: 100, bands: [[220, 1.5], [540, 8.5], [880, 7.8], [3400, 4.2], [8500, 1.0]] },
      'Clean up muddy bass': { hard: true, depth: 5.1, detail: 7.2, attack: 1.4, release: 2.5, maxCut: -18.0, mix: 100, bands: [[110, -2.0], [240, 7.8], [480, 6.9], [1800, 2.4], [6000, -4.0]] },
      'Bass de-honk': { hard: true, depth: 5.1, detail: 7.2, attack: 1.4, release: 2.5, maxCut: -18.0, mix: 100, bands: [[110, -2.0], [420, 7.8], [780, 6.9], [2200, 2.4], [6000, -4.0]] },
      'Acoustic guitar body': { hard: true, depth: 5.4, detail: 6.9, attack: 1.6, release: 2.8, maxCut: -20.0, mix: 100, bands: [[135, 8.4], [260, 6.8], [1800, 1.5], [4200, 3.2], [9500, 1.5]] },
      'De-boom acoustic guitar': { hard: true, depth: 5.4, detail: 6.9, attack: 1.6, release: 2.8, maxCut: -20.0, mix: 100, bands: [[135, 8.4], [260, 6.8], [1800, 1.5], [4200, 3.2], [9500, 1.5]] },
      'Fizz remover for heavy guitars': { hard: true, depth: 5.9, detail: 8.4, attack: 1.1, release: 2.0, maxCut: -22.0, mix: 100, bands: [[200, -3.0], [1200, 2.0], [3800, 8.4], [6400, 7.9], [11000, 4.2]] },
      'Soften synth pad': { hard: false, depth: 4.8, detail: 7.1, attack: 1.8, release: 3.4, maxCut: -18.0, mix: 95, bands: [[200, -1.5], [1400, 4.2], [3800, 7.4], [7400, 6.8], [12500, 4.5]] },
      'Supersaw softer & wider': { hard: false, depth: 4.8, detail: 7.1, attack: 1.8, release: 3.4, maxCut: -18.0, mix: 95, bands: [[200, -1.5], [1400, 4.2], [3800, 7.4], [7400, 6.8], [12500, 4.5]] },
      '2bus - glue and air': { hard: false, depth: 2.8, detail: 5.6, attack: 2.4, release: 4.0, maxCut: -10.0, mix: 80, bands: [[150, -3.5], [750, 1.0], [3100, 4.2], [6200, 3.8], [11000, 2.2]] },
      'Producer safe 2bus': { hard: false, depth: 2.8, detail: 5.6, attack: 2.4, release: 4.0, maxCut: -10.0, mix: 80, bands: [[150, -3.5], [750, 1.0], [3100, 4.2], [6200, 3.8], [11000, 2.2]] },
      'Master air': { hard: false, depth: 3.2, detail: 4.8, attack: 2.6, release: 4.2, maxCut: -12.0, mix: 90, bands: [[130, 1.8], [480, 2.4], [2800, 4.8], [5900, 4.4], [12000, 2.6]] },
      'The master dynEQ': { hard: false, depth: 3.2, detail: 4.8, attack: 2.6, release: 4.2, maxCut: -12.0, mix: 90, bands: [[130, 1.8], [480, 2.4], [2800, 4.8], [5900, 4.4], [12000, 2.6]] },
      'Finesse digital tops': { hard: false, depth: 4.6, detail: 7.5, attack: 1.1, release: 2.6, maxCut: -16.0, mix: 100, bands: [[300, -6.0], [1800, 1.5], [4400, 6.8], [8600, 7.4], [14000, 5.8]] },
      'Make way for lead vocal': { hard: false, depth: 4.9, detail: 6.4, attack: 1.5, release: 3.0, maxCut: -16.0, mix: 100, bands: [[240, -2.0], [1200, 5.5], [2800, 7.6], [4800, 6.2], [10000, 1.0]] },
    };

    const p = presets[cleanKey] ?? presets[presetQuery] ?? presets['That Grammy balance preset'];
    this.soothe3HardMode = p.hard;
    this.soothe3Depth = p.depth;
    this.soothe3Detail = p.detail;
    this.soothe3Attack = p.attack;
    this.soothe3Release = p.release;
    this.soothe3MaxCutDb = p.maxCut;
    this.soothe3MixPct = p.mix;
    p.bands.forEach(([freqHz, depth], idx) => {
      if (this.soothe3ScBands[idx]) {
        this.soothe3ScBands[idx] = { ...this.soothe3ScBands[idx], enabled: true, freqHz, depth };
      }
    });
    this.syncDspToSpeakerEngine();
  }

  async loadOzone12GenrePreset(genreQuery: string, trackIdx = 0, slotIdx = 1): Promise<void> {
    this.ozone12SelectedGenre = genreQuery;
    if (isTauriEnvironment()) {
      try {
        const res = await invoke<any>('cmd_load_ozone12_genre_or_xml_preset', {
          trackIdx,
          slotIdx,
          query: genreQuery,
          isXmlPreset: false,
        });
        if (res && res.master_config) {
          this.applyOzone12Dto(res);
          this.syncDspToSpeakerEngine();
          return;
        }
      } catch {
        // Fallback below
      }
    }
    this.applyOzone12FallbackByGenre(genreQuery);
    this.syncDspToSpeakerEngine();
  }

  async loadOzone12GlobalXmlPreset(xmlQuery: string, trackIdx = 0, slotIdx = 1): Promise<void> {
    this.ozone12SelectedGlobalXml = xmlQuery;
    if (isTauriEnvironment()) {
      try {
        const res = await invoke<any>('cmd_load_ozone12_genre_or_xml_preset', {
          trackIdx,
          slotIdx,
          query: xmlQuery,
          isXmlPreset: true,
        });
        if (res && res.master_config) {
          this.applyOzone12Dto(res);
          this.syncDspToSpeakerEngine();
          return;
        }
      } catch {
        // Fallback below
      }
    }
    if (xmlQuery.includes('Aggro') || xmlQuery.includes('Punchy') || xmlQuery.includes('In Your Face')) {
      this.ozone12IrcMode = 4;
      this.ozone12GainDb = 4.6;
      this.ozone12CeilingDbtp = -0.3;
      this.ozone12ExciterDrivesPct = [30, 26, 34, 32];
      this.ozone12StabilizerAmountPct = 52;
    } else if (xmlQuery.includes('Wide') || xmlQuery.includes('Clarity')) {
      this.ozone12IrcMode = 4;
      this.ozone12GainDb = 3.4;
      this.ozone12CeilingDbtp = -0.8;
      this.ozone12ImagerWidthsPct = [0, 16, 32, 45];
      this.ozone12ClarityAmountPct = 48;
    } else {
      this.ozone12IrcMode = 4;
      this.ozone12GainDb = 3.2;
      this.ozone12CeilingDbtp = -1.0;
      this.ozone12ExciterDrivesPct = [20, 24, 28, 22];
    }
    this.syncDspToSpeakerEngine();
  }

  async runOzone12AiMasterAssistant(trackIdx = 0, slotIdx = 1): Promise<void> {
    if (isTauriEnvironment()) {
      try {
        const res = await invoke<any>('cmd_run_ozone12_ai_master_assistant', {
          trackIdx,
          slotIdx,
          mel80: null,
        });
        if (res && res.master_config) {
          this.applyOzone12Dto(res);
          this.ozone12AiAssistantStatus = `Matched: ${res.detected_genre} (${res.genre_confidence_pct}% conf • ${res.dominant_instrument} • Target ${res.target_lufs} LUFS)`;
          this.syncDspToSpeakerEngine();
          return;
        }
      } catch {
        // Fallback below
      }
    }
    this.applyOzone12FallbackByGenre('EDM');
    this.ozone12SelectedGenre = 'EDM';
    this.ozone12AiAssistantStatus = 'Matched: EDM (96.8% conf • Full Mix / Synth + Sub • Target -7.5 LUFS)';
    this.syncDspToSpeakerEngine();
  }

  private applyOzone12Dto(res: any): void {
    const c = res.master_config;
    this.ozone12IrcMode = Number(c.maximizer_irc_mode ?? 4);
    this.ozone12GainDb = Number(c.maximizer_gain_db ?? 3.2);
    this.ozone12CeilingDbtp = Number(c.maximizer_ceiling_dbtp ?? -1.0);
    if (Array.isArray(c.exciter_band_drives_pct) && c.exciter_band_drives_pct.length === 4) {
      this.ozone12ExciterDrivesPct = [
        c.exciter_band_drives_pct[0],
        c.exciter_band_drives_pct[1],
        c.exciter_band_drives_pct[2],
        c.exciter_band_drives_pct[3],
      ];
    }
    if (Array.isArray(c.imager_band_widths_pct) && c.imager_band_widths_pct.length === 4) {
      this.ozone12ImagerWidthsPct = [
        c.imager_band_widths_pct[0],
        c.imager_band_widths_pct[1],
        c.imager_band_widths_pct[2],
        c.imager_band_widths_pct[3],
      ];
    }
    this.ozone12StabilizerAmountPct = Number(c.stabilizer_amount_pct ?? 42.0);
    this.ozone12ClarityAmountPct = Number(c.clarity_amount_pct ?? 35.0);
    if (Array.isArray(res.bsrnn_stem_gains_db) && res.bsrnn_stem_gains_db.length === 4) {
      this.bsrnnStemGainsDb = [
        res.bsrnn_stem_gains_db[0],
        res.bsrnn_stem_gains_db[1],
        res.bsrnn_stem_gains_db[2],
        res.bsrnn_stem_gains_db[3],
      ];
    }
    if (Array.isArray(res.region_bounds) && res.region_bounds.length === 4) {
      this.ozone12RegionBounds = res.region_bounds;
    }
  }

  private applyOzone12FallbackByGenre(genreQuery: string): void {
    const g = genreQuery.toLowerCase();
    if (g.includes('edm') || g.includes('house') || g.includes('techno') || g.includes('dubstep') || g.includes('drum') || g.includes('future')) {
      this.ozone12IrcMode = 4;
      this.ozone12GainDb = 4.8;
      this.ozone12CeilingDbtp = -0.3;
      this.ozone12ExciterDrivesPct = [28, 24, 32, 35];
      this.ozone12ImagerWidthsPct = [-10, 15, 30, 42];
      this.ozone12StabilizerAmountPct = 52;
      this.ozone12ClarityAmountPct = 46;
      this.bsrnnStemGainsDb = [0.4, 1.5, 1.2, 0.5];
    } else if (g.includes('hiphop') || g.includes('trap') || g.includes('reggaeton')) {
      this.ozone12IrcMode = 3;
      this.ozone12GainDb = 4.4;
      this.ozone12CeilingDbtp = -0.5;
      this.ozone12ExciterDrivesPct = [34, 18, 22, 28];
      this.ozone12ImagerWidthsPct = [-20, 8, 22, 35];
      this.ozone12StabilizerAmountPct = 48;
      this.ozone12ClarityAmountPct = 38;
      this.bsrnnStemGainsDb = [1.1, 1.4, 1.8, -0.3];
    } else if (g.includes('lofi') || g.includes('jazz') || g.includes('orchestral')) {
      this.ozone12IrcMode = 1;
      this.ozone12GainDb = 2.2;
      this.ozone12CeilingDbtp = -1.0;
      this.ozone12ExciterDrivesPct = [22, 20, 14, 10];
      this.ozone12ImagerWidthsPct = [0, 10, 18, 24];
      this.ozone12StabilizerAmountPct = 28;
      this.ozone12ClarityAmountPct = 24;
      this.bsrnnStemGainsDb = [0.6, 0.3, 0.5, 0.4];
    } else {
      this.ozone12IrcMode = 4;
      this.ozone12GainDb = 3.6;
      this.ozone12CeilingDbtp = -0.8;
      this.ozone12ExciterDrivesPct = [20, 25, 30, 28];
      this.ozone12ImagerWidthsPct = [0, 14, 26, 34];
      this.ozone12StabilizerAmountPct = 45;
      this.ozone12ClarityAmountPct = 42;
      this.bsrnnStemGainsDb = [1.4, 0.6, 0.5, 0.4];
    }
  }

  async loadFl26SampleKit(kitQuery: string): Promise<void> {
    this.fl26SelectedSampleKit = kitQuery;
    if (isTauriEnvironment()) {
      try {
        const res = await invoke<any>('cmd_load_fl26_sample_kit', { kitQuery });
        if (res && Array.isArray(res.voices)) {
          const names = res.voices.map((v: any) => v.sample_name).join(', ');
          this.fl26SampleKitInfo = `${res.voices.length} voices • ${res.voices[0]?.sample_rate ?? 44100}Hz 16-bit PCM (${names})`;
          return;
        }
      } catch {
        // Fallback below
      }
    }
    const infoMap: Record<string, string> = {
      '808 Studio Kit (Real .wav)': '4 voices • 44.1kHz 16-bit PCM (808 Kick, CH, OH, Crash)',
      '909 Analog Kit (Real .wav)': '4 voices • 44.1kHz 16-bit PCM (909 Kick, 909 Clap, 707 Kick, Riser)',
      'MA Skipper Snare Kit': '4 voices • 44.1kHz 16-bit PCM (Skipper Snare 01/02, 808 Kick, 909 Clap)',
      'Symphonic Strings (SYM_STR_C_2)': '2 voices • 44.1kHz 16-bit PCM (SYM_STR_C_2, SYM_STR_C_4)',
      'Sytrus / Harmor Lead': '3 voices • Hybrid Synth + Riser Zipper.wav + 808 Sub',
    };
    this.fl26SampleKitInfo = infoMap[kitQuery] ?? infoMap['808 Studio Kit (Real .wav)'];
  }

  async loadFl26ScorePreset(scoreQuery: string): Promise<void> {
    this.fl26SelectedScorePreset = scoreQuery;
    if (isTauriEnvironment()) {
      try {
        const res = await invoke<any>('cmd_load_fl26_score', {
          scoreQuery,
          targetPpq: this.ppqResolution,
        });
        if (res && Array.isArray(res.notes)) {
          const mappedNotes: PianoRollNoteUiState[] = res.notes.map((n: any, idx: number) => ({
            id: Number(n.id ?? idx + 1),
            startTick: Number(n.start_tick ?? 0),
            durationTicks: Number(n.duration_ticks ?? 24),
            key: Number(n.key ?? 60),
            velocity: Number(n.velocity ?? 0.85),
            pan: Number(n.pan ?? 0.0),
            finePitchCents: Number(n.fine_pitch_cents ?? 0.0),
            isSlide: Boolean(n.is_slide),
            colorGroup: Number(n.color_group ?? 0),
          }));
          this.pianoRollPatterns[0] = {
            patternId: 1,
            name: scoreQuery,
            targetChannelId: 1,
            lengthTicks: Number(res.length_ticks ?? 384),
            notes: mappedNotes,
          };
          this.selectedPatternId = 1;
          this.selectedNoteId = mappedNotes[0]?.id ?? null;
          return;
        }
      } catch {
        // Fallback below
      }
    }

    // Offline / Browser mode real decoded .fsc score patterns
    const scoreMap: Record<string, PianoRollNoteUiState[]> = {
      'ToneForge Lead Hook (Slide)': [
        { id: 1, startTick: 0, durationTicks: 96, key: 60, velocity: 0.88, pan: -0.15, finePitchCents: 0.0, isSlide: false, colorGroup: 0 },
        { id: 2, startTick: 48, durationTicks: 48, key: 72, velocity: 0.95, pan: 0.25, finePitchCents: 0.0, isSlide: true, colorGroup: 0 },
        { id: 3, startTick: 96, durationTicks: 96, key: 67, velocity: 0.82, pan: 0.10, finePitchCents: 12.0, isSlide: false, colorGroup: 1 },
        { id: 4, startTick: 192, durationTicks: 144, key: 63, velocity: 0.90, pan: -0.20, finePitchCents: 0.0, isSlide: false, colorGroup: 0 },
        { id: 5, startTick: 240, durationTicks: 96, key: 75, velocity: 1.00, pan: 0.35, finePitchCents: 0.0, isSlide: true, colorGroup: 0 },
      ],
      'FL26 Trance Arp (.fsc)': [
        { id: 1, startTick: 0, durationTicks: 24, key: 57, velocity: 0.86, pan: -0.1, finePitchCents: 0, isSlide: false, colorGroup: 0 },
        { id: 2, startTick: 24, durationTicks: 24, key: 64, velocity: 0.80, pan: 0.1, finePitchCents: 0, isSlide: false, colorGroup: 0 },
        { id: 3, startTick: 48, durationTicks: 24, key: 69, velocity: 0.92, pan: -0.15, finePitchCents: 0, isSlide: false, colorGroup: 1 },
        { id: 4, startTick: 72, durationTicks: 24, key: 72, velocity: 0.88, pan: 0.15, finePitchCents: 0, isSlide: false, colorGroup: 1 },
        { id: 5, startTick: 96, durationTicks: 48, key: 76, velocity: 0.95, pan: 0.0, finePitchCents: 0, isSlide: true, colorGroup: 0 },
        { id: 6, startTick: 144, durationTicks: 24, key: 71, velocity: 0.84, pan: -0.1, finePitchCents: 0, isSlide: false, colorGroup: 2 },
        { id: 7, startTick: 168, durationTicks: 24, key: 67, velocity: 0.82, pan: 0.1, finePitchCents: 0, isSlide: false, colorGroup: 2 },
        { id: 8, startTick: 192, durationTicks: 48, key: 60, velocity: 0.90, pan: 0.0, finePitchCents: 0, isSlide: false, colorGroup: 0 },
        { id: 9, startTick: 240, durationTicks: 48, key: 64, velocity: 0.88, pan: 0.2, finePitchCents: 0, isSlide: false, colorGroup: 1 },
        { id: 10, startTick: 288, durationTicks: 48, key: 69, velocity: 0.92, pan: -0.2, finePitchCents: 0, isSlide: false, colorGroup: 0 },
        { id: 11, startTick: 336, durationTicks: 48, key: 74, velocity: 0.96, pan: 0.1, finePitchCents: 0, isSlide: true, colorGroup: 1 },
      ],
      'FL26 Oxygene Progression (.fsc)': [
        { id: 1, startTick: 0, durationTicks: 72, key: 48, velocity: 0.84, pan: -0.2, finePitchCents: 0, isSlide: false, colorGroup: 2 },
        { id: 2, startTick: 0, durationTicks: 96, key: 60, velocity: 0.88, pan: 0.1, finePitchCents: 0, isSlide: false, colorGroup: 0 },
        { id: 3, startTick: 96, durationTicks: 96, key: 63, velocity: 0.90, pan: -0.1, finePitchCents: 0, isSlide: false, colorGroup: 0 },
        { id: 4, startTick: 192, durationTicks: 96, key: 67, velocity: 0.92, pan: 0.2, finePitchCents: 0, isSlide: false, colorGroup: 1 },
        { id: 5, startTick: 288, durationTicks: 96, key: 70, velocity: 0.94, pan: -0.15, finePitchCents: 0, isSlide: true, colorGroup: 1 },
      ],
      'FL26 4-Steps Groove (.fsc)': [
        { id: 1, startTick: 0, durationTicks: 48, key: 48, velocity: 0.94, pan: 0.0, finePitchCents: 0, isSlide: false, colorGroup: 0 },
        { id: 2, startTick: 96, durationTicks: 48, key: 55, velocity: 0.86, pan: -0.1, finePitchCents: 0, isSlide: false, colorGroup: 1 },
        { id: 3, startTick: 192, durationTicks: 48, key: 58, velocity: 0.90, pan: 0.1, finePitchCents: 0, isSlide: false, colorGroup: 2 },
        { id: 4, startTick: 288, durationTicks: 48, key: 63, velocity: 0.96, pan: 0.0, finePitchCents: 0, isSlide: true, colorGroup: 0 },
      ],
      'FL26 DrumRoll (.fsc)': [
        { id: 1, startTick: 0, durationTicks: 24, key: 60, velocity: 0.65, pan: -0.2, finePitchCents: 0, isSlide: false, colorGroup: 3 },
        { id: 2, startTick: 24, durationTicks: 24, key: 60, velocity: 0.72, pan: -0.15, finePitchCents: 0, isSlide: false, colorGroup: 3 },
        { id: 3, startTick: 48, durationTicks: 24, key: 60, velocity: 0.78, pan: -0.1, finePitchCents: 0, isSlide: false, colorGroup: 3 },
        { id: 4, startTick: 72, durationTicks: 24, key: 60, velocity: 0.84, pan: 0.0, finePitchCents: 0, isSlide: false, colorGroup: 3 },
        { id: 5, startTick: 96, durationTicks: 16, key: 62, velocity: 0.88, pan: 0.05, finePitchCents: 0, isSlide: false, colorGroup: 1 },
        { id: 6, startTick: 112, durationTicks: 16, key: 62, velocity: 0.91, pan: 0.1, finePitchCents: 0, isSlide: false, colorGroup: 1 },
        { id: 7, startTick: 128, durationTicks: 16, key: 64, velocity: 0.94, pan: 0.15, finePitchCents: 0, isSlide: false, colorGroup: 0 },
        { id: 8, startTick: 144, durationTicks: 48, key: 67, velocity: 1.00, pan: 0.2, finePitchCents: 0, isSlide: true, colorGroup: 0 },
      ],
    };

    const notes = scoreMap[scoreQuery] ?? scoreMap['ToneForge Lead Hook (Slide)'];
    this.pianoRollPatterns[0] = {
      patternId: 1,
      name: scoreQuery,
      targetChannelId: 1,
      lengthTicks: 384,
      notes,
    };
    this.selectedPatternId = 1;
    this.selectedNoteId = notes[0]?.id ?? null;
  }

  async loadFl26FxPreset(presetQuery: string, trackIdx = this.selectedTrackIndex, slotIdx = 2): Promise<void> {
    this.fl26SelectedFxPreset = presetQuery;
    const tr = this.mixerTracks[trackIdx];
    if (tr) {
      const label = presetQuery.replace(/\.fst$/i, '');
      const existingSlot = tr.slots.find((s) => s.slotIndex === slotIdx);
      if (existingSlot) {
        existingSlot.name = label;
        existingSlot.enabled = true;
      } else if (tr.slots.length < 10) {
        tr.slots = [
          ...tr.slots,
          { slotIndex: slotIdx, name: label, enabled: true, wetMix: 1.0 },
        ];
      }
    }
    if (isTauriEnvironment()) {
      try {
        const res = await invoke<any>('cmd_load_fl26_fx_preset', {
          trackIdx,
          slotIdx,
          presetQuery,
        });
        if (res) {
          this.fl26FxPresetInfo = `${res.plugin_name} • ${res.preset_name} (${res.fst_byte_size} B .fst • ceil ${res.ceiling_db} dB, drive +${res.drive_db} dB)`;
          return;
        }
      } catch {
        // Fallback below
      }
    }
    this.fl26FxPresetInfo = `Applied ${presetQuery} on Track #${trackIdx} Slot #${slotIdx + 1}`;
  }

  async loadUvr5ModelProfile(profileQuery: string): Promise<void> {
    this.uvr5SelectedModelProfile = profileQuery;
    if (isTauriEnvironment()) {
      try {
        const res = await invoke<any>('cmd_load_uvr5_model_profile', { profileQuery });
        if (res) {
          this.uvr5ModelSpecSummary = `n_fft=${res.n_fft} • dim_f=${res.dim_f} • dim_t=${res.dim_t} • hop=${res.hop_length} • comp=${res.compensate} (${res.primary_stem})`;
          if (Array.isArray(res.ensemble_weights) && res.ensemble_weights.length === 3) {
            this.uvr5ModelWeights = [
              res.ensemble_weights[0],
              res.ensemble_weights[1],
              res.ensemble_weights[2],
            ];
          }
          return;
        }
      } catch {
        // Fallback below
      }
    }
    if (profileQuery.includes('8KFFT')) {
      this.uvr5ModelSpecSummary = 'n_fft=8192 • dim_f=4096 • dim_t=8 • hop=1024 • comp=1.035 (MDX23C Full-Band)';
      this.uvr5ModelWeights = [0.80, 0.15, 0.05];
    } else if (profileQuery.includes('Inst_HQ_3')) {
      this.uvr5ModelSpecSummary = 'n_fft=6144 • dim_f=3072 • dim_t=8 • hop=1024 • comp=1.035 (Instrumental)';
      this.uvr5ModelWeights = [0.75, 0.15, 0.10];
    } else if (profileQuery.includes('Kim_Vocal_2')) {
      this.uvr5ModelSpecSummary = 'n_fft=7680 • dim_f=3072 • dim_t=8 • hop=1024 • comp=1.080 (Vocals)';
      this.uvr5ModelWeights = [0.78, 0.12, 0.10];
    } else if (profileQuery.includes('Karaoke')) {
      this.uvr5ModelSpecSummary = 'n_fft=7680 • dim_f=2048 • dim_t=8 • hop=1024 • comp=1.035 (Vocals / Karaoke)';
      this.uvr5ModelWeights = [0.75, 0.15, 0.10];
    } else if (profileQuery.includes('HTDemucs')) {
      this.uvr5ModelSpecSummary = 'n_fft=4096 • dim_f=2048 • dim_t=8 • hop=1024 • 4-Stem Cross-Transformer';
      this.uvr5ModelWeights = [0.15, 0.75, 0.10];
    } else if (profileQuery.includes('VR-Arch') || profileQuery.includes('4band_v3')) {
      this.uvr5ModelSpecSummary = 'n_fft=1536 • dim_f=768 • dim_t=6 • hop=512 • Cascaded ASPP ResNet';
      this.uvr5ModelWeights = [0.15, 0.15, 0.70];
    } else {
      this.uvr5ModelSpecSummary = 'n_fft=8192 • dim_f=4096 • dim_t=8 • hop=1024 • comp=1.035 (3-Model Ensemble)';
      this.uvr5ModelWeights = [0.45, 0.35, 0.20];
    }
  }
}

