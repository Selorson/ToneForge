// Zero-Copy 128 KB Binary Telemetry Decoder & Svelte 5 Reactive Store (`daw_master_store.svelte.ts`)
// Maps `VisualTelemetryFrame` (`131,072` bytes = `0x20000`, `#[repr(C, align(64))]`) directly into
// `DataView` and `Float32Array` views for 120 FPS WebGPU rendering without JSON serialization.

import { invoke } from '@tauri-apps/api/core';

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
  // 3. SVELTE 5 RUNES — FABFILTER PRO-Q 4 (24 BANDS / 601 PARAMS)
  // =========================================================================
  proq4SelectedBand = $state(0);
  proq4PhaseMode = $state<'ZeroLatency' | 'NaturalPhase' | 'LinearPhase'>('NaturalPhase');
  proq4GainScalePct = $state(100.0);
  proq4AutoGain = $state(true);
  proq4ShowCollisions = $state(true);
  proq4Bands = $state<ProQ4BandState[]>(this.createInitialProQ4Bands());

  // =========================================================================
  // 4. SVELTE 5 RUNES — OEKSOUND SOOTHE3 (143 PARAMS & 8 SC BANDS)
  // =========================================================================
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
    { bandIdx: 7, enabled: false, freqHz: 15000.0, depth: 0.0, q: 1.0, shape: 5 },
  ]);

  // =========================================================================
  // 5. SVELTE 5 RUNES — IZOTOPE OZONE 12 & 120MS BSRNN MASTER REBALANCE
  // =========================================================================
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

  /**
   * Polls the 128 KB binary telemetry frame at 120 Hz from the Tauri Rust backend
   * (or synthesizes deterministic 120 Hz telemetry when running in standalone browser mode).
   */
  async pollTelemetry120Hz(timestampMs: number): Promise<void> {
    if (isTauriEnvironment()) {
      try {
        const raw = await invoke<ArrayBuffer | number[]>('cmd_poll_binary_telemetry_frame');
        if (raw instanceof ArrayBuffer && raw.byteLength === VISUAL_TELEMETRY_FRAME_BYTES) {
          this.u8View.set(new Uint8Array(raw));
          this.syncHeaderFromSharedMemory();
          return;
        } else if (Array.isArray(raw) && raw.length === VISUAL_TELEMETRY_FRAME_BYTES) {
          this.u8View.set(raw);
          this.syncHeaderFromSharedMemory();
          return;
        }
      } catch {
        // Fallback to local 120 Hz synthesis if command is not registered in preview mode
      }
    }
    this.synthesizeDemoTelemetryFrame(timestampMs);
    this.syncHeaderFromSharedMemory();
  }

  /**
   * Synthesizes a realistic 128 KB binary frame directly into `this.buffer` at exact byte offsets
   * so all WebGPU shaders and meters animate at 120 FPS even in standalone browser mode.
   */
  synthesizeDemoTelemetryFrame(timestampMs: number): void {
    const t = timestampMs * 0.001;
    const dv = this.dataView;
    const nextSeq = BigInt(this.frameSeq + 1);
    const nextSamples = BigInt(Math.floor(t * 48000));
    const nextPpq = BigInt(Math.floor((t * (this.tempoBpm / 60.0)) * 96.0));

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
    for (let k = 0; k < 1024; k++) {
      const normX = k / 1023.0;
      const freq = 20.0 * Math.pow(1000.0, normX);
      const pinkSlope = -18.0 - 12.0 * normX;
      const harmonic = 8.0 * Math.sin(normX * 28.0 + t * 3.2) + 4.5 * Math.cos(normX * 75.0 - t * 6.0);
      const preDb = pinkSlope + harmonic;
      const eqBoost = this.evaluateProQ4MagnitudeDbAt(freq);
      const postDb = preDb + eqBoost;

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
  // IPC COMMAND DISPATCHERS TO RUST `daw-tauri-host`
  // =========================================================================

  async setTransport(playing: boolean, bpm: number): Promise<void> {
    this.isPlaying = playing;
    this.tempoBpm = Math.max(10.0, Math.min(999.0, bpm));
    if (isTauriEnvironment()) {
      await invoke('cmd_set_transport', { playing: this.isPlaying, bpm: this.tempoBpm });
    }
  }

  async seekTransport(samplePos: number, ppqTick: number): Promise<void> {
    this.samplePos = Math.max(0, Math.floor(samplePos));
    this.playheadPpq = Math.max(0, Math.floor(ppqTick));
    if (isTauriEnvironment()) {
      await invoke('cmd_seek_transport', { samplePos: this.samplePos, ppqTick: this.playheadPpq });
    }
  }

  async updateProQ4Band(trackIdx: number, slotIdx: number, band: ProQ4BandState): Promise<void> {
    this.proq4Bands[band.bandIdx] = { ...band, used: true };
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
}
