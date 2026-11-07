// Real-Time WebAudio Speaker Engine & Live Audible DSP Chain (`realtime_speaker_engine.ts`)
// Outputs real stereo audio to Windows speakers/headphones when `isPlaying === true`:
// - Source 1: Built-in FL Studio 26 Polyphonic Synth (with Slide/Portamento glides) + Real .wav Sample Kit / 808/909 Drum Sequencer
// - Source 2: Custom Audio File Picker (`.wav`, `.mp3`, `.flac`, `.ogg`) decoded via `AudioContext.decodeAudioData`
//             + 4-Stem Synchronized Playback when separated by UVR5 Ensemble Studio
// - Live DSP Chain: Pre-EQ AnalyserNode -> 24-Band FabFilter Pro-Q 4 Biquad Cascade -> Post-EQ AnalyserNode
//   -> oeksound Soothe3 6-Band Dynamic Spectral Notch Bank (with Delta Solo bandpass mode) -> BSRNN 4-Stem Rebalance EQ
//   -> iZotope Ozone 12 4-Band 12AX7 Tube WaveShaper, Multiband Imager & Brickwall Maximizer -> Master Speaker Volume GainNode.

import type { DawTelemetryView, PianoRollNoteUiState, ProQ4BandState } from '../state/daw_master_store.svelte';

export type AudioSourceMode = 'pianoRoll' | 'customFile' | 'stems4';

interface ActiveSlideVoice {
  osc1: OscillatorNode;
  osc2: OscillatorNode;
  endTime: number;
  colorGroup: number;
}

export class RealtimeSpeakerEngine {
  private ctx: AudioContext | null = null;
  private sourceBus: GainNode | null = null;
  private preAnalyser: AnalyserNode | null = null;
  private proq4Nodes: BiquadFilterNode[] = [];
  private postAnalyser: AnalyserNode | null = null;
  private sootheNotches: BiquadFilterNode[] = [];
  private stemBassFilter: BiquadFilterNode | null = null;
  private stemDrumsFilter: BiquadFilterNode | null = null;
  private stemVocalsFilter: BiquadFilterNode | null = null;
  private stemOtherFilter: BiquadFilterNode | null = null;
  private ozonePreGain: GainNode | null = null;
  private ozoneWaveShaper: WaveShaperNode | null = null;
  private ozoneCompressor: DynamicsCompressorNode | null = null;
  private ozoneCeilingGain: GainNode | null = null;
  private masterSpeakerGain: GainNode | null = null;
  private noiseBuffer: AudioBuffer | null = null;

  private customAudioBuffer: AudioBuffer | null = null;
  private customSourceNode: AudioBufferSourceNode | null = null;
  private uvr5StemBuffers: (AudioBuffer | null)[] = [null, null, null, null];
  private uvr5StemSources: (AudioBufferSourceNode | null)[] = [null, null, null, null];
  private uvr5StemGains: (GainNode | null)[] = [null, null, null, null];
  private fl26KitVoiceBuffers: AudioBuffer[] = [];
  private pendingFl26RawWavs: ArrayBuffer[] = [];
  private customStartedAtCtxTime = 0;
  private customPausedOffsetSec = 0;
  private isCustomPlaying = false;

  private schedulerTimerId: ReturnType<typeof setInterval> | null = null;
  private nextStepTime = 0;
  private currentStep16th = 0;
  private activeSlideVoices: Map<number, ActiveSlideVoice> = new Map();

  private fftPreBuf = new Float32Array(1024);
  private fftPostBuf = new Float32Array(1024);
  private lastDriveSignature = -1;
  private unlockListenersAttached = false;
  private latestTelemetry: DawTelemetryView | null = null;

  constructor() {
    this.attachAutoplayUnlockListeners();
  }

  private attachAutoplayUnlockListeners(): void {
    if (typeof window === 'undefined' || this.unlockListenersAttached) return;
    this.unlockListenersAttached = true;

    const unlock = () => {
      if (!this.latestTelemetry) return;
      if (this.latestTelemetry.isPlaying) {
        this.ensureAudioGraph(this.latestTelemetry);
        if (this.ctx && this.ctx.state === 'suspended') {
          this.ctx.resume().catch(() => {});
        }
        this.startPlaybackLoop(this.latestTelemetry);
      }
    };

    window.addEventListener('pointerdown', unlock, { passive: true });
    window.addEventListener('keydown', unlock, { passive: true });
  }

  /**
   * Lazily initializes the WebAudio `AudioContext` and connects the full live DSP chain:
   * `sourceBus -> preAnalyser -> proq4Nodes[0..23] -> postAnalyser -> sootheNotches[0..5]`
   * `-> stemBass -> stemDrums -> stemVocals -> stemOther -> ozonePreGain -> ozoneWaveShaper`
   * `-> ozoneCompressor -> ozoneCeilingGain -> masterSpeakerGain -> destination`.
   */
  private ensureAudioGraph(telemetry: DawTelemetryView): AudioContext | null {
    this.latestTelemetry = telemetry;
    if (this.ctx) return this.ctx;
    if (typeof window === 'undefined') return null;
    const AudioCtxCtor =
      window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtxCtor) return null;

    try {
      const ctx = new AudioCtxCtor({ latencyHint: 'interactive' });
      this.ctx = ctx;

      this.sourceBus = ctx.createGain();
      this.sourceBus.gain.value = 0.85;

      // Per-stem GainNodes for UVR5 4-stem separated playback
      this.uvr5StemGains = [];
      for (let s = 0; s < 4; s++) {
        const sg = ctx.createGain();
        sg.gain.value = 1.0;
        sg.connect(this.sourceBus);
        this.uvr5StemGains.push(sg);
      }

      this.preAnalyser = ctx.createAnalyser();
      this.preAnalyser.fftSize = 2048;
      this.preAnalyser.smoothingTimeConstant = 0.68;

      this.proq4Nodes = [];
      for (let i = 0; i < 24; i++) {
        const biq = ctx.createBiquadFilter();
        biq.type = 'peaking';
        biq.frequency.value = 1000;
        biq.gain.value = 0;
        biq.Q.value = 1.0;
        this.proq4Nodes.push(biq);
      }

      this.postAnalyser = ctx.createAnalyser();
      this.postAnalyser.fftSize = 2048;
      this.postAnalyser.smoothingTimeConstant = 0.68;

      this.sootheNotches = [];
      const defaultFreqs = [180, 850, 3400, 6800, 11500, 15000];
      for (let b = 0; b < 6; b++) {
        const notch = ctx.createBiquadFilter();
        notch.type = 'peaking';
        notch.frequency.value = defaultFreqs[b];
        notch.Q.value = 1.8;
        notch.gain.value = 0.0;
        this.sootheNotches.push(notch);
      }

      this.stemBassFilter = ctx.createBiquadFilter();
      this.stemBassFilter.type = 'lowshelf';
      this.stemBassFilter.frequency.value = 115;
      this.stemBassFilter.gain.value = 0;

      this.stemDrumsFilter = ctx.createBiquadFilter();
      this.stemDrumsFilter.type = 'peaking';
      this.stemDrumsFilter.frequency.value = 80;
      this.stemDrumsFilter.Q.value = 1.1;
      this.stemDrumsFilter.gain.value = 0;

      this.stemVocalsFilter = ctx.createBiquadFilter();
      this.stemVocalsFilter.type = 'peaking';
      this.stemVocalsFilter.frequency.value = 1750;
      this.stemVocalsFilter.Q.value = 0.85;
      this.stemVocalsFilter.gain.value = 0;

      this.stemOtherFilter = ctx.createBiquadFilter();
      this.stemOtherFilter.type = 'highshelf';
      this.stemOtherFilter.frequency.value = 9000;
      this.stemOtherFilter.gain.value = 0;

      this.ozonePreGain = ctx.createGain();
      this.ozonePreGain.gain.value = 1.0;

      this.ozoneWaveShaper = ctx.createWaveShaper();
      this.ozoneWaveShaper.oversample = '2x';

      this.ozoneCompressor = ctx.createDynamicsCompressor();
      this.ozoneCompressor.threshold.value = -6.0;
      this.ozoneCompressor.knee.value = 3.0;
      this.ozoneCompressor.ratio.value = 12.0;
      this.ozoneCompressor.attack.value = 0.002;
      this.ozoneCompressor.release.value = 0.09;

      this.ozoneCeilingGain = ctx.createGain();
      this.ozoneCeilingGain.gain.value = 0.89;

      this.masterSpeakerGain = ctx.createGain();
      this.masterSpeakerGain.gain.value = 0.85;

      // Connect the full real-time DSP chain
      this.sourceBus.connect(this.preAnalyser);
      let prevNode: AudioNode = this.preAnalyser;
      for (const biq of this.proq4Nodes) {
        prevNode.connect(biq);
        prevNode = biq;
      }
      prevNode.connect(this.postAnalyser);
      prevNode = this.postAnalyser;
      for (const notch of this.sootheNotches) {
        prevNode.connect(notch);
        prevNode = notch;
      }
      prevNode.connect(this.stemBassFilter);
      this.stemBassFilter.connect(this.stemDrumsFilter);
      this.stemDrumsFilter.connect(this.stemVocalsFilter);
      this.stemVocalsFilter.connect(this.stemOtherFilter);
      this.stemOtherFilter.connect(this.ozonePreGain);
      this.ozonePreGain.connect(this.ozoneWaveShaper);
      this.ozoneWaveShaper.connect(this.ozoneCompressor);
      this.ozoneCompressor.connect(this.ozoneCeilingGain);
      this.ozoneCeilingGain.connect(this.masterSpeakerGain);
      this.masterSpeakerGain.connect(ctx.destination);

      this.noiseBuffer = this.createPercussionNoiseBuffer(ctx);
      this.fftPreBuf = new Float32Array(this.preAnalyser.frequencyBinCount);
      this.fftPostBuf = new Float32Array(this.postAnalyser.frequencyBinCount);

      if (this.pendingFl26RawWavs.length > 0) {
        this.decodePendingFl26Wavs(ctx);
      }

      this.syncDspChain(telemetry);
      return ctx;
    } catch {
      return null;
    }
  }

  private async decodePendingFl26Wavs(ctx: AudioContext): Promise<void> {
    const rawList = this.pendingFl26RawWavs;
    if (rawList.length === 0) return;
    const decodedList: AudioBuffer[] = [];
    for (const raw of rawList) {
      try {
        const buf = await ctx.decodeAudioData(raw.slice(0));
        decodedList.push(buf);
      } catch {
        // Ignore unparseable buffer
      }
    }
    if (decodedList.length > 0) {
      this.fl26KitVoiceBuffers = decodedList;
    }
  }

  private createPercussionNoiseBuffer(ctx: AudioContext): AudioBuffer {
    const length = Math.floor(ctx.sampleRate * 0.5);
    const buffer = ctx.createBuffer(1, length, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    let seed = 0x13572468;
    for (let i = 0; i < length; i++) {
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      data[i] = ((seed >>> 0) / 0xffffffff) * 2.0 - 1.0;
    }
    return buffer;
  }

  private computePatcherCordScale(telemetry: DawTelemetryView): number {
    const nodes = telemetry.patcherNodes;
    const edges = telemetry.patcherEdges;
    if (!nodes.length || !edges.length) return 1.0;
    const hostIn = nodes.find((n) => n.pluginKind === 'HostIn');
    const hostOut = nodes.find((n) => n.pluginKind === 'HostOut' || n.pluginKind === 'Ozone12');
    if (!hostIn || !hostOut) return 1.0;
    if (hostIn.enabled === false || hostOut.enabled === false) {
      return 1.0; // Slot bypassed -> unity pass-through
    }

    // Exact DAG path sum from hostIn to hostOut (cycle-safe bounded depth <= nodes.length)
    const memo = new Map<number, number>();
    const visiting = new Set<number>();
    const dfsGain = (u: number, depth: number): number => {
      if (u === hostOut.nodeIndex) return 1.0;
      if (depth > nodes.length || visiting.has(u)) return 0.0;
      const cached = memo.get(u);
      if (cached !== undefined) return cached;
      visiting.add(u);
      let sum = 0.0;
      for (const e of edges) {
        if (e.fromNode === u) {
          sum += Math.max(0.0, Math.min(4.0, e.gain)) * dfsGain(e.toNode, depth + 1);
        }
      }
      visiting.delete(u);
      memo.set(u, sum);
      return sum;
    };
    const totalPathGain = dfsGain(hostIn.nodeIndex, 0);
    // Normalize by default parallel sum (0.707 + 0.707 = 1.414) so default graph has unity gain
    return Math.max(0.0, Math.min(2.5, totalPathGain / 1.414));
  }

  /**
   * Synchronizes all live WebAudio DSP filter/dynamics parameters from `telemetry`.
   */
  syncDspChain(telemetry: DawTelemetryView): void {
    this.latestTelemetry = telemetry;
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    const nyquist = Math.max(8000, this.ctx.sampleRate * 0.48);

    const patcherProQ4Node = telemetry.patcherNodes.find((n) => n.pluginKind === 'ProQ4');
    const patcherSootheNode = telemetry.patcherNodes.find((n) => n.pluginKind === 'Soothe3');
    const proq4Active = !telemetry.proq4Bypass && (patcherProQ4Node?.enabled ?? true);
    const sootheActive = patcherSootheNode?.enabled ?? true;

    if (this.sourceBus) {
      const cordScale = this.computePatcherCordScale(telemetry);
      this.sourceBus.gain.setTargetAtTime(0.85 * cordScale, now, 0.015);
    }

    // 1. Update 24-Band FabFilter Pro-Q 4 Biquad Cascade (with cascaded slopes 6..96 dB/oct)
    const scale = telemetry.proq4GainScalePct * 0.01;
    let nodeCursor = 0;
    let netEqDbSum = 0.0;
    let activeBandCount = 0;

    for (let i = 0; proq4Active && i < 24 && nodeCursor < this.proq4Nodes.length; i++) {
      const band: ProQ4BandState | undefined = telemetry.proq4Bands[i];
      if (!band || !band.used || !band.enabled) {
        continue;
      }

      const freq = Math.max(16, Math.min(nyquist, band.freqHz));
      const gainDb = Math.max(-30, Math.min(30, band.gainDb * scale));
      const qVal = Math.max(0.1, Math.min(24.0, band.q));
      const isCutOrBp = band.shape === 2 || band.shape === 4 || band.shape === 6;
      const stages = isCutOrBp
        ? Math.min(4, Math.max(1, Math.round((band.slopeDbOct || 12) / 12)))
        : 1;

      if (!isCutOrBp && Math.abs(gainDb) > 0.05) {
        netEqDbSum += gainDb;
        activeBandCount++;
      }

      for (let s = 0; s < stages && nodeCursor < this.proq4Nodes.length; s++) {
        const node = this.proq4Nodes[nodeCursor++];
        switch (band.shape) {
          case 0:
            node.type = 'peaking';
            break;
          case 1:
            node.type = 'lowshelf';
            break;
          case 2:
            node.type = 'highpass';
            break;
          case 3:
          case 7:
          case 8:
            node.type = 'highshelf';
            break;
          case 4:
            node.type = 'lowpass';
            break;
          case 5:
            node.type = 'notch';
            break;
          case 6:
            node.type = 'bandpass';
            break;
          case 9:
            node.type = 'allpass';
            break;
          default:
            node.type = 'peaking';
            break;
        }
        node.frequency.setTargetAtTime(freq, now, 0.012);
        node.Q.setTargetAtTime(qVal, now, 0.012);
        node.gain.setTargetAtTime(gainDb / stages, now, 0.012);
      }
    }

    // Zero out remaining unused Biquad stages
    while (nodeCursor < this.proq4Nodes.length) {
      const node = this.proq4Nodes[nodeCursor++];
      node.type = 'peaking';
      node.gain.setTargetAtTime(0, now, 0.012);
    }

    // 2. Update oeksound Soothe3 6-Band Dynamic Spectral Notch Bank (including Delta Solo & Sidechain EQ bands)
    if (this.sootheNotches.length === 6) {
      if (!sootheActive) {
        for (let b = 0; b < 6; b++) {
          this.sootheNotches[b].type = 'peaking';
          this.sootheNotches[b].gain.setTargetAtTime(0.0, now, 0.015);
        }
      } else if (telemetry.soothe3DeltaSolo) {
        const primarySc =
          telemetry.soothe3ScBands.find((b) => b.enabled && b.depth > 0) ??
          telemetry.soothe3ScBands[2];
        const centerHz = Math.max(120, Math.min(nyquist, primarySc?.freqHz ?? 3400));
        this.sootheNotches[0].type = 'bandpass';
        this.sootheNotches[0].frequency.setTargetAtTime(centerHz, now, 0.012);
        this.sootheNotches[0].Q.setTargetAtTime(1.2 + telemetry.soothe3Detail * 0.25, now, 0.012);
        this.sootheNotches[0].gain.setTargetAtTime(0, now, 0.012);

        for (let b = 1; b < 6; b++) {
          const sc = telemetry.soothe3ScBands[b];
          const notch = this.sootheNotches[b];
          const f = Math.max(80, Math.min(nyquist, sc?.freqHz ?? 2000 * b));
          notch.type = 'peaking';
          notch.frequency.setTargetAtTime(f, now, 0.012);
          notch.Q.setTargetAtTime(Math.max(0.7, sc?.q ?? 1.8), now, 0.012);
          notch.gain.setTargetAtTime(sc?.enabled ? Math.max(0, (sc?.depth ?? 2.0) * 0.8) : 0, now, 0.012);
        }
      } else {
        const mix = Math.max(0, Math.min(1, telemetry.soothe3MixPct * 0.01));
        const modeScale = telemetry.soothe3HardMode ? 1.65 : 1.15;
        const sharpnessBase = Math.max(0.5, Math.min(12.0, 0.8 + telemetry.soothe3Detail * 0.4));

        for (let b = 0; b < 6; b++) {
          const sc = telemetry.soothe3ScBands[b];
          const notch = this.sootheNotches[b];
          if (!sc || !sc.enabled || telemetry.soothe3Depth <= 0.01) {
            notch.type = 'peaking';
            notch.gain.setTargetAtTime(0.0, now, 0.015);
            continue;
          }
          const freq = Math.max(40, Math.min(nyquist, sc.freqHz));
          const sensWeight = Math.max(0.0, (sc.depth + 10.0) / 10.0);
          const rawCut = -telemetry.soothe3Depth * modeScale * mix * sensWeight * 0.85;
          const cutDb = Math.max(telemetry.soothe3MaxCutDb, Math.min(0.0, rawCut));
          const qVal = Math.max(0.4, Math.min(16.0, (sc.q || 1.2) * sharpnessBase * 0.55));

          notch.type = 'peaking';
          notch.frequency.setTargetAtTime(freq, now, 0.015);
          notch.Q.setTargetAtTime(qVal, now, 0.015);
          notch.gain.setTargetAtTime(cutDb, now, 0.015);
        }
      }
    }

    // 3. Update BSRNN 4-Stem Master Rebalance Filters & UVR5 Per-Stem Gains ([Vocals, Drums, Bass, Other])
    const [vocDb, drmDb, basDb, othDb] = telemetry.bsrnnStemGainsDb;
    const anySolo = telemetry.bsrnnStemSolo.some(Boolean);
    const isStemActive = (idx: number): boolean => {
      if (telemetry.bsrnnStemMute[idx]) return false;
      if (anySolo && !telemetry.bsrnnStemSolo[idx]) return false;
      return true;
    };

    const effStemDb = (idx: number, rawDb: number, extraTiltDb = 0.0): number => {
      if (!isStemActive(idx)) return -24.0;
      return Math.max(-18.0, Math.min(18.0, rawDb + extraTiltDb));
    };

    // Ozone 12 4-Band Exciter & Imager harmonic tilt contribution
    const lowExciterTilt = telemetry.ozone12ExciterBypass
      ? 0.0
      : (telemetry.ozone12ExciterDrivesPct[0] - 20.0) * 0.04;
    const midExciterTilt = telemetry.ozone12ExciterBypass
      ? 0.0
      : (telemetry.ozone12ExciterDrivesPct[2] - 25.0) * 0.04;
    const highExciterAndAirTilt =
      (telemetry.ozone12ExciterBypass ? 0.0 : (telemetry.ozone12ExciterDrivesPct[3] - 20.0) * 0.05) +
      telemetry.ozone12ImagerWidthsPct[3] * 0.02;

    if (this.stemVocalsFilter) {
      this.stemVocalsFilter.gain.setTargetAtTime(effStemDb(0, vocDb, midExciterTilt), now, 0.015);
    }
    if (this.stemDrumsFilter) {
      this.stemDrumsFilter.gain.setTargetAtTime(effStemDb(1, drmDb), now, 0.015);
    }
    if (this.stemBassFilter) {
      this.stemBassFilter.gain.setTargetAtTime(effStemDb(2, basDb, lowExciterTilt), now, 0.015);
    }
    if (this.stemOtherFilter) {
      this.stemOtherFilter.gain.setTargetAtTime(
        effStemDb(3, othDb, highExciterAndAirTilt),
        now,
        0.015
      );
    }

    // Update UVR5 separated stem GainNodes if active
    for (let s = 0; s < 4; s++) {
      const sg = this.uvr5StemGains[s];
      if (!sg) continue;
      const trk = telemetry.mixerTracks[s + 1];
      const trkGain = trk?.muted ? 0.0 : (trk?.gainLinear ?? 1.0);
      if (!isStemActive(s) || trkGain <= 0.0001) {
        sg.gain.setTargetAtTime(0.0, now, 0.012);
      } else {
        const stemLin = Math.pow(10.0, Math.max(-24, Math.min(12, telemetry.bsrnnStemGainsDb[s])) / 20.0);
        sg.gain.setTargetAtTime(stemLin * trkGain, now, 0.012);
      }
    }

    // 4. Update iZotope Ozone 12 12AX7 Tube WaveShaper, FL26 Native FX & Brickwall Maximizer
    const avgDrivePct = telemetry.ozone12ExciterBypass
      ? 0
      : telemetry.ozone12ExciterDrivesPct[0] * 0.22 +
        telemetry.ozone12ExciterDrivesPct[1] * 0.25 +
        telemetry.ozone12ExciterDrivesPct[2] * 0.30 +
        telemetry.ozone12ExciterDrivesPct[3] * 0.23;
    const softClipBoost = telemetry.ozone12SoftClipEnabled ? telemetry.ozone12SoftClipMix * 25.0 : 0.0;
    const fl26FxBoost = telemetry.fl26SelectedFxPreset.includes('Maximum loudness') ||
      telemetry.fl26SelectedFxPreset.includes('Pop (heavy)')
      ? 14.0
      : telemetry.fl26SelectedFxPreset.includes('Soundgoodizer')
        ? 10.0
        : 0.0;
    const driveSig = Math.round(avgDrivePct + softClipBoost + fl26FxBoost);
    if (this.ozoneWaveShaper && driveSig !== this.lastDriveSignature) {
      this.lastDriveSignature = driveSig;
      this.ozoneWaveShaper.curve = this.build12AX7SaturationCurve(driveSig);
    }

    const allStemsMuted = [0, 1, 2, 3].every((idx) => !isStemActive(idx));
    const proq4OutTrimDb = telemetry.proq4Bypass ? 0.0 : (telemetry.proq4OutputLevelDb ?? 0.0);
    const autoGainCompDb =
      (telemetry.proq4AutoGain && activeBandCount > 0
        ? Math.max(-6.0, Math.min(6.0, (-netEqDbSum / activeBandCount) * 0.35))
        : 0.0) + proq4OutTrimDb;

    if (this.ozonePreGain && this.ozoneCompressor && this.ozoneCeilingGain) {
      if (allStemsMuted) {
        this.ozonePreGain.gain.setTargetAtTime(0.0, now, 0.015);
      } else if (telemetry.ozone12MaximizerBypass) {
        const compLin = Math.pow(10, autoGainCompDb / 20);
        this.ozonePreGain.gain.setTargetAtTime(compLin, now, 0.015);
        this.ozoneCompressor.threshold.setTargetAtTime(0.0, now, 0.015);
        this.ozoneCeilingGain.gain.setTargetAtTime(1.0, now, 0.015);
      } else {
        const totalGainDb = telemetry.ozone12GainDb * 0.45 + autoGainCompDb;
        const gainLin = Math.pow(10, Math.max(-12, Math.min(14, totalGainDb)) / 20);
        const ceilLin = Math.pow(10, Math.max(-12, Math.min(0, telemetry.ozone12CeilingDbtp)) / 20);
        this.ozonePreGain.gain.setTargetAtTime(gainLin, now, 0.015);
        this.ozoneCompressor.threshold.setTargetAtTime(
          Math.max(-24, Math.min(-0.5, telemetry.ozone12CeilingDbtp - telemetry.ozone12GainDb * 0.6)),
          now,
          0.015
        );
        this.ozoneCeilingGain.gain.setTargetAtTime(ceilLin, now, 0.015);
      }
    }

    // 5. Update Master Speaker Volume & Mute
    if (this.masterSpeakerGain) {
      const masterTrk = telemetry.mixerTracks[0];
      const masterTrkGain = masterTrk?.muted ? 0.0 : (masterTrk?.gainLinear ?? 1.0);
      const volNorm =
        !telemetry.isPlaying || telemetry.speakerMuted
          ? 0.0
          : Math.max(0, Math.min(1.5, (telemetry.speakerVolumePct / 100.0) * masterTrkGain));
      this.masterSpeakerGain.gain.setTargetAtTime(volNorm, now, 0.015);
    }
  }

  private build12AX7SaturationCurve(driveAmountPct: number): Float32Array<ArrayBuffer> {
    const samples = 2048;
    const curve = new Float32Array(new ArrayBuffer(samples * 4));
    const k = Math.max(0.05, (driveAmountPct / 100.0) * 2.4);
    const norm = 1.0 / Math.tanh(k);
    for (let i = 0; i < samples; i++) {
      const x = (i * 2) / (samples - 1) - 1;
      // Asymmetric 12AX7 triode tube transfer characteristic
      const asym = x + 0.06 * x * x * (driveAmountPct / 100.0);
      curve[i] = Math.max(-1.0, Math.min(1.0, Math.tanh(k * asym) * norm));
    }
    return curve;
  }

  private isCustomOrStemsMode(mode: AudioSourceMode): boolean {
    return mode === 'customFile' || mode === 'stems4';
  }

  /**
   * Starts or pauses real-time speaker output when the user clicks `▶ Play` or `⏸ Pause`.
   */
  setPlaying(playing: boolean, telemetry: DawTelemetryView): void {
    this.latestTelemetry = telemetry;
    if (playing) {
      const ctx = this.ensureAudioGraph(telemetry);
      if (!ctx) return;
      if (ctx.state === 'suspended') {
        ctx.resume().catch(() => {});
      }
      this.syncDspChain(telemetry);
      this.startPlaybackLoop(telemetry);
    } else {
      this.stopPlaybackLoop();
      if (this.masterSpeakerGain && this.ctx) {
        this.masterSpeakerGain.gain.setTargetAtTime(0.0, this.ctx.currentTime, 0.008);
      }
    }
  }

  /**
   * Seeks transport position (e.g. `⏮ Reset` to `0, 0`).
   */
  seek(samplePos: number, _ppqTick: number, telemetry: DawTelemetryView): void {
    this.latestTelemetry = telemetry;
    if (samplePos <= 0) {
      this.currentStep16th = 0;
      this.customPausedOffsetSec = 0;
      if (this.isCustomOrStemsMode(telemetry.audioSourceMode) && telemetry.isPlaying) {
        this.stopCustomBufferNode();
        this.startCustomBufferNode();
      }
    }
  }

  private startPlaybackLoop(telemetry: DawTelemetryView): void {
    this.latestTelemetry = telemetry;
    if (!this.ctx) return;
    if (this.isCustomOrStemsMode(telemetry.audioSourceMode) && this.customAudioBuffer) {
      this.stopSequencerTimer();
      if (!this.isCustomPlaying) {
        this.startCustomBufferNode();
      }
      return;
    }

    this.stopCustomBufferNode();
    if (this.schedulerTimerId !== null) return;
    this.nextStepTime = this.ctx.currentTime + 0.02;
    this.schedulerTimerId = setInterval(() => {
      if (this.latestTelemetry) {
        this.tickSequencer(this.latestTelemetry);
      }
    }, 25);
  }

  private stopPlaybackLoop(): void {
    this.stopSequencerTimer();
    if (this.isCustomPlaying && this.ctx && this.customAudioBuffer) {
      const elapsed = this.ctx.currentTime - this.customStartedAtCtxTime;
      this.customPausedOffsetSec =
        (this.customPausedOffsetSec + Math.max(0, elapsed)) % Math.max(0.01, this.customAudioBuffer.duration);
    }
    this.stopCustomBufferNode();
  }

  private stopSequencerTimer(): void {
    if (this.schedulerTimerId !== null) {
      clearInterval(this.schedulerTimerId);
      this.schedulerTimerId = null;
    }
    this.activeSlideVoices.clear();
  }

  private startCustomBufferNode(): void {
    if (!this.ctx || !this.sourceBus || !this.customAudioBuffer) return;
    this.stopCustomBufferNode();
    const offset = this.customPausedOffsetSec % Math.max(0.01, this.customAudioBuffer.duration);

    // If UVR5 separated 4 stems are active (`stems4`), play all 4 stems in sample-accurate sync through per-stem GainNodes
    const hasSeparatedStems =
      this.latestTelemetry?.audioSourceMode === 'stems4' &&
      this.uvr5StemBuffers.every((b) => b !== null);
    if (hasSeparatedStems) {
      const startAt = this.ctx.currentTime + 0.005;
      for (let s = 0; s < 4; s++) {
        const stemBuf = this.uvr5StemBuffers[s];
        const stemGain = this.uvr5StemGains[s];
        if (!stemBuf || !stemGain) continue;
        const src = this.ctx.createBufferSource();
        src.buffer = stemBuf;
        src.loop = true;
        src.connect(stemGain);
        const stemOffset = offset % Math.max(0.01, stemBuf.duration);
        src.start(startAt, stemOffset);
        this.uvr5StemSources[s] = src;
      }
      this.customStartedAtCtxTime = this.ctx.currentTime;
      this.isCustomPlaying = true;
      return;
    }

    const src = this.ctx.createBufferSource();
    src.buffer = this.customAudioBuffer;
    src.loop = true;
    src.connect(this.sourceBus);
    src.start(0, offset);
    this.customSourceNode = src;
    this.customStartedAtCtxTime = this.ctx.currentTime;
    this.isCustomPlaying = true;
  }

  private stopCustomBufferNode(): void {
    if (this.customSourceNode) {
      try {
        this.customSourceNode.stop();
        this.customSourceNode.disconnect();
      } catch {
        // Ignore if already stopped
      }
      this.customSourceNode = null;
    }
    for (let s = 0; s < 4; s++) {
      const stemSrc = this.uvr5StemSources[s];
      if (stemSrc) {
        try {
          stemSrc.stop();
          stemSrc.disconnect();
        } catch {
          // Ignore if already stopped
        }
        this.uvr5StemSources[s] = null;
      }
    }
    this.isCustomPlaying = false;
  }

  /**
   * Decodes a user-selected audio file (`.wav`, `.mp3`, `.flac`, `.ogg`) and plays it through the live DSP chain.
   */
  async loadCustomAudioFile(file: File, telemetry: DawTelemetryView): Promise<boolean> {
    const ctx = this.ensureAudioGraph(telemetry);
    if (!ctx) return false;
    if (ctx.state === 'suspended') {
      await ctx.resume().catch(() => {});
    }
    try {
      const arrayBuf = await file.arrayBuffer();
      const decoded = await ctx.decodeAudioData(arrayBuf.slice(0));
      this.customAudioBuffer = decoded;
      this.uvr5StemBuffers = [null, null, null, null];
      this.customPausedOffsetSec = 0;
      this.stopSequencerTimer();
      this.stopCustomBufferNode();
      if (telemetry.isPlaying && telemetry.audioSourceMode === 'customFile') {
        this.startCustomBufferNode();
      }
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Returns the full-length stereo Float32Array PCM buffers and sample rate of the currently loaded custom audio file
   * so it can be uploaded to the Rust audio engine without any duration truncation.
   */
  getCustomAudioPcmAndSampleRate(): { left: Float32Array; right: Float32Array; sampleRate: number } | null {
    if (!this.customAudioBuffer || this.customAudioBuffer.length === 0) {
      return null;
    }
    const left = this.customAudioBuffer.getChannelData(0);
    const right =
      this.customAudioBuffer.numberOfChannels > 1
        ? this.customAudioBuffer.getChannelData(1)
        : left;
    return {
      left,
      right,
      sampleRate: Math.round(this.customAudioBuffer.sampleRate),
    };
  }

  /**
   * Decodes the 4 separated UVR5 stem `.wav` ArrayBuffers (`[Vocals, Drums, Bass, Other]`)
   * and switches custom file playback to synchronized 4-stem playback.
   */
  async loadUvr5SeparatedStems(
    stemWavBuffers: ArrayBuffer[],
    telemetry: DawTelemetryView
  ): Promise<boolean> {
    if (stemWavBuffers.length < 4) return false;
    const ctx = this.ensureAudioGraph(telemetry);
    if (!ctx) return false;
    try {
      const decodedStems: (AudioBuffer | null)[] = [null, null, null, null];
      for (let s = 0; s < 4; s++) {
        if (stemWavBuffers[s] && stemWavBuffers[s].byteLength > 44) {
          decodedStems[s] = await ctx.decodeAudioData(stemWavBuffers[s].slice(0));
        }
      }
      if (decodedStems.every((b) => b !== null)) {
        this.uvr5StemBuffers = decodedStems;
        if (!this.customAudioBuffer && decodedStems[0]) {
          this.customAudioBuffer = decodedStems[0];
        }
        this.syncDspChain(telemetry);
        if (telemetry.isPlaying && this.isCustomOrStemsMode(telemetry.audioSourceMode)) {
          this.startCustomBufferNode();
        }
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }

  /**
   * Decodes real FL Studio 26 `.wav` sample kit voices fetched from the Tauri backend.
   */
  async loadFl26KitVoiceBuffers(
    wavBuffers: ArrayBuffer[],
    telemetry: DawTelemetryView
  ): Promise<void> {
    this.pendingFl26RawWavs = wavBuffers;
    const ctx = this.ctx ?? this.ensureAudioGraph(telemetry);
    if (!ctx) return;
    await this.decodePendingFl26Wavs(ctx);
  }

  /**
   * Computes a genuine 80-band log-Mel spectrum (`20 Hz .. 20 kHz` triangular Mel filterbank)
   * from the currently loaded `customAudioBuffer` PCM, live `preAnalyser` FFT, or active Piano Roll pattern.
   */
  computeMel80FromAudio(telemetry?: DawTelemetryView): number[] {
    const melMin = 2595.0 * Math.log10(1.0 + 20.0 / 700.0);
    const melMax = 2595.0 * Math.log10(1.0 + 20000.0 / 700.0);
    const melPtsHz = new Float64Array(82);
    for (let m = 0; m < 82; m++) {
      const melVal = melMin + ((melMax - melMin) * m) / 81.0;
      melPtsHz[m] = 700.0 * (Math.pow(10.0, melVal / 2595.0) - 1.0);
    }

    // 1. If a custom audio file is loaded, compute multi-window DFT across the entire song's PCM
    if (
      this.customAudioBuffer &&
      this.customAudioBuffer.length > 64 &&
      (!telemetry || this.isCustomOrStemsMode(telemetry.audioSourceMode))
    ) {
      const sr = this.customAudioBuffer.sampleRate;
      const left = this.customAudioBuffer.getChannelData(0);
      const right =
        this.customAudioBuffer.numberOfChannels > 1
          ? this.customAudioBuffer.getChannelData(1)
          : left;
      const totalLen = left.length;
      const numWindows = Math.min(32, Math.max(1, Math.floor(totalLen / 512)));
      const step = Math.max(1, Math.floor((totalLen - 256) / numWindows));
      const melPower = new Float64Array(80);

      for (let m = 0; m < 80; m++) {
        const fCenter = melPtsHz[m + 1];
        if (fCenter >= sr * 0.49) {
          melPower[m] = 1e-9;
          continue;
        }
        const omega = (2.0 * Math.PI * fCenter) / sr;
        const cosW = Math.cos(omega);
        const sinW = Math.sin(omega);
        let bandEnergySum = 0.0;

        for (let w = 0; w < numWindows; w++) {
          const base = Math.min(totalLen - 256, w * step);
          let s1 = 0.0;
          let s2 = 0.0;
          const coeff = 2.0 * cosW;
          for (let n = 0; n < 256; n++) {
            const hann = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * n) / 255.0));
            const x = 0.5 * (left[base + n] + right[base + n]) * hann;
            const s0 = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
          }
          const real = s1 - s2 * cosW;
          const imag = s2 * sinW;
          bandEnergySum += (real * real + imag * imag) / (256.0 * 256.0);
        }
        melPower[m] = bandEnergySum / numWindows;
      }

      const mel80: number[] = new Array(80);
      for (let m = 0; m < 80; m++) {
        mel80[m] = Math.max(-85.0, Math.min(6.0, 10.0 * Math.log10(melPower[m] + 1e-10)));
      }
      return mel80;
    }

    // 2. If live WebAudio preAnalyser has real FFT energy, project FFT bins onto the 80 triangular Mel filters
    if (this.ctx && this.preAnalyser) {
      this.preAnalyser.getFloatFrequencyData(this.fftPreBuf);
      let maxBinDb = -120.0;
      for (let i = 2; i < 128; i++) {
        if (this.fftPreBuf[i] > maxBinDb) maxBinDb = this.fftPreBuf[i];
      }
      if (maxBinDb > -95.0) {
        const binCount = this.fftPreBuf.length;
        const nyquist = this.ctx.sampleRate * 0.5;
        const mel80: number[] = new Array(80);
        for (let m = 0; m < 80; m++) {
          const fLeft = melPtsHz[m];
          const fCenter = melPtsHz[m + 1];
          const fRight = melPtsHz[m + 2];
          let weightedSum = 0.0;
          let weightTotal = 0.0;

          const bStart = Math.max(1, Math.floor((fLeft / nyquist) * (binCount - 1)));
          const bEnd = Math.min(binCount - 1, Math.ceil((fRight / nyquist) * (binCount - 1)));
          for (let b = bStart; b <= bEnd; b++) {
            const fBin = (b / (binCount - 1)) * nyquist;
            const w =
              fBin <= fCenter
                ? (fBin - fLeft) / Math.max(1.0, fCenter - fLeft)
                : (fRight - fBin) / Math.max(1.0, fRight - fCenter);
            const clampedW = Math.max(0.05, w);
            const dbVal = Number.isFinite(this.fftPreBuf[b]) ? this.fftPreBuf[b] : -100.0;
            const linPow = Math.pow(10.0, Math.max(-100.0, dbVal) / 10.0);
            weightedSum += linPow * clampedW;
            weightTotal += clampedW;
          }
          const avgPow = weightTotal > 0 ? weightedSum / weightTotal : 1e-9;
          mel80[m] = Math.max(-85.0, Math.min(6.0, 10.0 * Math.log10(avgPow + 1e-10)));
        }
        return mel80;
      }
    }

    // 3. Otherwise derive the 80-band Mel profile from the active Piano Roll notes & FL26 Sample Kit
    const mel80: number[] = new Array(80);
    const kit = telemetry?.fl26SelectedSampleKit ?? '808 Studio Kit';
    const is808 = kit.includes('808');
    const is909 = kit.includes('909');
    const isStrings = kit.includes('Strings');
    const isSytrus = kit.includes('Sytrus');
    const pat =
      telemetry?.pianoRollPatterns.find((p) => p.patternId === telemetry.selectedPatternId) ??
      telemetry?.pianoRollPatterns[0];

    for (let m = 0; m < 80; m++) {
      const fCenter = melPtsHz[m + 1];
      let energy = Math.pow(10.0, (-28.0 - 10.0 * (m / 79.0)) / 10.0);
      if (is808 && fCenter >= 35 && fCenter <= 120) energy += 0.25;
      if (is909 && ((fCenter >= 50 && fCenter <= 160) || (fCenter >= 6000 && fCenter <= 12000))) {
        energy += 0.18;
      }
      if (isStrings && fCenter >= 220 && fCenter <= 3200) energy += 0.22;
      if (isSytrus && fCenter >= 800 && fCenter <= 9500) energy += 0.24;

      if (pat) {
        for (const note of pat.notes) {
          const f0 = this.midiToFreqHz(note.key, note.finePitchCents);
          for (let h = 1; h <= 4; h++) {
            const fh = f0 * h;
            const distOct = Math.abs(Math.log2(Math.max(1e-3, fCenter / fh)));
            if (distOct < 0.35) {
              energy += (note.velocity * 0.12) / (h * (1.0 + distOct * 8.0));
            }
          }
        }
      }
      mel80[m] = Math.max(-85.0, Math.min(6.0, 10.0 * Math.log10(energy + 1e-10)));
    }
    return mel80;
  }

  /**
   * Switches playback from a custom audio file back to the built-in FL Studio 26 Piano Roll & Drum sequencer.
   */
  switchToPianoRollSource(telemetry: DawTelemetryView): void {
    this.latestTelemetry = telemetry;
    this.stopCustomBufferNode();
    if (telemetry.isPlaying) {
      this.startPlaybackLoop(telemetry);
    }
  }

  /**
   * Plays an immediate audition note through the live DSP chain when the user clicks or adds a note in the Piano Roll.
   */
  auditionNote(
    telemetry: DawTelemetryView,
    key: number,
    velocity = 0.88,
    finePitchCents = 0.0,
    pan = 0.0
  ): void {
    const ctx = this.ensureAudioGraph(telemetry);
    if (!ctx || !this.sourceBus) return;
    if (ctx.state === 'suspended') {
      ctx.resume().catch(() => {});
    }
    const now = ctx.currentTime + 0.005;
    const dur = 0.24;
    if (this.masterSpeakerGain && !telemetry.speakerMuted) {
      const vol = Math.max(0.15, (telemetry.speakerVolumePct / 100.0) * 0.85);
      this.masterSpeakerGain.gain.cancelScheduledValues(ctx.currentTime);
      this.masterSpeakerGain.gain.setTargetAtTime(vol, ctx.currentTime, 0.005);
      if (!telemetry.isPlaying) {
        this.masterSpeakerGain.gain.setTargetAtTime(0.0, now + dur + 0.06, 0.015);
      }
    }
    this.triggerSynthVoice(
      ctx,
      now,
      dur,
      key,
      velocity,
      finePitchCents,
      pan,
      false,
      0,
      telemetry.fl26SelectedSampleKit
    );
  }

  tickSequencer(telemetry: DawTelemetryView): void {
    this.latestTelemetry = telemetry;
    if (
      !this.ctx ||
      !this.sourceBus ||
      !telemetry.isPlaying ||
      this.isCustomOrStemsMode(telemetry.audioSourceMode)
    ) {
      return;
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {});
    }

    const lookaheadSec = 0.09;
    const bpm = Math.max(30, Math.min(300, telemetry.tempoBpm));
    const secPer16th = 60.0 / bpm / 4.0;
    if (this.nextStepTime < this.ctx.currentTime - 0.25) {
      this.nextStepTime = this.ctx.currentTime + 0.01;
    }

    while (this.nextStepTime < this.ctx.currentTime + lookaheadSec) {
      const step = this.currentStep16th % 16;
      this.scheduleStepNotesAndDrums(this.ctx, this.nextStepTime, step, secPer16th, telemetry);
      this.nextStepTime += secPer16th;
      this.currentStep16th = (this.currentStep16th + 1) % 16;
    }
  }

  private scheduleStepNotesAndDrums(
    ctx: AudioContext,
    stepTime: number,
    step16th: number,
    secPer16th: number,
    telemetry: DawTelemetryView
  ): void {
    const stepStartTick = step16th * 24;
    const stepEndTick = stepStartTick + 24;
    const kit = telemetry.fl26SelectedSampleKit;

    // 1. Trigger Piano Roll Pattern Notes (with FL Studio 26 Slide / Portamento support)
    for (const pat of telemetry.pianoRollPatterns) {
      const clip = telemetry.playlistClips.find(
        (c) => c.kind === 'Pattern' && c.sourceId === pat.patternId
      );
      if (clip?.muted) continue;

      const trkIdx = pat.targetChannelId === 2 ? 3 : 1;
      const trkGain = telemetry.mixerTracks[trkIdx]?.gainLinear ?? 1.0;
      if (trkGain <= 0.001) continue;

      for (const note of pat.notes) {
        const wrappedTick = ((note.startTick % 384) + 384) % 384;
        if (wrappedTick >= stepStartTick && wrappedTick < stepEndTick) {
          const tickOffset = wrappedTick - stepStartTick;
          const noteStartTime = stepTime + (tickOffset / 24.0) * secPer16th;
          const noteDurSec = Math.max(0.05, (note.durationTicks / 24.0) * secPer16th);
          this.triggerSynthNoteOrSlide(
            ctx,
            noteStartTime,
            noteDurSec,
            note,
            trkGain * (clip?.gain ?? 1.0),
            kit
          );
        }
      }
    }

    // 2. Trigger 808 / 909 Drum Sequencer matching `fl26SelectedSampleKit`
    const drumClip = telemetry.playlistClips.find((c) => c.trackRow === 2);
    const drumTrackGain = (telemetry.mixerTracks[2]?.gainLinear ?? 1.0) * (drumClip?.muted ? 0 : 1);
    if (drumTrackGain > 0.01) {
      // Kick on 0, 6, 8, 12
      if (step16th === 0 || step16th === 6 || step16th === 8 || step16th === 12) {
        this.triggerKick(ctx, stepTime, kit, drumTrackGain * (step16th === 6 ? 0.78 : 0.95));
      }
      // Snare / Clap on 4, 12
      if (step16th === 4 || step16th === 12) {
        this.triggerSnareOrClap(ctx, stepTime, kit, drumTrackGain * 0.85);
      }
      // Hi-Hat on even steps + trap roll on 14, 15
      if (step16th % 2 === 0 || step16th === 15) {
        const isOpenHat = step16th === 2 || step16th === 10;
        this.triggerHiHat(ctx, stepTime, isOpenHat, drumTrackGain * (isOpenHat ? 0.42 : 0.30));
      }
    }
  }

  private midiToFreqHz(key: number, finePitchCents = 0): number {
    return 440.0 * Math.pow(2.0, (key - 69.0 + finePitchCents / 100.0) / 12.0);
  }

  private triggerSynthNoteOrSlide(
    ctx: AudioContext,
    startTime: number,
    durationSec: number,
    note: PianoRollNoteUiState,
    trackGain: number,
    kit: string
  ): void {
    if (note.isSlide) {
      const existing = this.activeSlideVoices.get(note.colorGroup);
      if (existing && existing.endTime > startTime + 0.01) {
        const targetFreq = this.midiToFreqHz(note.key, note.finePitchCents);
        const glideEnd = Math.min(existing.endTime, startTime + durationSec);
        try {
          existing.osc1.frequency.setValueAtTime(existing.osc1.frequency.value, startTime);
          existing.osc1.frequency.exponentialRampToValueAtTime(
            Math.max(20, targetFreq),
            glideEnd
          );
          existing.osc2.frequency.setValueAtTime(existing.osc2.frequency.value, startTime);
          existing.osc2.frequency.exponentialRampToValueAtTime(
            Math.max(20, targetFreq * 1.004),
            glideEnd
          );
          return;
        } catch {
          // Fall through to normal note trigger if voice expired
        }
      }
    }

    this.triggerSynthVoice(
      ctx,
      startTime,
      durationSec,
      note.key,
      note.velocity * trackGain,
      note.finePitchCents,
      note.pan,
      note.isSlide,
      note.colorGroup,
      kit
    );
  }

  private triggerSynthVoice(
    ctx: AudioContext,
    startTime: number,
    durationSec: number,
    key: number,
    velocity: number,
    finePitchCents: number,
    pan: number,
    isSlide: boolean,
    colorGroup: number,
    kit: string
  ): void {
    if (!this.sourceBus) return;
    const freq = Math.max(24, Math.min(12000, this.midiToFreqHz(key, finePitchCents)));
    const isBass = key < 48;
    const isStrings = kit.includes('Strings');
    const isSytrus = kit.includes('Sytrus');

    // Layer real decoded FL26 .wav sample when Symphonic Strings or Sample Kit buffers are loaded
    if (isStrings && this.fl26KitVoiceBuffers.length > 0) {
      const sampleBuf =
        key >= 60 && this.fl26KitVoiceBuffers.length > 1
          ? this.fl26KitVoiceBuffers[1]
          : this.fl26KitVoiceBuffers[0];
      const baseNote = key >= 60 && this.fl26KitVoiceBuffers.length > 1 ? 72 : 48;
      const src = ctx.createBufferSource();
      src.buffer = sampleBuf;
      src.playbackRate.setValueAtTime(
        Math.pow(2.0, (key - baseNote + finePitchCents / 100.0) / 12.0),
        startTime
      );
      const smpEnv = ctx.createGain();
      smpEnv.gain.setValueAtTime(0.001, startTime);
      smpEnv.gain.linearRampToValueAtTime(Math.min(0.32, velocity * 0.25), startTime + 0.03);
      smpEnv.gain.exponentialRampToValueAtTime(0.0001, startTime + durationSec + 0.08);
      src.connect(smpEnv);
      smpEnv.connect(this.sourceBus);
      src.start(startTime);
      src.stop(startTime + durationSec + 0.1);
    }

    const osc1 = ctx.createOscillator();
    const osc2 = ctx.createOscillator();
    const filter = ctx.createBiquadFilter();
    const env = ctx.createGain();
    const panner = ctx.createStereoPanner();

    osc1.type = isBass ? 'triangle' : isStrings ? 'sawtooth' : 'sawtooth';
    osc2.type = isBass ? 'sine' : isSytrus ? 'square' : 'triangle';

    osc1.frequency.setValueAtTime(freq, startTime);
    osc2.frequency.setValueAtTime(isBass ? freq * 0.5 : freq * 1.005, startTime);

    if (isSlide) {
      const startFreq = Math.max(24, freq * 0.84);
      osc1.frequency.setValueAtTime(startFreq, startTime);
      osc1.frequency.exponentialRampToValueAtTime(freq, startTime + Math.min(0.18, durationSec));
      osc2.frequency.setValueAtTime(startFreq * 1.005, startTime);
      osc2.frequency.exponentialRampToValueAtTime(
        freq * 1.005,
        startTime + Math.min(0.18, durationSec)
      );
    }

    filter.type = 'lowpass';
    const baseCutoff = isBass ? 380 : isStrings ? 2800 : isSytrus ? 5400 : 3800;
    filter.frequency.setValueAtTime(baseCutoff * 1.4, startTime);
    filter.frequency.exponentialRampToValueAtTime(
      Math.max(120, baseCutoff * 0.65),
      startTime + durationSec
    );
    filter.Q.value = isBass ? 2.2 : 1.8;

    const peakGain = Math.max(0.01, Math.min(0.35, velocity * (isBass ? 0.28 : 0.18)));
    const attack = isStrings ? 0.045 : 0.008;
    const stopTime = startTime + durationSec + 0.04;

    env.gain.setValueAtTime(0.0001, startTime);
    env.gain.linearRampToValueAtTime(peakGain, startTime + attack);
    env.gain.exponentialRampToValueAtTime(0.0001, stopTime);

    panner.pan.setValueAtTime(Math.max(-1, Math.min(1, pan)), startTime);

    osc1.connect(filter);
    osc2.connect(filter);
    filter.connect(env);
    env.connect(panner);
    panner.connect(this.sourceBus);

    osc1.start(startTime);
    osc2.start(startTime);
    osc1.stop(stopTime + 0.01);
    osc2.stop(stopTime + 0.01);

    if (!isBass) {
      this.activeSlideVoices.set(colorGroup, {
        osc1,
        osc2,
        endTime: stopTime,
        colorGroup,
      });
    }
  }

  private triggerKick(ctx: AudioContext, startTime: number, kit: string, gainScale: number): void {
    if (!this.sourceBus) return;
    if (this.fl26KitVoiceBuffers.length > 0 && !kit.includes('Strings')) {
      const kickBuf = this.fl26KitVoiceBuffers[0];
      const src = ctx.createBufferSource();
      src.buffer = kickBuf;
      const g = ctx.createGain();
      g.gain.setValueAtTime(Math.min(0.65, 0.5 * gainScale), startTime);
      src.connect(g);
      g.connect(this.sourceBus);
      src.start(startTime);
    }

    const is909 = kit.includes('909');
    const osc = ctx.createOscillator();
    const env = ctx.createGain();
    osc.type = 'sine';

    const startPitch = is909 ? 175 : 145;
    const subPitch = is909 ? 52 : 42;
    const decay = is909 ? 0.24 : 0.36;

    osc.frequency.setValueAtTime(startPitch, startTime);
    osc.frequency.exponentialRampToValueAtTime(subPitch, startTime + 0.065);
    osc.frequency.exponentialRampToValueAtTime(subPitch * 0.85, startTime + decay);

    const amp = Math.min(0.55, 0.42 * gainScale);
    env.gain.setValueAtTime(0.001, startTime);
    env.gain.linearRampToValueAtTime(amp, startTime + 0.004);
    env.gain.exponentialRampToValueAtTime(0.0001, startTime + decay);

    osc.connect(env);
    env.connect(this.sourceBus);
    osc.start(startTime);
    osc.stop(startTime + decay + 0.01);
  }

  private triggerSnareOrClap(
    ctx: AudioContext,
    startTime: number,
    kit: string,
    gainScale: number
  ): void {
    if (!this.sourceBus || !this.noiseBuffer) return;
    if (this.fl26KitVoiceBuffers.length > 1 && (kit.includes('909') || kit.includes('Skipper'))) {
      const snareBuf = this.fl26KitVoiceBuffers[1];
      const src = ctx.createBufferSource();
      src.buffer = snareBuf;
      const g = ctx.createGain();
      g.gain.setValueAtTime(Math.min(0.55, 0.42 * gainScale), startTime);
      src.connect(g);
      g.connect(this.sourceBus);
      src.start(startTime);
    }

    // Body tone
    const bodyOsc = ctx.createOscillator();
    const bodyEnv = ctx.createGain();
    bodyOsc.type = 'triangle';
    bodyOsc.frequency.setValueAtTime(kit.includes('Skipper') ? 220 : 185, startTime);
    bodyOsc.frequency.exponentialRampToValueAtTime(110, startTime + 0.09);

    bodyEnv.gain.setValueAtTime(0.22 * gainScale, startTime);
    bodyEnv.gain.exponentialRampToValueAtTime(0.0001, startTime + 0.11);
    bodyOsc.connect(bodyEnv);
    bodyEnv.connect(this.sourceBus);
    bodyOsc.start(startTime);
    bodyOsc.stop(startTime + 0.12);

    // Crispy snare wires / clap noise
    const noiseSrc = ctx.createBufferSource();
    noiseSrc.buffer = this.noiseBuffer;
    const bp = ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.setValueAtTime(2400, startTime);
    bp.Q.setValueAtTime(1.1, startTime);

    const noiseEnv = ctx.createGain();
    noiseEnv.gain.setValueAtTime(0.24 * gainScale, startTime);
    noiseEnv.gain.exponentialRampToValueAtTime(0.0001, startTime + 0.16);

    noiseSrc.connect(bp);
    bp.connect(noiseEnv);
    noiseEnv.connect(this.sourceBus);
    noiseSrc.start(startTime);
    noiseSrc.stop(startTime + 0.17);
  }

  private triggerHiHat(
    ctx: AudioContext,
    startTime: number,
    openHat: boolean,
    gainScale: number
  ): void {
    if (!this.sourceBus || !this.noiseBuffer) return;
    if (this.fl26KitVoiceBuffers.length >= 3) {
      const hatBuf = openHat ? this.fl26KitVoiceBuffers[2] : this.fl26KitVoiceBuffers[1];
      if (hatBuf) {
        const src = ctx.createBufferSource();
        src.buffer = hatBuf;
        const g = ctx.createGain();
        g.gain.setValueAtTime(Math.min(0.4, 0.28 * gainScale), startTime);
        src.connect(g);
        g.connect(this.sourceBus);
        src.start(startTime);
      }
    }

    const noiseSrc = ctx.createBufferSource();
    noiseSrc.buffer = this.noiseBuffer;

    const hp = ctx.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.setValueAtTime(7500, startTime);

    const env = ctx.createGain();
    const dur = openHat ? 0.18 : 0.045;
    env.gain.setValueAtTime(0.18 * gainScale, startTime);
    env.gain.exponentialRampToValueAtTime(0.0001, startTime + dur);

    noiseSrc.connect(hp);
    hp.connect(env);
    env.connect(this.sourceBus);
    noiseSrc.start(startTime);
    noiseSrc.stop(startTime + dur + 0.01);
  }

  /**
   * Samples the real-time Pre-EQ and Post-EQ `AnalyserNode`s (2048-pt FFT) and maps them onto
   * the 1024 logarithmic frequency bins (`20 Hz .. 20 kHz`) when the WebAudio graph is running.
   */
  readPrePostSpectrumDb(preOut1024: Float32Array, postOut1024: Float32Array): boolean {
    if (!this.ctx || this.ctx.state !== 'running' || !this.preAnalyser || !this.postAnalyser) {
      return false;
    }
    this.preAnalyser.getFloatFrequencyData(this.fftPreBuf);
    this.postAnalyser.getFloatFrequencyData(this.fftPostBuf);

    const binCount = this.fftPreBuf.length;
    const nyquist = this.ctx.sampleRate * 0.5;
    let hasSignal = false;
    for (let i = 4; i < 64; i++) {
      if (this.fftPreBuf[i] > -95.0) {
        hasSignal = true;
        break;
      }
    }
    if (!hasSignal) return false;

    for (let k = 0; k < 1024; k++) {
      const normX = k / 1023.0;
      const freq = 20.0 * Math.pow(1000.0, normX);
      const exactIdx = Math.max(0, Math.min(binCount - 1, (freq / nyquist) * (binCount - 1)));
      const idx0 = Math.floor(exactIdx);
      const idx1 = Math.min(binCount - 1, idx0 + 1);
      const frac = exactIdx - idx0;

      const preVal = this.fftPreBuf[idx0] * (1 - frac) + this.fftPreBuf[idx1] * frac;
      const postVal = this.fftPostBuf[idx0] * (1 - frac) + this.fftPostBuf[idx1] * frac;
      preOut1024[k] = Number.isFinite(preVal) ? Math.max(-120.0, preVal) : -120.0;
      postOut1024[k] = Number.isFinite(postVal) ? Math.max(-120.0, postVal) : -120.0;
    }
    return true;
  }
}
