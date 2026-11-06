// Real-Time WebAudio Speaker Engine & Live Audible DSP Chain (`realtime_speaker_engine.ts`)
// Outputs real stereo audio to Windows speakers/headphones when `isPlaying === true`:
// - Source 1: Built-in FL Studio 26 Polyphonic Synth (with Slide/Portamento glides) + 808/909 Drum Sequencer
// - Source 2: Custom Audio File Picker (`.wav`, `.mp3`, `.flac`, `.ogg`) decoded via `AudioContext.decodeAudioData`
// - Live DSP Chain: Pre-EQ AnalyserNode -> 24-Band FabFilter Pro-Q 4 Biquad Cascade -> Post-EQ AnalyserNode
//   -> oeksound Soothe3 Resonance Filter (with Delta Solo bandpass mode) -> BSRNN 4-Stem Rebalance EQ
//   -> iZotope Ozone 12 12AX7 Tube WaveShaper & Brickwall Maximizer -> Master Speaker Volume GainNode.

import type { DawTelemetryView, PianoRollNoteUiState, ProQ4BandState } from '../state/daw_master_store.svelte';

export type AudioSourceMode = 'pianoRoll' | 'customFile';

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
  private sootheFilter1: BiquadFilterNode | null = null;
  private sootheFilter2: BiquadFilterNode | null = null;
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
   * `sourceBus -> preAnalyser -> proq4Nodes[0..23] -> postAnalyser -> sootheFilter1 -> sootheFilter2`
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

      this.sootheFilter1 = ctx.createBiquadFilter();
      this.sootheFilter1.type = 'peaking';
      this.sootheFilter1.frequency.value = 3400;
      this.sootheFilter1.Q.value = 2.2;
      this.sootheFilter1.gain.value = -4.0;

      this.sootheFilter2 = ctx.createBiquadFilter();
      this.sootheFilter2.type = 'peaking';
      this.sootheFilter2.frequency.value = 6800;
      this.sootheFilter2.Q.value = 1.8;
      this.sootheFilter2.gain.value = -3.0;

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
      this.postAnalyser.connect(this.sootheFilter1);
      this.sootheFilter1.connect(this.sootheFilter2);
      this.sootheFilter2.connect(this.stemBassFilter);
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

      this.syncDspChain(telemetry);
      return ctx;
    } catch {
      return null;
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

  /**
   * Synchronizes all live WebAudio DSP filter/dynamics parameters from `telemetry`.
   */
  syncDspChain(telemetry: DawTelemetryView): void {
    this.latestTelemetry = telemetry;
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    const nyquist = Math.max(8000, this.ctx.sampleRate * 0.48);

    // 1. Update 24-Band FabFilter Pro-Q 4 Biquad Cascade
    const scale = telemetry.proq4GainScalePct * 0.01;
    for (let i = 0; i < 24; i++) {
      const node = this.proq4Nodes[i];
      const band: ProQ4BandState | undefined = telemetry.proq4Bands[i];
      if (!node) continue;
      if (!band || !band.used || !band.enabled) {
        node.type = 'peaking';
        node.gain.setTargetAtTime(0, now, 0.012);
        continue;
      }

      const freq = Math.max(16, Math.min(nyquist, band.freqHz));
      const gainDb = Math.max(-30, Math.min(30, band.gainDb * scale));
      const qVal = Math.max(0.1, Math.min(24.0, band.q));

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
      node.gain.setTargetAtTime(gainDb, now, 0.012);
    }

    // 2. Update oeksound Soothe3 Resonance Suppressor (including Delta Solo bandpass mode)
    if (this.sootheFilter1 && this.sootheFilter2) {
      if (telemetry.soothe3DeltaSolo) {
        this.sootheFilter1.type = 'bandpass';
        this.sootheFilter1.frequency.setTargetAtTime(3600, now, 0.012);
        this.sootheFilter1.Q.setTargetAtTime(1.4 + telemetry.soothe3Detail * 0.25, now, 0.012);
        this.sootheFilter1.gain.setTargetAtTime(0, now, 0.012);

        this.sootheFilter2.type = 'peaking';
        this.sootheFilter2.frequency.setTargetAtTime(5200, now, 0.012);
        this.sootheFilter2.Q.setTargetAtTime(1.8, now, 0.012);
        this.sootheFilter2.gain.setTargetAtTime(6.0, now, 0.012);
      } else {
        const mix = Math.max(0, Math.min(1, telemetry.soothe3MixPct * 0.01));
        const modeScale = telemetry.soothe3HardMode ? 1.65 : 1.15;
        const cut1 = Math.max(
          telemetry.soothe3MaxCutDb,
          -telemetry.soothe3Depth * modeScale * mix
        );
        const cut2 = Math.max(
          telemetry.soothe3MaxCutDb,
          -telemetry.soothe3Depth * modeScale * 0.82 * mix
        );
        const f1 = Math.max(200, Math.min(nyquist, telemetry.soothe3ScBands[2]?.freqHz ?? 3400));
        const f2 = Math.max(400, Math.min(nyquist, telemetry.soothe3ScBands[3]?.freqHz ?? 6800));
        const sharpnessQ = Math.max(0.7, Math.min(12.0, 0.9 + telemetry.soothe3Detail * 0.45));

        this.sootheFilter1.type = 'peaking';
        this.sootheFilter1.frequency.setTargetAtTime(f1, now, 0.015);
        this.sootheFilter1.Q.setTargetAtTime(sharpnessQ, now, 0.015);
        this.sootheFilter1.gain.setTargetAtTime(cut1, now, 0.015);

        this.sootheFilter2.type = 'peaking';
        this.sootheFilter2.frequency.setTargetAtTime(f2, now, 0.015);
        this.sootheFilter2.Q.setTargetAtTime(sharpnessQ * 0.85, now, 0.015);
        this.sootheFilter2.gain.setTargetAtTime(cut2, now, 0.015);
      }
    }

    // 3. Update BSRNN 4-Stem Master Rebalance Filters ([Vocals, Drums, Bass, Other])
    const [vocDb, drmDb, basDb, othDb] = telemetry.bsrnnStemGainsDb;
    const anySolo = telemetry.bsrnnStemSolo.some(Boolean);
    const effStemDb = (idx: number, rawDb: number): number => {
      if (telemetry.bsrnnStemMute[idx]) return -18.0;
      if (anySolo && !telemetry.bsrnnStemSolo[idx]) return -15.0;
      return Math.max(-12.0, Math.min(12.0, rawDb));
    };

    if (this.stemVocalsFilter) {
      this.stemVocalsFilter.gain.setTargetAtTime(effStemDb(0, vocDb), now, 0.015);
    }
    if (this.stemDrumsFilter) {
      this.stemDrumsFilter.gain.setTargetAtTime(effStemDb(1, drmDb), now, 0.015);
    }
    if (this.stemBassFilter) {
      this.stemBassFilter.gain.setTargetAtTime(effStemDb(2, basDb), now, 0.015);
    }
    if (this.stemOtherFilter) {
      this.stemOtherFilter.gain.setTargetAtTime(effStemDb(3, othDb), now, 0.015);
    }

    // 4. Update iZotope Ozone 12 12AX7 Tube WaveShaper & Brickwall Maximizer
    const avgDrivePct = telemetry.ozone12ExciterBypass
      ? 0
      : (telemetry.ozone12ExciterDrivesPct[0] +
          telemetry.ozone12ExciterDrivesPct[1] +
          telemetry.ozone12ExciterDrivesPct[2] +
          telemetry.ozone12ExciterDrivesPct[3]) *
        0.25;
    const softClipBoost = telemetry.ozone12SoftClipEnabled ? telemetry.ozone12SoftClipMix * 25.0 : 0.0;
    const driveSig = Math.round(avgDrivePct + softClipBoost);
    if (this.ozoneWaveShaper && driveSig !== this.lastDriveSignature) {
      this.lastDriveSignature = driveSig;
      this.ozoneWaveShaper.curve = this.build12AX7SaturationCurve(driveSig);
    }

    if (this.ozonePreGain && this.ozoneCompressor && this.ozoneCeilingGain) {
      if (telemetry.ozone12MaximizerBypass) {
        this.ozonePreGain.gain.setTargetAtTime(1.0, now, 0.015);
        this.ozoneCompressor.threshold.setTargetAtTime(0.0, now, 0.015);
        this.ozoneCeilingGain.gain.setTargetAtTime(1.0, now, 0.015);
      } else {
        const gainLin = Math.pow(10, Math.max(-12, Math.min(14, telemetry.ozone12GainDb * 0.45)) / 20);
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
      if (telemetry.audioSourceMode === 'customFile' && telemetry.isPlaying) {
        this.stopCustomBufferNode();
        this.startCustomBufferNode();
      }
    }
  }

  private startPlaybackLoop(telemetry: DawTelemetryView): void {
    this.latestTelemetry = telemetry;
    if (!this.ctx) return;
    if (telemetry.audioSourceMode === 'customFile' && this.customAudioBuffer) {
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
    const src = this.ctx.createBufferSource();
    src.buffer = this.customAudioBuffer;
    src.loop = true;
    src.connect(this.sourceBus);
    const offset = this.customPausedOffsetSec % Math.max(0.01, this.customAudioBuffer.duration);
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
    // Temporarily ensure speaker gain is audible for the audition note even if transport is paused
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
      telemetry.audioSourceMode === 'customFile'
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
