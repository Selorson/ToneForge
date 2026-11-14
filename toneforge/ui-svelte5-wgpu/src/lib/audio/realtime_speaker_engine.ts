// Uncolored Standalone-Browser Raw PCM Fallback (`realtime_speaker_engine.ts`)
//
// HARD ARCHITECTURAL CONTRACT (ToneForge v2 Studio Mastering & Real-Time DAW):
// - Inside Tauri Desktop (`__TAURI_INTERNALS__` / `__TAURI__`), this module is 100% HARD-BYPASSED:
//   `window.AudioContext` is NEVER instantiated (preventing WebView2 from holding a Windows Audiosrv
//   shared session that would block WASAPI Exclusive Mode `AUDCLNT_E_DEVICE_IN_USE`), and 100% of
//   audible sound comes exclusively from the Rust Lock-Free DSP/AI Engine via Windows WASAPI hardware.
// - All simulated WebAudio EQ / Soothe3 / BSRNN / Ozone 12 WaveShaper / Compressor nodes have been
//   permanently retired. Only an uncolored raw PCM (`sourceBus -> rawAnalyser -> masterSpeakerGain`)
//   preview path remains strictly for standalone web browser (`http://localhost:5173`) inspection.

import type { DawTelemetryView, PianoRollNoteUiState } from '../state/daw_master_store.svelte';

export type AudioSourceMode = 'pianoRoll' | 'customFile' | 'stems4';

export function isDesktopTauriRuntime(): boolean {
  return (
    typeof window !== 'undefined' &&
    ('__TAURI_INTERNALS__' in window || '__TAURI__' in window)
  );
}

interface ActiveSlideVoice {
  osc1: OscillatorNode;
  osc2: OscillatorNode;
  endTime: number;
  colorGroup: number;
}

export class RealtimeSpeakerEngine {
  private ctx: AudioContext | null = null;
  private sourceBus: GainNode | null = null;
  private rawAnalyser: AnalyserNode | null = null;
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

  private fftRawBuf = new Float32Array(1024);
  private unlockListenersAttached = false;
  private latestTelemetry: DawTelemetryView | null = null;

  constructor() {
    if (!isDesktopTauriRuntime()) {
      this.attachAutoplayUnlockListeners();
    }
  }

  private attachAutoplayUnlockListeners(): void {
    if (isDesktopTauriRuntime() || typeof window === 'undefined' || this.unlockListenersAttached) {
      return;
    }
    this.unlockListenersAttached = true;

    const unlock = () => {
      if (isDesktopTauriRuntime() || !this.latestTelemetry) return;
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
   * Standalone pure-browser fallback ONLY: initializes an uncolored raw PCM graph
   * (`sourceBus -> rawAnalyser -> masterSpeakerGain -> destination`).
   * Strictly returns `null` inside Tauri Desktop so WebView2 never touches Windows Audio!
   */
  private ensureAudioGraph(telemetry: DawTelemetryView): AudioContext | null {
    this.latestTelemetry = telemetry;
    if (isDesktopTauriRuntime()) {
      return null;
    }
    if (this.ctx) return this.ctx;
    if (typeof window === 'undefined') return null;
    const AudioCtxCtor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtxCtor) return null;

    try {
      const ctx = new AudioCtxCtor({ latencyHint: 'interactive' });
      this.ctx = ctx;

      this.sourceBus = ctx.createGain();
      this.sourceBus.gain.value = 0.85;

      this.uvr5StemGains = [];
      for (let s = 0; s < 4; s++) {
        const sg = ctx.createGain();
        sg.gain.value = 1.0;
        sg.connect(this.sourceBus);
        this.uvr5StemGains.push(sg);
      }

      this.rawAnalyser = ctx.createAnalyser();
      this.rawAnalyser.fftSize = 2048;
      this.rawAnalyser.smoothingTimeConstant = 0.68;

      this.masterSpeakerGain = ctx.createGain();
      this.masterSpeakerGain.gain.value = 0.85;

      // Uncolored unity pass-through: ZERO simulated EQ, Soothe3, BSRNN, WaveShaper, or Compressor nodes
      this.sourceBus.connect(this.rawAnalyser);
      this.rawAnalyser.connect(this.masterSpeakerGain);
      this.masterSpeakerGain.connect(ctx.destination);

      this.noiseBuffer = this.createPercussionNoiseBuffer(ctx);
      this.fftRawBuf = new Float32Array(this.rawAnalyser.frequencyBinCount);

      if (this.pendingFl26RawWavs.length > 0) {
        this.decodePendingFl26Wavs(ctx);
      }

      this.syncDspChain(telemetry);
      return ctx;
    } catch {
      return null;
    }
  }

  private async decodePendingFl26Wavs(ctx: BaseAudioContext): Promise<void> {
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

  private createPercussionNoiseBuffer(ctx: BaseAudioContext): AudioBuffer {
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
   * In Tauri Desktop: 100% hard-bypassed (all DSP & volume run in Rust WASAPI).
   * In Standalone Browser: updates only uncolored master speaker gain & stem mute/solo gains.
   */
  syncDspChain(telemetry: DawTelemetryView): void {
    this.latestTelemetry = telemetry;
    if (isDesktopTauriRuntime() || !this.ctx) return;

    const now = this.ctx.currentTime;
    const anySolo = telemetry.bsrnnStemSolo.some(Boolean);
    const isStemActive = (idx: number): boolean => {
      if (telemetry.bsrnnStemMute[idx]) return false;
      if (anySolo && !telemetry.bsrnnStemSolo[idx]) return false;
      return true;
    };

    for (let s = 0; s < 4; s++) {
      const sg = this.uvr5StemGains[s];
      if (!sg) continue;
      const trk = telemetry.mixerTracks[s + 1];
      const trkGain = trk?.muted ? 0.0 : (trk?.gainLinear ?? 1.0);
      if (!isStemActive(s) || trkGain <= 0.0001) {
        sg.gain.setTargetAtTime(0.0, now, 0.012);
      } else {
        const stemLin = Math.pow(
          10.0,
          Math.max(-24, Math.min(12, telemetry.bsrnnStemGainsDb[s])) / 20.0
        );
        sg.gain.setTargetAtTime(stemLin * trkGain, now, 0.012);
      }
    }

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

  private isCustomOrStemsMode(mode: AudioSourceMode): boolean {
    return mode === 'customFile' || mode === 'stems4';
  }

  setPlaying(playing: boolean, telemetry: DawTelemetryView): void {
    this.latestTelemetry = telemetry;
    if (isDesktopTauriRuntime()) return;

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

  seek(samplePos: number, _ppqTick: number, telemetry: DawTelemetryView): void {
    this.latestTelemetry = telemetry;
    if (isDesktopTauriRuntime()) return;

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
    if (isDesktopTauriRuntime() || !this.ctx) return;

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
        (this.customPausedOffsetSec + Math.max(0, elapsed)) %
        Math.max(0.01, this.customAudioBuffer.duration);
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
    if (isDesktopTauriRuntime() || !this.ctx || !this.sourceBus || !this.customAudioBuffer) return;
    this.stopCustomBufferNode();
    const offset = this.customPausedOffsetSec % Math.max(0.01, this.customAudioBuffer.duration);

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
   * Decodes a browser `File` object:
   * - In Tauri Desktop (fallback only when `File.path` is unavailable): uses a transient
   *   `OfflineAudioContext` so NO Windows audio endpoint / Audiosrv session is ever opened!
   * - In Standalone Browser: decodes and queues on the uncolored raw PCM fallback graph.
   */
  async loadCustomAudioFile(file: File, telemetry: DawTelemetryView): Promise<boolean> {
    const arrayBuf = await file.arrayBuffer();
    if (isDesktopTauriRuntime()) {
      if (typeof OfflineAudioContext === 'undefined') return false;
      try {
        const offline = new OfflineAudioContext(2, 1, 48000);
        this.customAudioBuffer = await offline.decodeAudioData(arrayBuf.slice(0));
        return true;
      } catch {
        return false;
      }
    }
    return this.loadCustomAudioArrayBuffer(arrayBuf, telemetry);
  }

  async loadCustomAudioArrayBuffer(
    audioBytes: ArrayBuffer,
    telemetry: DawTelemetryView
  ): Promise<boolean> {
    if (isDesktopTauriRuntime()) {
      return false;
    }
    const ctx = this.ensureAudioGraph(telemetry);
    if (!ctx) return false;
    if (ctx.state === 'suspended') {
      await ctx.resume().catch(() => {});
    }
    try {
      const decoded = await ctx.decodeAudioData(audioBytes.slice(0));
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

  getCustomAudioPcmAndSampleRate(): {
    left: Float32Array;
    right: Float32Array;
    sampleRate: number;
  } | null {
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

  async loadUvr5SeparatedStems(
    stemWavBuffers: ArrayBuffer[],
    telemetry: DawTelemetryView
  ): Promise<boolean> {
    if (isDesktopTauriRuntime() || stemWavBuffers.length < 4) return false;
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

  async loadFl26KitVoiceBuffers(
    wavBuffers: ArrayBuffer[],
    telemetry: DawTelemetryView
  ): Promise<void> {
    if (isDesktopTauriRuntime()) return;
    this.pendingFl26RawWavs = wavBuffers;
    const ctx = this.ctx ?? this.ensureAudioGraph(telemetry);
    if (!ctx) return;
    await this.decodePendingFl26Wavs(ctx);
  }

  /**
   * Computes an 80-band log-Mel spectrum (`20 Hz .. 20 kHz`):
   * - In Tauri Desktop: projects the genuine Rust 1024-bin `proq4PreDb` shared-memory spectrum
   *   (while the Rust backend `cmd_run_ozone12_ai_master_assistant` also scans the full song PCM directly).
   * - In Standalone Browser: computes from `customAudioBuffer` or `rawAnalyser`.
   */
  computeMel80FromAudio(telemetry?: DawTelemetryView): number[] {
    const melMin = 2595.0 * Math.log10(1.0 + 20.0 / 700.0);
    const melMax = 2595.0 * Math.log10(1.0 + 20000.0 / 700.0);
    const melPtsHz = new Float64Array(82);
    for (let m = 0; m < 82; m++) {
      const melVal = melMin + ((melMax - melMin) * m) / 81.0;
      melPtsHz[m] = 700.0 * (Math.pow(10.0, melVal / 2595.0) - 1.0);
    }

    // 1. In Tauri Desktop, read directly from the Rust engine's 1024-bin log spectrum (`proq4PreDb`)
    if (telemetry && isDesktopTauriRuntime()) {
      const pre = telemetry.proq4PreDb;
      let maxDb = -120.0;
      for (let i = 0; i < pre.length; i++) {
        if (pre[i] > maxDb) maxDb = pre[i];
      }
      if (maxDb > -105.0) {
        const mel80: number[] = new Array(80);
        for (let m = 0; m < 80; m++) {
          const fLeft = Math.max(20.0, melPtsHz[m]);
          const fCenter = Math.max(20.0, melPtsHz[m + 1]);
          const fRight = Math.min(20000.0, melPtsHz[m + 2]);
          const kStart = Math.max(
            0,
            Math.floor((Math.log(fLeft / 20.0) / Math.log(1000.0)) * 1023.0)
          );
          const kEnd = Math.min(
            1023,
            Math.ceil((Math.log(Math.max(fLeft + 1.0, fRight) / 20.0) / Math.log(1000.0)) * 1023.0)
          );
          let weightedSum = 0.0;
          let weightTotal = 0.0;
          for (let k = kStart; k <= kEnd; k++) {
            const fk = 20.0 * Math.pow(1000.0, k / 1023.0);
            const w =
              fk <= fCenter
                ? (fk - fLeft) / Math.max(1.0, fCenter - fLeft)
                : (fRight - fk) / Math.max(1.0, fRight - fCenter);
            const cw = Math.max(0.05, w);
            const dbVal = Number.isFinite(pre[k]) ? pre[k] : -100.0;
            weightedSum += Math.pow(10.0, Math.max(-100.0, dbVal) / 10.0) * cw;
            weightTotal += cw;
          }
          const avgPow = weightTotal > 0 ? weightedSum / weightTotal : 1e-9;
          mel80[m] = Math.max(-85.0, Math.min(6.0, 10.0 * Math.log10(avgPow + 1e-10)));
        }
        return mel80;
      }
    }

    // 2. Standalone browser custom buffer DFT
    if (
      !isDesktopTauriRuntime() &&
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

    // 3. Harmonic profile from active Piano Roll pattern
    const mel80: number[] = new Array(80);
    const pat =
      telemetry?.pianoRollPatterns.find((p) => p.patternId === telemetry.selectedPatternId) ??
      telemetry?.pianoRollPatterns[0];
    for (let m = 0; m < 80; m++) {
      const fCenter = melPtsHz[m + 1];
      let energy = Math.pow(10.0, (-30.0 - 10.0 * (m / 79.0)) / 10.0);
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

  switchToPianoRollSource(telemetry: DawTelemetryView): void {
    this.latestTelemetry = telemetry;
    if (isDesktopTauriRuntime()) return;
    this.stopCustomBufferNode();
    if (telemetry.isPlaying) {
      this.startPlaybackLoop(telemetry);
    }
  }

  auditionNote(
    telemetry: DawTelemetryView,
    key: number,
    velocity = 0.88,
    finePitchCents = 0.0,
    pan = 0.0
  ): void {
    if (isDesktopTauriRuntime()) return;
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
      isDesktopTauriRuntime() ||
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

    const drumClip = telemetry.playlistClips.find((c) => c.trackRow === 2);
    const drumTrackGain = (telemetry.mixerTracks[2]?.gainLinear ?? 1.0) * (drumClip?.muted ? 0 : 1);
    if (drumTrackGain > 0.01) {
      if (step16th === 0 || step16th === 6 || step16th === 8 || step16th === 12) {
        this.triggerKick(ctx, stepTime, kit, drumTrackGain * (step16th === 6 ? 0.78 : 0.95));
      }
      if (step16th === 4 || step16th === 12) {
        this.triggerSnareOrClap(ctx, stepTime, kit, drumTrackGain * 0.85);
      }
      if (step16th % 2 === 0 || step16th === 15) {
        const isOpenHat = step16th === 2 || step16th === 10;
        this.triggerHiHat(ctx, stepTime, isOpenHat, drumTrackGain * (isOpenHat ? 0.42 : 0.3));
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
          // Fall through
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
    const env = ctx.createGain();
    const panner = ctx.createStereoPanner();

    osc1.type = isBass ? 'triangle' : 'sawtooth';
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

    const peakGain = Math.max(0.01, Math.min(0.3, velocity * (isBass ? 0.24 : 0.15)));
    const attack = isStrings ? 0.045 : 0.008;
    const stopTime = startTime + durationSec + 0.04;

    env.gain.setValueAtTime(0.0001, startTime);
    env.gain.linearRampToValueAtTime(peakGain, startTime + attack);
    env.gain.exponentialRampToValueAtTime(0.0001, stopTime);
    panner.pan.setValueAtTime(Math.max(-1, Math.min(1, pan)), startTime);

    osc1.connect(env);
    osc2.connect(env);
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
      return;
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
      return;
    }

    const noiseSrc = ctx.createBufferSource();
    noiseSrc.buffer = this.noiseBuffer;
    const noiseEnv = ctx.createGain();
    noiseEnv.gain.setValueAtTime(0.22 * gainScale, startTime);
    noiseEnv.gain.exponentialRampToValueAtTime(0.0001, startTime + 0.14);
    noiseSrc.connect(noiseEnv);
    noiseEnv.connect(this.sourceBus);
    noiseSrc.start(startTime);
    noiseSrc.stop(startTime + 0.15);
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
        return;
      }
    }

    const noiseSrc = ctx.createBufferSource();
    noiseSrc.buffer = this.noiseBuffer;
    const env = ctx.createGain();
    const dur = openHat ? 0.18 : 0.045;
    env.gain.setValueAtTime(0.14 * gainScale, startTime);
    env.gain.exponentialRampToValueAtTime(0.0001, startTime + dur);
    noiseSrc.connect(env);
    env.connect(this.sourceBus);
    noiseSrc.start(startTime);
    noiseSrc.stop(startTime + dur + 0.01);
  }

  /**
   * Standalone pure-browser raw FFT reader ONLY.
   * Always returns `false` inside Tauri Desktop so `DawTelemetryView` displays 100% authentic
   * Rust 128 KB `VisualTelemetryFrame` spectrum & reduction telemetry without browser blending!
   */
  readPrePostSpectrumDb(preOut1024: Float32Array, postOut1024: Float32Array): boolean {
    if (
      isDesktopTauriRuntime() ||
      !this.ctx ||
      this.ctx.state !== 'running' ||
      !this.rawAnalyser
    ) {
      return false;
    }
    this.rawAnalyser.getFloatFrequencyData(this.fftRawBuf);

    const binCount = this.fftRawBuf.length;
    const nyquist = this.ctx.sampleRate * 0.5;
    let hasSignal = false;
    for (let i = 4; i < 64; i++) {
      if (this.fftRawBuf[i] > -95.0) {
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

      const rawVal = this.fftRawBuf[idx0] * (1 - frac) + this.fftRawBuf[idx1] * frac;
      const clamped = Number.isFinite(rawVal) ? Math.max(-120.0, rawVal) : -120.0;
      preOut1024[k] = clamped;
      postOut1024[k] = clamped;
    }
    return true;
  }
}
