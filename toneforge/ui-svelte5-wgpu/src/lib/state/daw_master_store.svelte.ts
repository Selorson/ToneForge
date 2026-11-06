// ui-svelte5-wgpu/src/lib/state/daw_master_store.svelte.ts
import { invoke } from '@tauri-apps/api/core';

export class DawTelemetryView {
  private buffer: ArrayBuffer;
  private f32View: Float32Array;
  private f64View: Float64Array;
  private u32View: Uint32Array;

  // Svelte 5 Fine-Grained Reactive Runes for Low-Frequency Transport State
  isPlaying = $state(false);
  tempoBpm = $state(128.0);
  playheadPpq = $state(0.0);
  cpuLoadPercent = $state(0.0);
  xrunCount = $state(0);

  constructor(sharedBuffer: ArrayBuffer) {
    if (sharedBuffer.byteLength !== 131072) {
      throw new Error(`Invalid VisualTelemetryFrame size: ${sharedBuffer.byteLength} (expected 131072)`);
    }
    this.buffer = sharedBuffer;
    this.f32View = new Float32Array(this.buffer);
    this.f64View = new Float64Array(this.buffer);
    this.u32View = new Uint32Array(this.buffer);
  }

  /** Called once per requestAnimationFrame (120 Hz) */
  syncHeaderFromSharedMemory(): void {
    this.playheadPpq = this.f64View[1];       // offset 0x0008
    this.tempoBpm = this.f64View[2];          // offset 0x0010
    this.isPlaying = this.u32View[7] !== 0;   // offset 0x001C
    this.cpuLoadPercent = this.f32View[8];    // offset 0x0020
    this.xrunCount = this.u32View[9];         // offset 0x0024
  }

  /** Zero-copy TypedArray views passed directly to WebGPU queue.writeBuffer() */
  getMixerPeaksLeft(): Float32Array {
    return new Float32Array(this.buffer, 0x0040, 512);
  }

  getMixerPeaksRight(): Float32Array {
    return new Float32Array(this.buffer, 0x0840, 512);
  }

  getProQ4SpectrumDb(): Float32Array {
    return new Float32Array(this.buffer, 0x4040, 2048);
  }

  getSoothe3GainReductionDb(): Float32Array {
    return new Float32Array(this.buffer, 0xA040, 2048);
  }

  async setTransport(playing: boolean, bpm: number): Promise<void> {
    this.isPlaying = playing;
    this.tempoBpm = bpm;
    await invoke('cmd_set_transport', { playing, bpm });
  }
}
