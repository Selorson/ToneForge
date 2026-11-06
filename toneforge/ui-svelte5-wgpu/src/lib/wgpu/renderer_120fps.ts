// 120 FPS WebGPU Multi-Pipeline Renderer (`renderer_120fps.ts`)
// Uploads Zero-Copy Float32Array views from the 128 KB `VisualTelemetryFrame` directly to GPU
// buffers via `queue.writeBuffer()` and executes `proq4_curve.wgsl`, `soothe3_heatmap.wgsl`,
// and `mixer_meters.wgsl` at 120 Hz.

import {
  DawTelemetryView,
  OFFSET_OZONE12_TELEMETRY,
  OFFSET_PROQ4_SPECTRUM,
  OFFSET_SOOTHE3_REDUCTION,
  OFFSET_SPECTROGRAM_RING,
  OFFSET_TRACK_METERS,
} from '../state/daw_master_store.svelte';
import proq4CurveWgsl from './shaders/proq4_curve.wgsl?raw';
import soothe3HeatmapWgsl from './shaders/soothe3_heatmap.wgsl?raw';
import mixerMetersWgsl from './shaders/mixer_meters.wgsl?raw';

export type ViewportShaderMode = 'proq4' | 'soothe3' | 'mixer_meters';

export class WebGpu120FpsRenderer {
  private canvas: HTMLCanvasElement;
  private mode: ViewportShaderMode;
  private device: any = null;
  private context: any = null;
  private pipeline: any = null;
  private computePipeline: any = null;
  private bindGroup: any = null;
  private computeBindGroup: any = null;
  private uniformBuffer: any = null;
  private storageBufferA: any = null;
  private storageBufferB: any = null;
  private storageBufferC: any = null;
  private isWebGpuReady = false;

  constructor(canvas: HTMLCanvasElement, mode: ViewportShaderMode) {
    this.canvas = canvas;
    this.mode = mode;
  }

  async initialize(): Promise<boolean> {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      this.isWebGpuReady = false;
      return false;
    }
    try {
      const gpu = (navigator as any).gpu;
      const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return false;
      this.device = await adapter.requestDevice();
      this.context = this.canvas.getContext('webgpu');
      if (!this.context) return false;

      const format = gpu.getPreferredCanvasFormat();
      this.context.configure({
        device: this.device,
        format,
        alphaMode: 'opaque',
      });

      const GPUBufferUsage = (globalThis as any).GPUBufferUsage ?? {
        COPY_DST: 0x0008,
        UNIFORM: 0x0040,
        STORAGE: 0x0080,
      };
      if (this.mode === 'proq4') {
        const module = this.device.createShaderModule({ code: proq4CurveWgsl });
        this.uniformBuffer = this.device.createBuffer({
          size: 3088, // BiquadCascadeUniforms (4 + 96*4 + 96*4 floats)
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.storageBufferA = this.device.createBuffer({
          size: 16384, // ProQ4SpectrumFrame (0x03000..0x07000)
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.storageBufferB = this.device.createBuffer({
          size: 8192, // out_magnitude_db[2048]
          usage: GPUBufferUsage.STORAGE,
        });
        this.storageBufferC = this.device.createBuffer({
          size: 8192, // out_phase_rad[2048]
          usage: GPUBufferUsage.STORAGE,
        });
        this.computePipeline = this.device.createComputePipeline({
          layout: 'auto',
          compute: { module, entryPoint: 'cs_eval_proq4_curve' },
        });
        this.computeBindGroup = this.device.createBindGroup({
          layout: this.computePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.uniformBuffer } },
            { binding: 2, resource: { buffer: this.storageBufferB } },
            { binding: 3, resource: { buffer: this.storageBufferC } },
          ],
        });
        this.pipeline = this.device.createRenderPipeline({
          layout: 'auto',
          vertex: { module, entryPoint: 'vs_main' },
          fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
          primitive: { topology: 'triangle-list' },
        });
        this.bindGroup = this.device.createBindGroup({
          layout: this.pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.uniformBuffer } },
            { binding: 1, resource: { buffer: this.storageBufferA } },
          ],
        });
      } else if (this.mode === 'soothe3') {
        const module = this.device.createShaderModule({ code: soothe3HeatmapWgsl });
        this.uniformBuffer = this.device.createBuffer({
          size: 32, // Soothe3HeatmapUniforms
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.storageBufferA = this.device.createBuffer({
          size: 8192 + 81920, // Soothe3ReductionFrame (8KB) + spectrogram_ring (80KB)
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.pipeline = this.device.createRenderPipeline({
          layout: 'auto',
          vertex: { module, entryPoint: 'vs_main' },
          fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
          primitive: { topology: 'triangle-list' },
        });
        this.bindGroup = this.device.createBindGroup({
          layout: this.pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.uniformBuffer } },
            { binding: 1, resource: { buffer: this.storageBufferA } },
          ],
        });
      } else {
        const module = this.device.createShaderModule({ code: mixerMetersWgsl });
        this.uniformBuffer = this.device.createBuffer({
          size: 32, // MixerMetersUniforms
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.storageBufferA = this.device.createBuffer({
          size: 12048, // TrackMeterEntry[502] (0x00040..0x02F50)
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.storageBufferB = this.device.createBuffer({
          size: 8192, // Ozone12MasterTelemetry (0x09000..0x0B000)
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.pipeline = this.device.createRenderPipeline({
          layout: 'auto',
          vertex: { module, entryPoint: 'vs_main' },
          fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
          primitive: { topology: 'triangle-list' },
        });
        this.bindGroup = this.device.createBindGroup({
          layout: this.pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.uniformBuffer } },
            { binding: 1, resource: { buffer: this.storageBufferA } },
            { binding: 2, resource: { buffer: this.storageBufferB } },
          ],
        });
      }

      this.isWebGpuReady = true;
      return true;
    } catch {
      this.isWebGpuReady = false;
      return false;
    }
  }

  /**
   * Renders a single 120 FPS frame using zero-copy slices from `telemetry.buffer`
   * (with automatic Canvas2D fallback when WebGPU is unavailable).
   */
  renderFrame(telemetry: DawTelemetryView): void {
    if (this.isWebGpuReady && this.device && this.context && this.pipeline) {
      if (this.mode === 'proq4') {
        const cascadeUniforms = telemetry.packProQ4BiquadCascadeUniforms(48000.0);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, cascadeUniforms.buffer);
        this.device.queue.writeBuffer(
          this.storageBufferA,
          0,
          telemetry.buffer,
          OFFSET_PROQ4_SPECTRUM,
          16384
        );
      } else if (this.mode === 'soothe3') {
        const u = new Float32Array([
          -90.0,
          0.0,
          4.5,
          (telemetry.frameSeq % 20) / 20.0,
          telemetry.soothe3MaxCutDb,
          telemetry.soothe3Detail,
          0,
          0,
        ]);
        const dv = new DataView(u.buffer);
        dv.setUint32(24, telemetry.soothe3HardMode ? 1 : 0, true);
        dv.setUint32(28, telemetry.soothe3DeltaSolo ? 1 : 0, true);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, u.buffer);
        this.device.queue.writeBuffer(
          this.storageBufferA,
          0,
          telemetry.buffer,
          OFFSET_SOOTHE3_REDUCTION,
          8192
        );
        this.device.queue.writeBuffer(
          this.storageBufferA,
          8192,
          telemetry.buffer,
          OFFSET_SPECTROGRAM_RING,
          81920
        );
      } else {
        const u = new ArrayBuffer(32);
        const dv = new DataView(u);
        dv.setFloat32(0, this.canvas.width, true);
        dv.setFloat32(4, this.canvas.height, true);
        dv.setUint32(8, telemetry.visibleBankStart, true);
        dv.setUint32(12, telemetry.visibleBankSize, true);
        dv.setFloat32(16, telemetry.lufsMomentary, true);
        dv.setFloat32(20, telemetry.lufsShortTerm, true);
        dv.setFloat32(24, telemetry.lufsIntegrated, true);
        dv.setFloat32(28, telemetry.masterGrDb, true);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, u);
        this.device.queue.writeBuffer(
          this.storageBufferA,
          0,
          telemetry.buffer,
          OFFSET_TRACK_METERS,
          12048
        );
        this.device.queue.writeBuffer(
          this.storageBufferB,
          0,
          telemetry.buffer,
          OFFSET_OZONE12_TELEMETRY,
          8192
        );
      }

      const encoder = this.device.createCommandEncoder();
      if (this.mode === 'proq4' && this.computePipeline && this.computeBindGroup) {
        const cpass = encoder.beginComputePass();
        cpass.setPipeline(this.computePipeline);
        cpass.setBindGroup(0, this.computeBindGroup);
        cpass.dispatchWorkgroups(32);
        cpass.end();
      }
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: this.context.getCurrentTexture().createView(),
            loadOp: 'clear',
            clearValue: { r: 0.035, g: 0.05, b: 0.085, a: 1.0 },
            storeOp: 'store',
          },
        ],
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.draw(6, 1, 0, 0);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      return;
    }

    // High-speed Canvas2D software fallback when WebGPU is not exposed
    const ctx = this.canvas.getContext('2d');
    if (!ctx) return;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, w, h);

    if (this.mode === 'proq4') {
      // Draw 1024-bin Post-EQ Spectrum (0x04000)
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let i = 0; i < 256; i++) {
        const bin = i * 4;
        const db = telemetry.proq4PostDb[bin];
        const x = (i / 255) * w;
        const y = h - Math.max(0, Math.min(1, (db + 90) / 90)) * h;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, h);
      ctx.fillStyle = 'rgba(34, 211, 238, 0.22)';
      ctx.fill();

      // Draw 24-Band Composite Response Curve
      ctx.beginPath();
      for (let i = 0; i <= 256; i++) {
        const nx = i / 256;
        const freq = 10 * Math.pow(3000, nx);
        const eqDb = telemetry.evaluateProQ4MagnitudeDbAt(freq);
        const x = nx * w;
        const y = h * 0.5 - (eqDb / 60) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 2.2;
      ctx.stroke();
    } else if (this.mode === 'soothe3') {
      // Draw Soothe3 512-bin dynamic resonance reduction curve hanging from top (0x07000)
      ctx.beginPath();
      ctx.moveTo(0, 0);
      for (let i = 0; i < 256; i++) {
        const grDb = telemetry.soothe3Reduction512[i * 2];
        const x = (i / 255) * w;
        const y = Math.min(h * 0.9, (Math.abs(grDb) / 36.0) * h);
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, 0);
      ctx.fillStyle = telemetry.soothe3DeltaSolo
        ? 'rgba(251, 146, 60, 0.45)'
        : 'rgba(244, 63, 94, 0.38)';
      ctx.fill();
      ctx.strokeStyle = '#fb7185';
      ctx.lineWidth = 2.0;
      ctx.stroke();
    } else {
      // Draw visible Mixer Track Meters + Ozone 12 Polar Vectorscope
      const meterW = (w * 0.65) / telemetry.visibleBankSize;
      for (let i = 0; i < telemetry.visibleBankSize; i++) {
        const m = telemetry.getTrackMeter(telemetry.visibleBankStart + i);
        const normL = Math.max(0, Math.min(1, (m.peakLDb + 60) / 60));
        const normR = Math.max(0, Math.min(1, (m.peakRDb + 60) / 60));
        const x = i * meterW;
        ctx.fillStyle = '#10b981';
        ctx.fillRect(x + 2, h * (1 - normL), meterW * 0.38, h * normL);
        ctx.fillRect(x + meterW * 0.45, h * (1 - normR), meterW * 0.38, h * normR);
      }
    }
  }
}
