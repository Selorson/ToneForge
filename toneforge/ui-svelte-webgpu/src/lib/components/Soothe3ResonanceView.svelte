<script lang="ts">
  import { DawTelemetryView } from '../state/daw_master_store.svelte';
  import { WebGpu120FpsRenderer } from '../wgpu/renderer_120fps';

  let { telemetry }: { telemetry: DawTelemetryView } = $props();

  let sootheCanvas: HTMLCanvasElement | undefined = $state();
  let renderer = $state<WebGpu120FpsRenderer | null>(null);

  $effect(() => {
    if (!sootheCanvas) return;
    const r = new WebGpu120FpsRenderer(sootheCanvas, 'soothe3');
    renderer = r;
    r.initialize();
  });

  $effect(() => {
    const _seq = telemetry.frameSeq;
    const _depth = telemetry.soothe3Depth;
    const _maxCut = telemetry.soothe3MaxCutDb;
    const _detail = telemetry.soothe3Detail;
    const _hard = telemetry.soothe3HardMode;
    const _delta = telemetry.soothe3DeltaSolo;
    if (renderer) {
      renderer.renderFrame(telemetry);
    }
  });
</script>

<section class="panel soothe3-panel">
  <header class="panel-header">
    <div>
      <h2>oeksound Soothe3 — Dynamic Resonance Suppressor (143 Params · 4096-Pt STFT)</h2>
      <span class="subtitle">
        512/1024-Bin Reduction Curve @ 0x07000 · 20×1024 Spectrogram Ring @ 0x0B000 · 4.5 dB/oct Perceptual Tilt
      </span>
    </div>
    <div class="mode-toggles">
      <button
        class={!telemetry.soothe3HardMode ? 'active' : ''}
        onclick={() => {
          telemetry.soothe3HardMode = false;
          telemetry.syncSoothe3ToEngine();
        }}
      >
        Soft Mode
      </button>
      <button
        class={telemetry.soothe3HardMode ? 'active hard' : ''}
        onclick={() => {
          telemetry.soothe3HardMode = true;
          telemetry.syncSoothe3ToEngine();
        }}
      >
        Hard Mode
      </button>
      <button
        class={telemetry.soothe3DeltaSolo ? 'active delta' : ''}
        onclick={() => {
          telemetry.soothe3DeltaSolo = !telemetry.soothe3DeltaSolo;
          telemetry.syncSoothe3ToEngine();
        }}
      >
        Delta Solo
      </button>
    </div>
  </header>

  <canvas bind:this={sootheCanvas} width={920} height={220} class="soothe3-canvas"></canvas>

  <div class="soothe-controls">
    <label>
      Depth: {telemetry.soothe3Depth.toFixed(1)}
      <input
        type="range"
        min="0"
        max="10"
        step="0.1"
        bind:value={telemetry.soothe3Depth}
        oninput={() => telemetry.syncSoothe3ToEngine()}
      />
    </label>
    <label>
      Detail (Sharpness/Selectivity): {telemetry.soothe3Detail.toFixed(1)}
      <input
        type="range"
        min="0"
        max="10"
        step="0.1"
        bind:value={telemetry.soothe3Detail}
        oninput={() => telemetry.syncSoothe3ToEngine()}
      />
    </label>
    <label>
      Attack: {telemetry.soothe3Attack.toFixed(1)}
      <input
        type="range"
        min="0"
        max="10"
        step="0.1"
        bind:value={telemetry.soothe3Attack}
        oninput={() => telemetry.syncSoothe3ToEngine()}
      />
    </label>
    <label>
      Release: {telemetry.soothe3Release.toFixed(1)}
      <input
        type="range"
        min="0"
        max="10"
        step="0.1"
        bind:value={telemetry.soothe3Release}
        oninput={() => telemetry.syncSoothe3ToEngine()}
      />
    </label>
    <label>
      Max Cut: {telemetry.soothe3MaxCutDb.toFixed(1)} dB
      <input
        type="range"
        min="-36"
        max="-3"
        step="0.5"
        bind:value={telemetry.soothe3MaxCutDb}
        oninput={() => telemetry.syncSoothe3ToEngine()}
      />
    </label>
    <label>
      Mix: {telemetry.soothe3MixPct.toFixed(0)}%
      <input
        type="range"
        min="0"
        max="100"
        step="1"
        bind:value={telemetry.soothe3MixPct}
        oninput={() => telemetry.syncSoothe3ToEngine()}
      />
    </label>
  </div>

  <div class="sc-bands-strip">
    {#each telemetry.soothe3ScBands.slice(0, 5) as sc (sc.bandIdx)}
      <div class="sc-card">
        <strong>SC Band {sc.bandIdx + 1}</strong>
        <span>{sc.freqHz.toFixed(0)} Hz · Sens {sc.depth.toFixed(1)}</span>
      </div>
    {/each}
  </div>
</section>

<style>
  .panel { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  h2 { margin: 0; font-size: 1rem; color: #fb7185; }
  .subtitle { font-size: 0.75rem; color: #94a3b8; }
  .mode-toggles button {
    background: #1e293b; color: #e2e8f0; border: 1px solid #334155; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.75rem; margin-left: 4px;
  }
  .mode-toggles button.active { background: #fb7185; color: #090d16; font-weight: 700; }
  .soothe3-canvas { width: 100%; height: 220px; border-radius: 6px; border: 1px solid #1e293b; }
  .soothe-controls { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; font-size: 0.76rem; background: #090d16; padding: 8px; border-radius: 6px; }
  .sc-bands-strip { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-top: 8px; }
  .sc-card { background: #131c31; border: 1px solid #1e293b; border-radius: 6px; padding: 6px; font-size: 0.72rem; display: flex; flex-direction: column; }
</style>
