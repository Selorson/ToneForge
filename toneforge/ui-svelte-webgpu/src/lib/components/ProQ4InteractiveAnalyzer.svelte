<script lang="ts">
  import { DawTelemetryView } from '../state/daw_master_store.svelte';
  import { WebGpu120FpsRenderer } from '../wgpu/renderer_120fps';

  let { telemetry }: { telemetry: DawTelemetryView } = $props();

  let eqCanvas: HTMLCanvasElement | undefined = $state();
  let renderer = $state<WebGpu120FpsRenderer | null>(null);
  let draggingBandIdx = $state<number | null>(null);

  const FILTER_SHAPE_NAMES = [
    'Bell',
    'Low Shelf',
    'Low Cut',
    'High Shelf',
    'High Cut',
    'Notch',
    'Band Pass',
    'Tilt Shelf',
    'Flat Tilt',
    'All Pass',
  ];

  const activeBand = $derived(telemetry.proq4Bands[telemetry.proq4SelectedBand]);

  function freqToXNorm(freqHz: number): number {
    return Math.log(Math.max(10, Math.min(30000, freqHz)) / 10.0) / Math.log(3000.0);
  }

  function xNormToFreq(nx: number): number {
    return 10.0 * Math.pow(3000.0, Math.max(0, Math.min(1, nx)));
  }

  function gainToYNorm(gainDb: number): number {
    return 0.5 - Math.max(-30, Math.min(30, gainDb)) / 60.0;
  }

  function yNormToGain(ny: number): number {
    return (0.5 - Math.max(0, Math.min(1, ny))) * 60.0;
  }

  function handlePointerMove(e: PointerEvent) {
    if (draggingBandIdx === null || !eqCanvas) return;
    const rect = eqCanvas.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top) / rect.height;
    const b = { ...telemetry.proq4Bands[draggingBandIdx] };
    b.used = true;
    b.enabled = true;
    b.freqHz = Math.round(xNormToFreq(nx));
    b.gainDb = parseFloat(yNormToGain(ny).toFixed(2));
    telemetry.updateProQ4Band(1, 0, b);
  }

  function handleDoubleClick(e: MouseEvent) {
    if (!eqCanvas) return;
    const rect = eqCanvas.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top) / rect.height;
    const unused = telemetry.proq4Bands.find((b) => !b.used);
    const targetIdx = unused ? unused.bandIdx : telemetry.proq4SelectedBand;
    const b = { ...telemetry.proq4Bands[targetIdx] };
    b.used = true;
    b.enabled = true;
    b.freqHz = Math.round(xNormToFreq(nx));
    b.gainDb = parseFloat(yNormToGain(ny).toFixed(2));
    telemetry.proq4SelectedBand = targetIdx;
    telemetry.updateProQ4Band(1, 0, b);
  }

  $effect(() => {
    if (!eqCanvas) return;
    const r = new WebGpu120FpsRenderer(eqCanvas, 'proq4');
    renderer = r;
    r.initialize();
  });

  $effect(() => {
    const _seq = telemetry.frameSeq;
    const _bandFreq = activeBand.freqHz;
    const _bandGain = activeBand.gainDb;
    const _bandQ = activeBand.q;
    const _bandShape = activeBand.shape;
    const _bandSlope = activeBand.slopeDbOct;
    const _scale = telemetry.proq4GainScalePct;
    if (renderer) {
      renderer.renderFrame(telemetry);
    }
  });
</script>

<section class="panel proq4-panel">
  <header class="panel-header">
    <div>
      <h2>FabFilter Pro-Q 4 — 24-Band Dynamic & Spectral EQ (601 Params · 120 FPS WGSL)</h2>
      <span class="subtitle">
        96-Section Biquad Cascade · Fractional Slopes (6..96 dB/oct & Brickwall) · Zero-Copy 1024-Bin Spectrum @ 0x03000
      </span>
    </div>
    <div class="phase-modes">
      <label class="preset-select">
        Factory Preset (171 Real .ffp):
        <select
          value={telemetry.proq4SelectedPreset}
          onchange={(e) =>
            telemetry.loadProQ4FactoryPreset((e.currentTarget as HTMLSelectElement).value)}
        >
          {#each [
            'Default Setting',
            'Vocals / Lead Vocal',
            'Vocals / Female Vocal',
            'Mixbus / Dynamic Smiley Curve ST',
            'Mixbus / Revigorate',
            'Mixbus / Mastering High & Low Cut',
            'Drums / Kick Punch',
            'Drums / Snare Snap',
            'Bass / Sub Tighten',
            'Guitar / Acoustic Sparkle',
            'Synth / Wide Pad',
            'Utility / Brickwall Sub Cut',
          ] as preset}
            <option value={preset}>{preset}</option>
          {/each}
        </select>
      </label>
      {#each ['ZeroLatency', 'NaturalPhase', 'LinearPhase'] as mode}
        <button
          class={telemetry.proq4PhaseMode === mode ? 'active' : ''}
          onclick={() => (telemetry.proq4PhaseMode = mode as any)}
        >
          {mode}
        </button>
      {/each}
    </div>
  </header>

  <div
    class="canvas-wrap"
    role="application"
    aria-label="Pro-Q 4 24-Band EQ Interactive Canvas"
    ondblclick={handleDoubleClick}
    onpointermove={handlePointerMove}
    onpointerup={() => (draggingBandIdx = null)}
    onpointerleave={() => (draggingBandIdx = null)}
  >
    <canvas bind:this={eqCanvas} width={920} height={260} class="proq4-canvas"></canvas>

    {#each telemetry.proq4Bands as band (band.bandIdx)}
      {#if band.used}
        <button
          class="eq-handle {telemetry.proq4SelectedBand === band.bandIdx ? 'selected' : ''} {band.spectralEnabled ? 'spectral' : ''}"
          style="left: {freqToXNorm(band.freqHz) * 100}%; top: {gainToYNorm(band.gainDb) * 100}%;"
          onpointerdown={(e) => {
            e.stopPropagation();
            telemetry.proq4SelectedBand = band.bandIdx;
            draggingBandIdx = band.bandIdx;
          }}
          title="Band {band.bandIdx + 1}: {band.freqHz} Hz, {band.gainDb} dB, Q={band.q}"
        >
          {band.bandIdx + 1}
        </button>
      {/if}
    {/each}
  </div>

  <div class="band-controls">
    <label>
      Band (1..24):
      <select
        value={telemetry.proq4SelectedBand}
        onchange={(e) => {
          const idx = parseInt((e.currentTarget as HTMLSelectElement).value, 10);
          telemetry.proq4SelectedBand = idx;
          telemetry.proq4Bands[idx].used = true;
          telemetry.proq4Bands[idx].enabled = true;
        }}
      >
        {#each telemetry.proq4Bands as b}
          <option value={b.bandIdx}>Band {b.bandIdx + 1} ({b.used ? `${b.freqHz} Hz` : 'Unused'})</option>
        {/each}
      </select>
    </label>

    <label>
      Shape:
      <select
        value={activeBand.shape}
        onchange={(e) => {
          const shape = parseInt((e.currentTarget as HTMLSelectElement).value, 10);
          telemetry.updateProQ4Band(1, 0, { ...activeBand, shape });
        }}
      >
        {#each FILTER_SHAPE_NAMES as name, sIdx}
          <option value={sIdx}>{name}</option>
        {/each}
      </select>
    </label>

    <label>
      Freq: {activeBand.freqHz.toFixed(0)} Hz
      <input
        type="range"
        min="10"
        max="30000"
        step="1"
        value={activeBand.freqHz}
        oninput={(e) =>
          telemetry.updateProQ4Band(1, 0, {
            ...activeBand,
            freqHz: parseFloat((e.currentTarget as HTMLInputElement).value),
          })}
      />
    </label>

    <label>
      Gain: {activeBand.gainDb.toFixed(1)} dB
      <input
        type="range"
        min="-30"
        max="30"
        step="0.1"
        value={activeBand.gainDb}
        oninput={(e) =>
          telemetry.updateProQ4Band(1, 0, {
            ...activeBand,
            gainDb: parseFloat((e.currentTarget as HTMLInputElement).value),
          })}
      />
    </label>

    <label>
      Q: {activeBand.q.toFixed(2)}
      <input
        type="range"
        min="0.1"
        max="24"
        step="0.05"
        value={activeBand.q}
        oninput={(e) =>
          telemetry.updateProQ4Band(1, 0, {
            ...activeBand,
            q: parseFloat((e.currentTarget as HTMLInputElement).value),
          })}
      />
    </label>

    <label>
      Slope: {activeBand.slopeDbOct} dB/oct
      <input
        type="range"
        min="6"
        max="96"
        step="3"
        value={activeBand.slopeDbOct}
        oninput={(e) =>
          telemetry.updateProQ4Band(1, 0, {
            ...activeBand,
            slopeDbOct: parseInt((e.currentTarget as HTMLInputElement).value, 10),
          })}
      />
    </label>

    <label>
      Dyn Range: {activeBand.dynamicRangeDb.toFixed(1)} dB
      <input
        type="range"
        min="-24"
        max="24"
        step="0.5"
        value={activeBand.dynamicRangeDb}
        oninput={(e) =>
          telemetry.updateProQ4Band(1, 0, {
            ...activeBand,
            dynamicRangeDb: parseFloat((e.currentTarget as HTMLInputElement).value),
          })}
      />
    </label>

    <label>
      <input
        type="checkbox"
        checked={activeBand.spectralEnabled}
        onchange={(e) =>
          telemetry.updateProQ4Band(1, 0, {
            ...activeBand,
            spectralEnabled: (e.currentTarget as HTMLInputElement).checked,
          })}
      />
      Spectral Dynamics
    </label>
  </div>
</section>

<style>
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  h2 { margin: 0; font-size: 1rem; color: var(--warn); font-weight: 600; }
  .subtitle { font-size: 0.75rem; color: var(--muted); }
  .phase-modes { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .preset-select { font-size: 0.75rem; color: var(--muted); margin-right: 6px; }
  .preset-select select {
    background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; margin-left: 4px; font-size: 0.74rem;
  }
  .phase-modes button {
    background: var(--surface-2); color: var(--muted); border: 1px solid var(--border); padding: 5px 10px; border-radius: 6px; cursor: pointer; font-size: 0.75rem; margin-left: 4px;
  }
  .phase-modes button.active { background: var(--accent); color: #0b2b1e; border-color: var(--accent); font-weight: 600; }
  .canvas-wrap { position: relative; width: 100%; height: 260px; background: var(--sub-surface); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
  .proq4-canvas { width: 100%; height: 100%; display: block; background: var(--sub-surface); }
  .eq-handle {
    position: absolute; transform: translate(-50%, -50%); width: 22px; height: 22px; border-radius: 50%;
    background: var(--warn); color: #0b2b1e; border: 2px solid var(--text); font-size: 0.7rem; font-weight: 700; cursor: grab;
  }
  .eq-handle.selected { box-shadow: 0 0 12px var(--warn); transform: translate(-50%, -50%) scale(1.18); }
  .eq-handle.spectral { background: var(--accent); }
  .band-controls {
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-top: 10px; font-size: 0.76rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; background: var(--sub-surface-2); border: 1px solid var(--border); padding: 8px 10px; border-radius: var(--radius);
  }
  .band-controls select {
    background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px;
  }
</style>

