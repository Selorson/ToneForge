<script lang="ts">
  import { DawTelemetryView } from '../state/daw_master_store.svelte';
  import { WebGpu120FpsRenderer } from '../wgpu/renderer_120fps';
  import Guide from './Guide.svelte';

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
    const _sc = telemetry.soothe3ScBands.map((b) => `${b.enabled}:${b.freqHz}:${b.depth}:${b.q}`).join('|');
    if (renderer) {
      renderer.renderFrame(telemetry);
    }
  });
  const SOOTHE3_PRESETS = [
    'That Grammy balance preset',
    'General purpose de-harsher',
    'SM7 Vocals',
    'Lead Vox Bright',
    'General de-ess',
    'De-ess normal',
    'Harsh Cymbals',
    'Dampen ringy snare',
    'Clean up muddy bass',
    'Acoustic guitar body',
    'Fizz remover for heavy guitars',
    'Soften synth pad',
    '2bus - glue and air',
    'Master air',
    'Default Setting',
  ];
</script>

<section class="panel soothe3-panel">
  <Guide id="soothe3" />
  <header class="panel-header">
    <div>
      <h2>oeksound Soothe3 — Dynamic Resonance Suppressor (143 Params · 4096-Pt STFT)</h2>
      <span class="subtitle">
        512/1024-Bin Reduction Curve @ 0x07000 · 20×1024 Spectrogram Ring @ 0x0B000 · 4.5 dB/oct Perceptual Tilt
      </span>
    </div>
    <div class="mode-toggles">
      <label class="preset-select">
        Factory Preset (68 Real .preset):
        <select
          value={telemetry.soothe3SelectedPreset}
          onchange={(e) =>
            telemetry.loadSoothe3FactoryPreset((e.currentTarget as HTMLSelectElement).value)}
        >
          {#each SOOTHE3_PRESETS as p}
            <option value={p}>{p}</option>
          {/each}
        </select>
      </label>
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
    {#each telemetry.soothe3ScBands.slice(0, 6) as sc (sc.bandIdx)}
      <div class="sc-card {sc.enabled ? '' : 'disabled'}">
        <div class="sc-card-head">
          <strong>SC Band {sc.bandIdx + 1}</strong>
          <input
            type="checkbox"
            checked={sc.enabled}
            onchange={(e) =>
              telemetry.updateSoothe3ScBand(sc.bandIdx, {
                enabled: (e.currentTarget as HTMLInputElement).checked,
              })}
            title="Enable/Disable Sidechain EQ Band {sc.bandIdx + 1}"
          />
        </div>
        <span>{sc.freqHz.toFixed(0)} Hz · Sens {sc.depth.toFixed(1)} · Q {sc.q.toFixed(1)}</span>
        <label class="sc-mini-row">
          Hz
          <input
            type="range"
            min="40"
            max="18000"
            step="10"
            value={sc.freqHz}
            oninput={(e) =>
              telemetry.updateSoothe3ScBand(sc.bandIdx, {
                enabled: true,
                freqHz: parseFloat((e.currentTarget as HTMLInputElement).value),
              })}
          />
        </label>
        <label class="sc-mini-row">
          Sens
          <input
            type="range"
            min="-10"
            max="10"
            step="0.2"
            value={sc.depth}
            oninput={(e) =>
              telemetry.updateSoothe3ScBand(sc.bandIdx, {
                enabled: true,
                depth: parseFloat((e.currentTarget as HTMLInputElement).value),
              })}
          />
        </label>
        <label class="sc-mini-row">
          Q
          <input
            type="range"
            min="0.4"
            max="8.0"
            step="0.1"
            value={sc.q}
            oninput={(e) =>
              telemetry.updateSoothe3ScBand(sc.bandIdx, {
                enabled: true,
                q: parseFloat((e.currentTarget as HTMLInputElement).value),
              })}
          />
        </label>
      </div>
    {/each}
  </div>
</section>

<style>
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  h2 { margin: 0; font-size: 1rem; color: var(--err); font-weight: 600; }
  .subtitle { font-size: 0.75rem; color: var(--muted); }
  .mode-toggles { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .preset-select { font-size: 0.75rem; color: var(--muted); margin-right: 6px; }
  .preset-select select {
    background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; margin-left: 4px; font-size: 0.74rem;
  }
  .mode-toggles button {
    background: var(--surface-2); color: var(--muted); border: 1px solid var(--border); padding: 5px 10px; border-radius: 6px; cursor: pointer; font-size: 0.75rem; margin-left: 4px;
  }
  .mode-toggles button.active { background: var(--accent); color: #0b2b1e; border-color: var(--accent); font-weight: 600; }
  .mode-toggles button.active.hard { background: var(--err); color: #0b2b1e; border-color: var(--err); }
  .mode-toggles button.active.delta { background: var(--warn); color: #0b2b1e; border-color: var(--warn); }
  .soothe3-canvas { width: 100%; height: 220px; background: var(--sub-surface); border-radius: var(--radius); border: 1px solid var(--border); display: block; }
  .soothe-controls { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; font-size: 0.76rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; background: var(--sub-surface-2); border: 1px solid var(--border); padding: 8px 10px; border-radius: var(--radius); }
  .sc-bands-strip { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin-top: 8px; }
  .sc-card { background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; font-size: 0.70rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; display: flex; flex-direction: column; gap: 3px; }
  .sc-card.disabled { opacity: 0.55; }
  .sc-card-head { display: flex; justify-content: space-between; align-items: center; }
  .sc-card strong { color: var(--accent); }
  .sc-mini-row { display: flex; justify-content: space-between; align-items: center; gap: 4px; font-size: 0.66rem; color: var(--muted); }
  .sc-mini-row input[type="range"] { width: 72%; }
</style>

