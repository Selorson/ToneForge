<script lang="ts">
  import { DawTelemetryView } from '../state/daw_master_store.svelte';
  import { WebGpu120FpsRenderer } from '../wgpu/renderer_120fps';
  import Guide from './Guide.svelte';

  let { telemetry }: { telemetry: DawTelemetryView } = $props();

  let eqCanvas: HTMLCanvasElement | undefined = $state();
  let renderer = $state<WebGpu120FpsRenderer | null>(null);
  let draggingBandIdx = $state<number | null>(null);
  let isDrawingSketch = $state(false);
  let sketchNormPts = $state<[number, number][]>([]);
  let sketchHzDbPts = $state<[number, number][]>([]);

  const PIANO_ROLL_FREQ_KEYS = [
    { midi: 24, label: 'C1' },
    { midi: 36, label: 'C2' },
    { midi: 43, label: 'G2' },
    { midi: 48, label: 'C3' },
    { midi: 55, label: 'G3' },
    { midi: 60, label: 'C4' },
    { midi: 69, label: 'A4 (440)' },
    { midi: 72, label: 'C5' },
    { midi: 84, label: 'C6' },
    { midi: 96, label: 'C7' },
    { midi: 108, label: 'C8' },
  ];

  function midiToHz(midi: number): number {
    return 440.0 * Math.pow(2.0, (midi - 69) / 12.0);
  }

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

  function updateActiveProQ4Band(band: (typeof telemetry.proq4Bands)[number]): Promise<void> {
    const target = telemetry.resolvePluginSlotTarget('ProQ4', 1, 0);
    return telemetry.updateProQ4Band(target.trackIdx, target.slotIdx, band);
  }

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

  function handleCanvasPointerDown(e: PointerEvent) {
    if (!telemetry.proq4SketchMode || !eqCanvas) return;
    const rect = eqCanvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const nx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const ny = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    isDrawingSketch = true;
    sketchNormPts = [[nx * 100, ny * 100]];
    sketchHzDbPts = [[xNormToFreq(nx), yNormToGain(ny)]];
  }

  function handlePointerMove(e: PointerEvent) {
    if (isDrawingSketch && telemetry.proq4SketchMode && eqCanvas) {
      const rect = eqCanvas.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        const nx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const ny = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
        sketchNormPts = [...sketchNormPts, [nx * 100, ny * 100]];
        sketchHzDbPts = [...sketchHzDbPts, [xNormToFreq(nx), yNormToGain(ny)]];
      }
      return;
    }
    if (draggingBandIdx === null || !eqCanvas) return;
    const rect = eqCanvas.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top) / rect.height;
    const b = { ...telemetry.proq4Bands[draggingBandIdx] };
    b.used = true;
    b.enabled = true;
    b.freqHz = Math.round(xNormToFreq(nx));
    b.gainDb = parseFloat(yNormToGain(ny).toFixed(2));
    updateActiveProQ4Band(b);
  }

  async function finishPointerGesture() {
    draggingBandIdx = null;
    if (isDrawingSketch) {
      isDrawingSketch = false;
      const pts = sketchHzDbPts;
      sketchNormPts = [];
      sketchHzDbPts = [];
      if (pts.length >= 2) {
        await telemetry.fitProQ4SketchCurve(pts, 6);
      }
    }
  }

  let collisionGlowPath = $derived.by(() => {
    const _seq = telemetry.frameSeq;
    const _ref = telemetry.proq4CollisionRefTrack;
    if (!telemetry.proq4ShowCollisions || _ref === null) return '';
    const coords: string[] = ['M 0 100'];
    let anyActive = false;
    for (let i = 0; i < 1024; i += 8) {
      const m = telemetry.proq4CollisionMask[i] ?? 0;
      if (m > 0.05) anyActive = true;
      const x = (i / 1023) * 100;
      const y = 100 - Math.min(85, m * 70);
      coords.push(`L ${x.toFixed(2)} ${y.toFixed(2)}`);
    }
    if (!anyActive) return '';
    coords.push('L 100 100 Z');
    return coords.join(' ');
  });

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
    updateActiveProQ4Band(b);
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
    const _bandEnabled = activeBand.enabled;
    const _scale = telemetry.proq4GainScalePct;
    const _bypass = telemetry.proq4Bypass;
    const _collisions = telemetry.proq4ShowCollisions;
    if (renderer) {
      renderer.renderFrame(telemetry);
    }
  });
</script>

<section class="panel proq4-panel">
  <Guide id="proq4" />
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
          onclick={() => telemetry.setProQ4PhaseMode(mode as 'ZeroLatency' | 'NaturalPhase' | 'LinearPhase')}
        >
          {mode}
        </button>
      {/each}

      <button
        class={telemetry.proq4SketchMode ? 'active sketch-btn' : ''}
        onclick={() => (telemetry.proq4SketchMode = !telemetry.proq4SketchMode)}
        title="Bật chế độ EQ Sketch: Kéo chuột vẽ một nét liền tay qua màn hình, máy sẽ tự khớp các điểm lọc Parametric EQ (< 0.85 dB RMS)"
      >
        ✏️ EQ Sketch ({telemetry.proq4SketchMode ? 'ON' : 'OFF'})
      </button>

      <label class="preset-select">
        Collision Ref:
        <select
          value={telemetry.proq4CollisionRefTrack === null ? 'off' : String(telemetry.proq4CollisionRefTrack)}
          onchange={(e) => {
            const v = (e.currentTarget as HTMLSelectElement).value;
            telemetry.setProQ4CollisionReference(v === 'off' ? null : parseInt(v, 10));
          }}
        >
          <option value="off">Off (No Ref)</option>
          <option value="0">Track 0 (Master)</option>
          {#each [1, 2, 3, 4, 5, 6, 7, 8] as trk}
            <option value={String(trk)}>
              Track {trk} ({telemetry.mixerTracks[trk]?.name ?? `Insert ${trk}`})
            </option>
          {/each}
        </select>
      </label>
    </div>
  </header>

  <div
    class="canvas-wrap {telemetry.proq4SketchMode ? 'sketch-cursor' : ''}"
    role="application"
    aria-label="Pro-Q 4 24-Band EQ Interactive Canvas"
    ondblclick={handleDoubleClick}
    onpointerdown={handleCanvasPointerDown}
    onpointermove={handlePointerMove}
    onpointerup={finishPointerGesture}
    onpointerleave={finishPointerGesture}
  >
    <canvas bind:this={eqCanvas} width={920} height={260} class="proq4-canvas"></canvas>

    <!-- Cross-Track Collision Glow & Live EQ Sketch Overlay -->
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" class="overlay-svg">
      {#if collisionGlowPath}
        <path d={collisionGlowPath} fill="rgba(239, 68, 68, 0.34)" stroke="#ef4444" stroke-width="0.5" />
      {/if}
      {#if sketchNormPts.length >= 2}
        <polyline
          points={sketchNormPts.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ')}
          fill="none"
          stroke="#fbbf24"
          stroke-width="0.9"
          stroke-dasharray="1.5 0.8"
        />
      {/if}
    </svg>

    {#each telemetry.proq4Bands as band (band.bandIdx)}
      {#if band.used}
        <button
          class="eq-handle {telemetry.proq4SelectedBand === band.bandIdx ? 'selected' : ''} {band.spectralEnabled ? 'spectral' : ''}"
          style="left: {freqToXNorm(band.freqHz) * 100}%; top: {gainToYNorm(band.gainDb) * 100}%;"
          onpointerdown={(e) => {
            if (telemetry.proq4SketchMode) return;
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

  <!-- Frequency Piano Roll Bar (C1–C8 Musical Note Frequency Snap) -->
  <div class="freq-piano-bar" aria-label="Pro-Q 4 Frequency Piano Roll">
    <span class="piano-bar-title">🎹 Note Snap:</span>
    {#each PIANO_ROLL_FREQ_KEYS as pk}
      {@const hz = midiToHz(pk.midi)}
      <button
        type="button"
        class="freq-key-btn {Math.abs(activeBand.freqHz - hz) < hz * 0.03 ? 'active' : ''}"
        onclick={() =>
          updateActiveProQ4Band({
            ...activeBand,
            used: true,
            enabled: true,
            freqHz: Math.round(hz * 10) / 10,
          })}
        title="Đặt Band {activeBand.bandIdx + 1} vào đúng tần số nốt {pk.label} ({hz.toFixed(1)} Hz)"
      >
        {pk.label} <small>{hz < 1000 ? `${hz.toFixed(0)}Hz` : `${(hz / 1000).toFixed(2)}k`}</small>
      </button>
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
          updateActiveProQ4Band({
            ...telemetry.proq4Bands[idx],
            used: true,
            enabled: true,
          });
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
          updateActiveProQ4Band({ ...activeBand, shape });
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
          updateActiveProQ4Band({
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
          updateActiveProQ4Band({
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
          updateActiveProQ4Band({
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
          updateActiveProQ4Band({
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
          updateActiveProQ4Band({
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
          updateActiveProQ4Band({
            ...activeBand,
            spectralEnabled: (e.currentTarget as HTMLInputElement).checked,
          })}
      />
      Spectral Dynamics
    </label>

    <label>
      Scale: {telemetry.proq4GainScalePct.toFixed(0)}%
      <input
        type="range"
        min="0"
        max="200"
        step="5"
        value={telemetry.proq4GainScalePct}
        oninput={(e) => {
          telemetry.proq4GainScalePct = parseFloat((e.currentTarget as HTMLInputElement).value);
          telemetry.syncProQ4GlobalToEngine();
        }}
      />
    </label>

    <label>
      <input
        type="checkbox"
        checked={telemetry.proq4AutoGain}
        onchange={(e) => {
          telemetry.proq4AutoGain = (e.currentTarget as HTMLInputElement).checked;
          telemetry.syncProQ4GlobalToEngine();
        }}
      />
      Auto Gain
    </label>

    <label>
      <input
        type="checkbox"
        checked={telemetry.proq4ShowCollisions}
        onchange={(e) => {
          telemetry.proq4ShowCollisions = (e.currentTarget as HTMLInputElement).checked;
          telemetry.syncProQ4GlobalToEngine();
        }}
      />
      Collisions
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
  .phase-modes button.active.sketch-btn { background: var(--warn); color: #0b2b1e; border-color: var(--warn); }
  .canvas-wrap { position: relative; width: 100%; height: 260px; background: var(--sub-surface); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
  .canvas-wrap.sketch-cursor { cursor: crosshair; }
  .proq4-canvas { width: 100%; height: 100%; display: block; background: var(--sub-surface); }
  .overlay-svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
  .freq-piano-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 4px; margin-top: 6px; padding: 5px 8px; background: var(--sub-surface); border: 1px solid var(--border); border-radius: 6px; font-family: var(--font-mono); font-size: 0.69rem; }
  .piano-bar-title { color: var(--muted); margin-right: 4px; font-weight: 600; }
  .freq-key-btn { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 3px 7px; font-size: 0.68rem; font-family: var(--font-mono); cursor: pointer; display: flex; align-items: center; gap: 4px; }
  .freq-key-btn small { color: var(--muted); font-size: 0.62rem; }
  .freq-key-btn.active { background: var(--warn); color: #0b2b1e; border-color: var(--warn); font-weight: 700; }
  .freq-key-btn.active small { color: #0b2b1e; }
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

