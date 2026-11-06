<script lang="ts">
  import { DawTelemetryView, FIXED_POINT_30BIT_MAX, MAX_MIXER_TRACKS } from '../state/daw_master_store.svelte';
  import { WebGpu120FpsRenderer } from '../wgpu/renderer_120fps';
  import Guide from './Guide.svelte';

  let { telemetry }: { telemetry: DawTelemetryView } = $props();

  let meterCanvas: HTMLCanvasElement | undefined = $state();
  let renderer = $state<WebGpu120FpsRenderer | null>(null);
  let routeTargetTrack = $state(0);
  let routeSendGain = $state(1.0);
  let routeIsSidechain = $state(false);

  const visibleTracks = $derived(
    telemetry.mixerTracks.slice(
      telemetry.visibleBankStart,
      Math.min(MAX_MIXER_TRACKS, telemetry.visibleBankStart + telemetry.visibleBankSize)
    )
  );

  const selectedTrack = $derived(telemetry.mixerTracks[telemetry.selectedTrackIndex] ?? telemetry.mixerTracks[0]);

  $effect(() => {
    if (!meterCanvas) return;
    const r = new WebGpu120FpsRenderer(meterCanvas, 'mixer_meters');
    renderer = r;
    r.initialize();
  });

  $effect(() => {
    // Re-render at 120 Hz whenever frameSeq advances
    const _seq = telemetry.frameSeq;
    if (renderer) {
      renderer.renderFrame(telemetry);
    }
  });
</script>

<section class="panel mixer-panel">
  <Guide id="mixer" />
  <header class="panel-header">
    <div>
      <h2>502-Track Topological Mixer & 30-Bit TensionSpline Bus</h2>
      <span class="subtitle">
        Master (0) · Inserts (1..500) · Preview (501) · 5-Point APDC Active · 0x40000000 Fixed-Point Controller
      </span>
    </div>
    <div class="bank-selector">
      <button onclick={() => (telemetry.visibleBankStart = 0)}>Tracks 0..31</button>
      <button onclick={() => (telemetry.visibleBankStart = 32)}>32..63</button>
      <button onclick={() => (telemetry.visibleBankStart = 128)}>128..159</button>
      <button onclick={() => (telemetry.visibleBankStart = 256)}>256..287</button>
      <button onclick={() => (telemetry.visibleBankStart = 470)}>470..501 (Preview)</button>
    </div>
  </header>

  <canvas
    bind:this={meterCanvas}
    width={920}
    height={130}
    class="wgpu-meter-bridge"
  ></canvas>

  <div class="mixer-strips">
    {#each visibleTracks as trk (trk.index)}
      {@const m = telemetry.getTrackMeter(trk.index)}
      <div
        class="strip {telemetry.selectedTrackIndex === trk.index ? 'selected' : ''}"
        role="button"
        tabindex="0"
        onclick={() => (telemetry.selectedTrackIndex = trk.index)}
        onkeydown={(e) => e.key === 'Enter' && (telemetry.selectedTrackIndex = trk.index)}
      >
        <div class="strip-idx">#{trk.index}</div>
        <div class="strip-name">{trk.name}</div>
        <div class="strip-peak">{m.peakLDb.toFixed(1)} dB</div>
        <input
          type="range"
          min="0"
          max="1.5"
          step="0.01"
          value={trk.gainLinear}
          oninput={(e) => {
            const val = parseFloat((e.currentTarget as HTMLInputElement).value);
            telemetry.setTrackGain(trk.index, val);
          }}
        />
        <div class="apdc-badge">{trk.apdcLatencySamples} smp</div>
      </div>
    {/each}
  </div>

  <div class="inspector-grid">
    <div class="subcard">
      <div class="subcard-header">
        <div>
          <h3>Track #{selectedTrack.index} ({selectedTrack.name}) — 10 FX Insert Slots & Routing</h3>
          <span class="subtitle">{telemetry.fl26FxPresetInfo}</span>
        </div>
        <label class="fx-preset-select">
          FL26 Native FX Preset (.fst):
          <select
            value={telemetry.fl26SelectedFxPreset}
            onchange={(e) =>
              telemetry.loadFl26FxPreset((e.currentTarget as HTMLSelectElement).value)}
          >
            <option value="Maximus: Clear Master.fst">Maximus: Clear Master.fst</option>
            <option value="Maximus: Maximum loudness.fst">Maximus: Maximum loudness.fst</option>
            <option value="Maximus: New York compression.fst">Maximus: New York compression.fst</option>
            <option value="Maximus: Soundgoodizer A.fst">Maximus: Soundgoodizer A.fst</option>
            <option value="Emphasis: Pop (heavy).fst">Emphasis: Pop (heavy).fst</option>
            <option value="Emphasis: Louder (transient shaping).fst">Emphasis: Louder (transient shaping).fst</option>
            <option value="Emphasis: Soft-clipper (heavy).fst">Emphasis: Soft-clipper (heavy).fst</option>
            <option value="Emphasis — Clean upward limiter.fst">Emphasis — Clean upward limiter.fst</option>
            <option value="Fruity Limiter: Drum compression 1.fst">Fruity Limiter: Drum compression 1.fst</option>
            <option value="Fruity Limiter: Ducking music behind voice.fst">Fruity Limiter: Ducking music behind voice.fst</option>
            <option value="Fruity Parametric EQ 2: Vocal presence.fst">Fruity Parametric EQ 2: Vocal presence.fst</option>
            <option value="Fruity Parametric EQ 2: Mastering.fst">Fruity Parametric EQ 2: Mastering.fst</option>
            <option value="Fruity Soft Clipper.fst">Fruity Soft Clipper.fst</option>
            <option value="LuxeVerb: Shimmer Cathedral.fst">LuxeVerb: Shimmer Cathedral.fst</option>
            <option value="Gross Beat: 1/2 Gate.fst">Gross Beat: 1/2 Gate.fst</option>
          </select>
        </label>
      </div>
      <div class="slots-list">
        {#each selectedTrack.slots as slot (slot.slotIndex)}
          <div class="slot-row">
            <span>Slot {slot.slotIndex + 1}: <strong>{slot.name}</strong></span>
            <span>Wet {(slot.wetMix * 100).toFixed(0)}%</span>
          </div>
        {:else}
          <div class="slot-empty">No insert FX loaded on Track #{selectedTrack.index}</div>
        {/each}
      </div>
      <div class="route-controls">
        <label>
          Send to Track:
          <input type="number" min="0" max="501" bind:value={routeTargetTrack} />
        </label>
        <label>
          Gain:
          <input type="range" min="0" max="1" step="0.01" bind:value={routeSendGain} />
        </label>
        <label>
          <input type="checkbox" bind:checked={routeIsSidechain} /> Sidechain Only
        </label>
        <button
          onclick={() =>
            telemetry.routeMixerTrack(selectedTrack.index, routeTargetTrack, routeSendGain, routeIsSidechain)}
        >
          Apply Route
        </button>
      </div>
    </div>

    <div class="subcard">
      <h3>
        30-Bit Fixed-Point TensionSpline (0x{telemetry.activeControllerFixed30
          .toString(16)
          .toUpperCase()
          .padStart(8, '0')} / 0x{FIXED_POINT_30BIT_MAX.toString(16).toUpperCase()})
      </h3>
      <div class="spline-list">
        {#each telemetry.tensionSplinePoints as pt, idx}
          <div class="spline-row">
            <span>Pt {idx + 1} (@ {pt.x} PPQ)</span>
            <label>
              Val {pt.y.toFixed(2)}
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={pt.y}
                oninput={(e) => {
                  const v = parseFloat((e.currentTarget as HTMLInputElement).value);
                  telemetry.tensionSplinePoints[idx].y = v;
                  telemetry.setInternalControllerNorm(telemetry.activeControllerId, v);
                }}
              />
            </label>
            <label>
              Tension {pt.tension.toFixed(2)}
              <input
                type="range"
                min="-0.95"
                max="0.95"
                step="0.05"
                value={pt.tension}
                oninput={(e) => {
                  telemetry.tensionSplinePoints[idx].tension = parseFloat(
                    (e.currentTarget as HTMLInputElement).value
                  );
                }}
              />
            </label>
          </div>
        {/each}
      </div>
    </div>
  </div>
</section>

<style>
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  h2 { margin: 0; font-size: 1rem; color: var(--accent); font-weight: 600; }
  .subtitle { font-size: 0.75rem; color: var(--muted); }
  .bank-selector button, .route-controls button {
    background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; cursor: pointer; font-size: 0.75rem; font-weight: 500;
  }
  .route-controls button {
    background: var(--accent); color: #0b2b1e; border-color: var(--accent); font-weight: 600;
  }
  .route-controls input[type="number"] {
    width: 60px; background: var(--sub-surface); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 4px 6px; font-family: var(--font-mono);
  }
  .wgpu-meter-bridge { width: 100%; height: 120px; background: var(--sub-surface); border-radius: var(--radius); border: 1px solid var(--border); margin-bottom: 10px; display: block; }
  .mixer-strips { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 8px; }
  .strip {
    min-width: 76px; background: var(--sub-surface-2); border: 1px solid var(--border); border-radius: 6px; padding: 6px; text-align: center; cursor: pointer; font-size: 0.72rem;
  }
  .strip.selected { border-color: var(--accent); background: var(--surface-2); box-shadow: inset 0 0 0 1px var(--accent); }
  .strip-idx { color: var(--muted); font-family: var(--font-mono); font-size: 0.68rem; }
  .strip-name { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text); }
  .strip-peak { color: var(--ok); font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  .apdc-badge { color: var(--warn); font-size: 0.68rem; font-family: var(--font-mono); }
  .inspector-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }
  .subcard { background: var(--sub-surface-2); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px; font-size: 0.76rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  .subcard-header { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }
  .fx-preset-select { font-size: 0.72rem; color: var(--muted); }
  .fx-preset-select select { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 3px 6px; margin-left: 4px; font-size: 0.72rem; }
  h3 { margin: 0; font-size: 0.82rem; font-family: var(--font-sans); color: var(--accent); font-weight: 600; }
  .slot-row, .spline-row, .route-controls { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 4px; }
</style>

