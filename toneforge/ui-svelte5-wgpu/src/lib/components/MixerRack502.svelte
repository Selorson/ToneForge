<script lang="ts">
  import {
    DawTelemetryView,
    FIXED_POINT_30BIT_MAX,
    MAX_MIXER_TRACKS,
    MIXER_INSERT_SLOT_COUNT,
    NATIVE_INSERT_PLUGINS,
  } from '../state/daw_master_store.svelte';
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

  // Fader law: -inf..+6 dB on a 0..1 slider (0.75 = 0 dB), mapped to linear gain for the engine.
  const FADER_MIN_DB = -60;
  const FADER_MAX_DB = 6;
  function gainToFader(g: number): number {
    if (g <= 0.000001) return 0;
    const db = 20 * Math.log10(g);
    return Math.max(0, Math.min(1, (db - FADER_MIN_DB) / (FADER_MAX_DB - FADER_MIN_DB)));
  }
  function faderToGain(v: number): number {
    if (v <= 0.0005) return 0;
    return Math.pow(10, (FADER_MIN_DB + v * (FADER_MAX_DB - FADER_MIN_DB)) / 20);
  }
  function fmtDb(g: number): string {
    return g <= 0.000001 ? '-∞' : `${(20 * Math.log10(g)).toFixed(1)}`;
  }
  function setBankStart(start: number): void {
    if (!Number.isFinite(start)) return;
    const maxStart = Math.max(0, MAX_MIXER_TRACKS - telemetry.visibleBankSize);
    telemetry.visibleBankStart = Math.max(0, Math.min(maxStart, Math.round(start)));
  }
  function selectTrack(idx: number): void {
    telemetry.selectedTrackIndex = idx;
    telemetry.setTelemetryFocusTrack(idx);
  }

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
      <button title="Nhóm kênh trước" onclick={() => setBankStart(telemetry.visibleBankStart - telemetry.visibleBankSize)}>◀</button>
      <label class="bank-start" title="Kênh đầu tiên hiển thị (0..501)">
        Từ kênh
        <input
          type="number"
          min="0"
          max={MAX_MIXER_TRACKS - 1}
          value={telemetry.visibleBankStart}
          onchange={(e) => setBankStart(parseInt((e.currentTarget as HTMLInputElement).value, 10))}
        />
      </label>
      <button title="Nhóm kênh sau" onclick={() => setBankStart(telemetry.visibleBankStart + telemetry.visibleBankSize)}>▶</button>
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
        onclick={() => selectTrack(trk.index)}
        onkeydown={(e) => e.key === 'Enter' && selectTrack(trk.index)}
      >
        <div class="strip-idx">#{trk.index}</div>
        <div class="strip-name">{trk.name}</div>
        <div class="strip-peak" title="Peak L / R">
          {Math.max(m.peakLDb, -99).toFixed(1)} / {Math.max(m.peakRDb, -99).toFixed(1)}
        </div>
        <div class="strip-buttons">
          <button
            class={trk.muted ? 'ms-btn mute on' : 'ms-btn mute'}
            title="Mute"
            onclick={(e) => {
              e.stopPropagation();
              telemetry.setTrackMute(trk.index, !trk.muted);
            }}>M</button
          >
          {#if trk.index !== 0}
            <button
              class={trk.soloed ? 'ms-btn solo on' : 'ms-btn solo'}
              title="Solo"
              onclick={(e) => {
                e.stopPropagation();
                telemetry.setTrackSolo(trk.index, !trk.soloed);
              }}>S</button
            >
          {/if}
        </div>
        <label class="pan-row" title="Pan (double-click: center)">
          <input
            type="range"
            min="-1"
            max="1"
            step="0.01"
            value={trk.pan}
            onclick={(e) => e.stopPropagation()}
            ondblclick={() => telemetry.setTrackPan(trk.index, 0)}
            oninput={(e) => telemetry.setTrackPan(trk.index, parseFloat((e.currentTarget as HTMLInputElement).value))}
          />
          <span>{trk.pan === 0 ? 'C' : trk.pan < 0 ? `L${Math.round(-trk.pan * 100)}` : `R${Math.round(trk.pan * 100)}`}</span>
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.001"
          value={gainToFader(trk.gainLinear)}
          title="Fader (dB) — double-click: 0 dB"
          onclick={(e) => e.stopPropagation()}
          ondblclick={() => telemetry.setTrackGain(trk.index, 1.0)}
          oninput={(e) => {
            const val = parseFloat((e.currentTarget as HTMLInputElement).value);
            telemetry.setTrackGain(trk.index, faderToGain(val));
          }}
        />
        <div class="fader-db">{fmtDb(trk.gainLinear)} dB</div>
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
        {#each Array.from({ length: MIXER_INSERT_SLOT_COUNT }, (_, i) => i) as slotIdx (slotIdx)}
          {@const slot = selectedTrack.slots.find((s) => s.slotIndex === slotIdx)}
          {#if slot}
            <div class="slot-row">
              <label class="slot-enable" title="Bật / Bypass">
                <input
                  type="checkbox"
                  checked={slot.enabled}
                  onchange={(e) =>
                    telemetry.setSlotEnabled(
                      selectedTrack.index,
                      slot.slotIndex,
                      (e.currentTarget as HTMLInputElement).checked
                    )}
                />
                Slot {slot.slotIndex + 1}: <strong>{slot.name}</strong>
              </label>
              <label class="slot-wet" title="Dry/Wet song song (APDC bù trễ)">
                Wet {(slot.wetMix * 100).toFixed(0)}%
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={slot.wetMix}
                  oninput={(e) =>
                    telemetry.setSlotWetMix(
                      selectedTrack.index,
                      slot.slotIndex,
                      parseFloat((e.currentTarget as HTMLInputElement).value)
                    )}
                />
              </label>
              <button
                class="slot-remove"
                title="Gỡ plugin khỏi khe"
                onclick={() => telemetry.removePluginSlot(selectedTrack.index, slot.slotIndex)}>✕</button
              >
            </div>
          {:else}
            <div class="slot-row slot-row-empty">
              <span class="slot-empty-label">Slot {slotIdx + 1}: <em>— Trống —</em></span>
              <select
                class="slot-insert-select"
                value=""
                title="Chèn plugin nội bộ vào Slot {slotIdx + 1}"
                onchange={(e) => {
                  const kind = (e.currentTarget as HTMLSelectElement).value;
                  if (kind) {
                    telemetry.insertNativePluginSlot(selectedTrack.index, slotIdx, kind);
                    (e.currentTarget as HTMLSelectElement).value = '';
                  }
                }}
              >
                <option value="">+ Chèn plugin…</option>
                {#each NATIVE_INSERT_PLUGINS as plugin (plugin.kind)}
                  <option value={plugin.kind}>{plugin.label}</option>
                {/each}
              </select>
            </div>
          {/if}
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
                  telemetry.updateTensionSplinePoint(idx, { y: v });
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
                  const tVal = parseFloat((e.currentTarget as HTMLInputElement).value);
                  telemetry.updateTensionSplinePoint(idx, { tension: tVal });
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
  .bank-start { display: inline-flex; align-items: center; gap: 4px; font-size: 0.72rem; color: var(--muted); }
  .bank-start input { width: 56px; }
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
  .strip-buttons { display: flex; justify-content: center; gap: 3px; margin: 3px 0; }
  .ms-btn { width: 22px; height: 18px; padding: 0; font-size: 0.66rem; font-weight: 700; border-radius: 4px; border: 1px solid var(--border); background: var(--surface-2); color: var(--muted); cursor: pointer; }
  .ms-btn.mute.on { background: #c0392b; color: #fff; border-color: #c0392b; }
  .ms-btn.solo.on { background: #e5b400; color: #1d1d1d; border-color: #e5b400; }
  .pan-row { display: flex; flex-direction: column; align-items: center; font-size: 0.62rem; color: var(--muted); }
  .pan-row input { width: 64px; }
  .fader-db { font-family: var(--font-mono); font-size: 0.66rem; color: var(--text); }
  .slot-enable { display: flex; align-items: center; gap: 4px; flex: 1; min-width: 0; }
  .slot-wet { display: flex; align-items: center; gap: 4px; color: var(--muted); }
  .slot-wet input { width: 80px; }
  .slot-remove { background: transparent; border: 1px solid var(--border); color: var(--muted); border-radius: 4px; cursor: pointer; padding: 1px 6px; }
  .slot-row-empty { opacity: 0.78; }
  .slot-empty-label { flex: 1; color: var(--muted); }
  .slot-insert-select { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 2px 6px; font-size: 0.72rem; }
</style>

