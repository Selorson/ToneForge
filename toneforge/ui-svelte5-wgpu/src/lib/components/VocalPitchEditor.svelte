<script lang="ts">
  import { DawTelemetryView, type VocalNoteBlobUi } from '../state/daw_master_store.svelte';
  import Guide from './Guide.svelte';

  let { telemetry }: { telemetry: DawTelemetryView } = $props();

  const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const SCALE_NAMES = [
    'Chromatic (12-Note)',
    'Major (Ionian)',
    'Natural Minor (Aeolian)',
    'Major Pentatonic',
    'Minor Pentatonic',
  ];

  const MIN_MIDI = 55; // G3
  const MAX_MIDI = 74; // D5
  const MIDI_ROWS = Array.from({ length: MAX_MIDI - MIN_MIDI + 1 }, (_, idx) => MAX_MIDI - idx);

  let dragBlobId = $state<number | null>(null);
  let dragMode = $state<'pitch' | 'warp' | null>(null);
  let dragStartY = $state(0);
  let dragStartX = $state(0);
  let dragOrigMidi = $state(60);
  let dragOrigWarpEnd = $state(28800);
  let canvasWrapEl = $state<HTMLDivElement | null>(null);

  function midiLabel(midi: number): string {
    const rounded = Math.round(midi);
    const name = NOTE_NAMES[((rounded % 12) + 12) % 12];
    const oct = Math.floor(rounded / 12) - 1;
    return `${name}${oct}`;
  }

  function isBlackKey(midi: number): boolean {
    const pc = ((midi % 12) + 12) % 12;
    return pc === 1 || pc === 3 || pc === 6 || pc === 8 || pc === 10;
  }

  function midiToTopPct(midi: number): number {
    const span = MAX_MIDI - MIN_MIDI + 1;
    const clamped = Math.max(MIN_MIDI, Math.min(MAX_MIDI + 0.99, midi));
    return ((MAX_MIDI - clamped) / span) * 100;
  }

  const selectedBlob = $derived<VocalNoteBlobUi | null>(
    telemetry.vocalBlobs.find((b) => b.blob_id === telemetry.vocalSelectedBlobId) ??
      telemetry.vocalBlobs[0] ??
      null
  );

  function updateBlob(blobId: number, patch: Partial<VocalNoteBlobUi>) {
    telemetry.vocalBlobs = telemetry.vocalBlobs.map((b) =>
      b.blob_id === blobId ? { ...b, ...patch } : b
    );
  }

  function startBlobDrag(e: PointerEvent, blob: VocalNoteBlobUi, mode: 'pitch' | 'warp') {
    e.stopPropagation();
    telemetry.vocalSelectedBlobId = blob.blob_id;
    dragBlobId = blob.blob_id;
    dragMode = mode;
    dragStartY = e.clientY;
    dragStartX = e.clientX;
    dragOrigMidi = blob.target_midi_note;
    dragOrigWarpEnd = blob.warped_end_sample;
  }

  function handleCanvasPointerMove(e: PointerEvent) {
    if (dragBlobId === null || !dragMode || !canvasWrapEl) return;
    const rect = canvasWrapEl.getBoundingClientRect();
    if (rect.height <= 0 || rect.width <= 0) return;

    if (dragMode === 'pitch') {
      const rowHeight = rect.height / MIDI_ROWS.length;
      const deltaSemitones = -Math.round((e.clientY - dragStartY) / Math.max(8, rowHeight));
      const nextMidi = Math.max(MIN_MIDI, Math.min(MAX_MIDI, dragOrigMidi + deltaSemitones));
      updateBlob(dragBlobId, { target_midi_note: nextMidi });
    } else if (dragMode === 'warp') {
      const target = telemetry.vocalBlobs.find((b) => b.blob_id === dragBlobId);
      if (!target) return;
      const deltaNorm = (e.clientX - dragStartX) / rect.width;
      const deltaSamples = Math.round(deltaNorm * viewSpan);
      const minEnd = target.warped_start_sample + 2048;
      const maxEnd = Math.round(telemetry.vocalTotalFrames * 1.5);
      updateBlob(dragBlobId, {
        warped_end_sample: Math.max(minEnd, Math.min(maxEnd, dragOrigWarpEnd + deltaSamples)),
      });
    }
  }

  function stopBlobDrag() {
    dragBlobId = null;
    dragMode = null;
  }

  // Viewport: a zoomable sample window; only blobs intersecting it are rendered (DOM + contour).
  let viewZoom = $state(1);
  let viewStartSample = $state(0);
  const viewSpan = $derived(Math.max(1, Math.round(Math.max(1, telemetry.vocalTotalFrames) / viewZoom)));
  const visibleBlobs = $derived(
    telemetry.vocalBlobs.filter(
      (b) => b.warped_start_sample < viewStartSample + viewSpan && b.warped_end_sample > viewStartSample
    )
  );
  function setZoom(z: number) {
    viewZoom = Math.max(1, Math.min(64, z));
    const maxStart = Math.max(0, telemetry.vocalTotalFrames - viewSpan);
    viewStartSample = Math.min(viewStartSample, maxStart);
  }
  function scrollView(fraction: number) {
    const maxStart = Math.max(0, telemetry.vocalTotalFrames - viewSpan);
    viewStartSample = Math.max(0, Math.min(maxStart, Math.round(viewStartSample + fraction * viewSpan)));
  }
  function sampleToPct(sample: number): number {
    return ((sample - viewStartSample) / viewSpan) * 100;
  }

  function buildContourPolyline(blob: VocalNoteBlobUi): string {
    if (!blob.pitch_contour_midi || blob.pitch_contour_midi.length === 0) return '';
    const startPct = sampleToPct(blob.warped_start_sample);
    const widthPct = ((blob.warped_end_sample - blob.warped_start_sample) / viewSpan) * 100;
    const targetCenter = blob.target_midi_note + blob.fine_pitch_cents / 100.0;
    const n = blob.pitch_contour_midi.length;
    const pts: string[] = [];
    for (let i = 0; i < n; i++) {
      const rawMidi = blob.pitch_contour_midi[i];
      const devFromDetected = rawMidi - blob.detected_midi_note;
      const shapedDev =
        devFromDetected * (1.0 - blob.pitch_drift_correction * 0.75) * blob.vibrato_scaling;
      const finalMidi = targetCenter + shapedDev;
      const x = startPct + (n > 1 ? (i / (n - 1)) * widthPct : widthPct * 0.5);
      const y = midiToTopPct(finalMidi) + 100 / MIDI_ROWS.length / 2;
      pts.push(`${x.toFixed(2)},${y.toFixed(2)}`);
    }
    return pts.join(' ');
  }
</script>

<section class="panel vocal-panel">
  <Guide id="vocal" />
  <header class="panel-header">
    <div>
      <h2>
        🎤 Pitcher Real-Time & NewTone Graphical Vocal Editor — True-Envelope Formant Preservation
      </h2>
      <span class="subtitle">
        pYIN Probabilistic YIN + Viterbi HMM · 4–10 kHz Sibilant Guard · Cepstral Liftering (Zero Chipmunk Effect)
      </span>
    </div>
    <div class="action-buttons">
      <button class="act-btn primary" onclick={() => telemetry.analyzeVocalPitchBlobs()}>
        🔍 Phân tích Giọng hát (pYIN)
      </button>
      <button class="act-btn" onclick={() => telemetry.snapAllVocalBlobsToScale()}>
        🧲 Snap All to Scale
      </button>
      {#if selectedBlob}
        <button class="act-btn" onclick={() => telemetry.splitVocalBlob(selectedBlob.blob_id)}>
          ✂ Split Blob #{selectedBlob.blob_id + 1}
        </button>
      {/if}
      <button class="act-btn commit" onclick={() => telemetry.commitVocalPitchEdits(1)}>
        ✅ Commit Vocal to Track #1
      </button>
    </div>
  </header>

  <!-- 1. PITCHER REAL-TIME AUTO-CORRECT STRIP -->
  <div class="pitcher-bar">
    <div class="pitcher-tag">⚡ Pitcher RT Insert</div>

    <label>
      Root Key:
      <select
        value={telemetry.pitcherRootKey}
        onchange={(e) => {
          telemetry.pitcherRootKey = parseInt((e.currentTarget as HTMLSelectElement).value, 10);
          telemetry.syncPitcherRtParams();
        }}
      >
        {#each NOTE_NAMES as kName, idx}
          <option value={idx}>{kName}</option>
        {/each}
      </select>
    </label>

    <label>
      Scale:
      <select
        value={telemetry.pitcherScaleIndex}
        onchange={(e) => {
          telemetry.pitcherScaleIndex = parseInt((e.currentTarget as HTMLSelectElement).value, 10);
          telemetry.syncPitcherRtParams();
        }}
      >
        {#each SCALE_NAMES as sName, idx}
          <option value={idx}>{sName}</option>
        {/each}
      </select>
    </label>

    <label>
      Retune: {telemetry.pitcherRetuneSpeedMs.toFixed(0)} ms
      <input
        type="range"
        min="0"
        max="200"
        step="1"
        bind:value={telemetry.pitcherRetuneSpeedMs}
        oninput={() => telemetry.syncPitcherRtParams()}
      />
    </label>

    <label>
      Humanize: {(telemetry.pitcherHumanize * 100).toFixed(0)}%
      <input
        type="range"
        min="0"
        max="1"
        step="0.02"
        bind:value={telemetry.pitcherHumanize}
        oninput={() => telemetry.syncPitcherRtParams()}
      />
    </label>

    <label>
      Formant: {telemetry.pitcherFormantShiftSemitones >= 0 ? '+' : ''}{telemetry.pitcherFormantShiftSemitones.toFixed(1)} st
      <input
        type="range"
        min="-12"
        max="12"
        step="0.5"
        bind:value={telemetry.pitcherFormantShiftSemitones}
        oninput={() => telemetry.syncPitcherRtParams()}
      />
    </label>

    <label>
      <input
        type="checkbox"
        bind:checked={telemetry.pitcherFormantPreserve}
        onchange={() => telemetry.syncPitcherRtParams()}
      />
      Formant Preserve
    </label>

    <label>
      <input
        type="checkbox"
        bind:checked={telemetry.pitcherBypass}
        onchange={() => telemetry.syncPitcherRtParams()}
      />
      Bypass
    </label>
  </div>

  <div class="status-banner">
    <span>Trạng thái: <strong>{telemetry.vocalStatusMessage}</strong></span>
    <span>
      Blobs: <strong>{telemetry.vocalBlobs.length}</strong> · SR:
      <strong>{(telemetry.vocalSampleRateHz / 1000).toFixed(1)} kHz</strong>
    </span>
  </div>

  <div class="view-nav">
    <button onclick={() => setZoom(viewZoom / 2)} title="Thu nhỏ">－</button>
    <button onclick={() => setZoom(viewZoom * 2)} title="Phóng to">＋</button>
    <button onclick={() => scrollView(-0.5)} title="Cuộn trái">◀</button>
    <button onclick={() => scrollView(0.5)} title="Cuộn phải">▶</button>
    <span>Zoom ×{viewZoom} · {visibleBlobs.length}/{telemetry.vocalBlobs.length} nốt hiển thị</span>
  </div>

  <!-- 2. NEWTONE GRAPHICAL PITCH & TIME BLOB CANVAS -->
  <div class="newtone-editor">
    <div class="piano-gutter">
      {#each MIDI_ROWS as m}
        <div class="piano-key {isBlackKey(m) ? 'black' : 'white'}">
          <span>{midiLabel(m)}</span>
        </div>
      {/each}
    </div>

    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="blob-canvas"
      bind:this={canvasWrapEl}
      onpointermove={handleCanvasPointerMove}
      onpointerup={stopBlobDrag}
      onpointerleave={stopBlobDrag}
    >
      {#each MIDI_ROWS as m}
        <div class="grid-row {isBlackKey(m) ? 'dark' : ''}"></div>
      {/each}

      <!-- Pitch Contour Overlay SVG -->
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" class="contour-svg">
        {#each visibleBlobs as b (b.blob_id)}
          <polyline
            points={buildContourPolyline(b)}
            fill="none"
            stroke="#fbbf24"
            stroke-width="0.65"
          />
        {/each}
      </svg>

      <!-- Draggable Vocal Note Blobs -->
      {#each visibleBlobs as blob (blob.blob_id)}
        {@const leftPct = sampleToPct(blob.warped_start_sample)}
        {@const widthPct = Math.max(0.5, ((blob.warped_end_sample - blob.warped_start_sample) / viewSpan) * 100)}
        {@const topPct = midiToTopPct(blob.target_midi_note + blob.fine_pitch_cents / 100)}
        {@const rowHeightPct = 100 / MIDI_ROWS.length}
        <div
          class="vocal-blob {telemetry.vocalSelectedBlobId === blob.blob_id ? 'selected' : ''}"
          style="left: {leftPct}%; width: {widthPct}%; top: {topPct}%; height: {rowHeightPct}%;"
          onpointerdown={(e) => startBlobDrag(e, blob, 'pitch')}
          title="Blob #{blob.blob_id + 1}: Detected {blob.detected_midi_note.toFixed(2)} → Target {midiLabel(blob.target_midi_note)} ({blob.fine_pitch_cents} ct)"
        >
          <span class="blob-label">
            {midiLabel(blob.target_midi_note)}
            {#if blob.formant_shift_semitones !== 0}
              · F{blob.formant_shift_semitones > 0 ? '+' : ''}{blob.formant_shift_semitones.toFixed(1)}
            {/if}
          </span>
          <span
            class="warp-handle"
            onpointerdown={(e) => startBlobDrag(e, blob, 'warp')}
            title="Kéo mép phải để co giãn độ dài nốt hát (Time-Warp)"
          ></span>
        </div>
      {/each}
    </div>
  </div>

  <!-- 3. SELECTED VOCAL BLOB INSPECTOR -->
  {#if selectedBlob}
    <div class="blob-inspector">
      <strong>
        Selected Blob #{selectedBlob.blob_id + 1} ({midiLabel(selectedBlob.target_midi_note)} · Raw {selectedBlob.detected_midi_note.toFixed(2)})
      </strong>

      <label>
        Target MIDI ({midiLabel(selectedBlob.target_midi_note)}):
        <input
          type="range"
          min={MIN_MIDI}
          max={MAX_MIDI}
          step="1"
          value={selectedBlob.target_midi_note}
          oninput={(e) =>
            updateBlob(selectedBlob.blob_id, {
              target_midi_note: parseInt((e.currentTarget as HTMLInputElement).value, 10),
            })}
        />
      </label>

      <label>
        Fine Cents: {selectedBlob.fine_pitch_cents >= 0 ? '+' : ''}{selectedBlob.fine_pitch_cents.toFixed(0)} ct
        <input
          type="range"
          min="-100"
          max="100"
          step="2"
          value={selectedBlob.fine_pitch_cents}
          oninput={(e) =>
            updateBlob(selectedBlob.blob_id, {
              fine_pitch_cents: parseFloat((e.currentTarget as HTMLInputElement).value),
            })}
        />
      </label>

      <label>
        Flatten Drift: {(selectedBlob.pitch_drift_correction * 100).toFixed(0)}%
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={selectedBlob.pitch_drift_correction}
          oninput={(e) =>
            updateBlob(selectedBlob.blob_id, {
              pitch_drift_correction: parseFloat((e.currentTarget as HTMLInputElement).value),
            })}
        />
      </label>

      <label>
        Vibrato: {(selectedBlob.vibrato_scaling * 100).toFixed(0)}%
        <input
          type="range"
          min="0"
          max="2"
          step="0.05"
          value={selectedBlob.vibrato_scaling}
          oninput={(e) =>
            updateBlob(selectedBlob.blob_id, {
              vibrato_scaling: parseFloat((e.currentTarget as HTMLInputElement).value),
            })}
        />
      </label>

      <label>
        Blob Formant: {selectedBlob.formant_shift_semitones >= 0 ? '+' : ''}{selectedBlob.formant_shift_semitones.toFixed(1)} st
        <input
          type="range"
          min="-12"
          max="12"
          step="0.5"
          value={selectedBlob.formant_shift_semitones}
          oninput={(e) =>
            updateBlob(selectedBlob.blob_id, {
              formant_shift_semitones: parseFloat((e.currentTarget as HTMLInputElement).value),
            })}
        />
      </label>
    </div>
  {/if}
</section>

<style>
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; display: flex; flex-direction: column; gap: 10px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
  h2 { margin: 0; font-size: 1rem; color: var(--accent); font-weight: 600; }
  .subtitle { font-size: 0.75rem; color: var(--muted); }
  .action-buttons { display: flex; gap: 6px; flex-wrap: wrap; }
  .act-btn { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; font-size: 0.75rem; font-weight: 600; cursor: pointer; }
  .act-btn:hover { border-color: var(--accent); }
  .act-btn.primary { background: rgba(118, 210, 219, 0.18); color: var(--accent); border-color: var(--accent); }
  .act-btn.commit { background: var(--ok); color: #0b2b1e; border-color: var(--ok); font-weight: 700; }
  .pitcher-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; background: var(--sub-surface-2); border: 1px solid var(--border); border-radius: var(--radius); padding: 8px 12px; font-size: 0.75rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  .pitcher-tag { background: var(--warn); color: #0b2b1e; padding: 3px 8px; border-radius: 5px; font-weight: 700; font-size: 0.72rem; }
  .pitcher-bar select { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 5px; padding: 3px 7px; margin-left: 4px; font-size: 0.73rem; }
  .status-banner { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 0.74rem; font-family: var(--font-mono); color: var(--muted); background: var(--sub-surface); border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; }
  .status-banner strong { color: var(--accent); }
  .newtone-editor { display: grid; grid-template-columns: 56px 1fr; border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; height: 340px; background: var(--sub-surface); }
  .piano-gutter { display: flex; flex-direction: column; border-right: 1px solid var(--border); }
  .piano-key { flex: 1; display: flex; align-items: center; justify-content: center; font-size: 0.66rem; font-family: var(--font-mono); border-bottom: 1px solid rgba(118, 210, 219, 0.08); }
  .piano-key.white { background: var(--surface); color: var(--text); }
  .piano-key.black { background: #0d1618; color: var(--muted); }
  .blob-canvas { position: relative; display: flex; flex-direction: column; user-select: none; overflow: hidden; }
  .grid-row { flex: 1; border-bottom: 1px solid rgba(118, 210, 219, 0.06); }
  .grid-row.dark { background: rgba(0, 0, 0, 0.22); }
  .contour-svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 3; }
  .vocal-blob { position: absolute; background: rgba(118, 210, 219, 0.34); border: 1.5px solid var(--accent); border-radius: 5px; display: flex; align-items: center; justify-content: space-between; padding: 0 6px; box-sizing: border-box; cursor: ns-resize; z-index: 2; }
  .vocal-blob.selected { background: rgba(74, 222, 128, 0.38); border-color: var(--ok); box-shadow: 0 0 10px rgba(74, 222, 128, 0.35); }
  .blob-label { font-size: 0.66rem; font-family: var(--font-mono); font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .warp-handle { width: 6px; height: 100%; cursor: ew-resize; background: rgba(255, 255, 255, 0.25); border-radius: 0 4px 4px 0; }
  .blob-inspector { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; background: var(--sub-surface-2); border: 1px solid var(--border); border-radius: var(--radius); padding: 8px 12px; font-size: 0.74rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  .blob-inspector strong { color: var(--ok); }
  .view-nav { display: flex; align-items: center; gap: 6px; font-size: 0.7rem; color: var(--muted); margin: 4px 0; }
  .view-nav button { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 1px 8px; cursor: pointer; }
</style>
