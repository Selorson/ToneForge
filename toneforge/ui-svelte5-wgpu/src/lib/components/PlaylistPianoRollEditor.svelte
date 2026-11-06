<script lang="ts">
  import {
    DawTelemetryView,
    FIXED_POINT_30BIT_MAX,
    type PianoRollNoteUiState,
  } from '../state/daw_master_store.svelte';

  let { telemetry }: { telemetry: DawTelemetryView } = $props();

  const PPQ_OPTIONS = [96, 144, 192, 384, 480, 960] as const;
  const TIMELINE_TICKS = 384;
  const PIANO_KEYS = [
    { key: 75, label: 'D#5' },
    { key: 74, label: 'D5' },
    { key: 72, label: 'C5' },
    { key: 71, label: 'B4' },
    { key: 69, label: 'A4' },
    { key: 67, label: 'G4' },
    { key: 65, label: 'F4' },
    { key: 63, label: 'D#4' },
    { key: 62, label: 'D4' },
    { key: 60, label: 'C4' },
    { key: 58, label: 'A#3' },
    { key: 55, label: 'G3' },
  ];

  const COLOR_GROUP_PALETTE = [
    '#10b981',
    '#38bdf8',
    '#f59e0b',
    '#ec4899',
    '#a78bfa',
    '#22d3ee',
    '#fb7185',
    '#84cc16',
  ];

  let draggingNoteId = $state<number | null>(null);
  let dragMode = $state<'move' | 'resize' | null>(null);
  let dragStartX = $state(0);
  let dragOrigStartTick = $state(0);
  let dragOrigDuration = $state(48);

  let draggingSplineIdx = $state<number | null>(null);

  let activePattern = $derived(
    telemetry.pianoRollPatterns.find((p) => p.patternId === telemetry.selectedPatternId) ??
      telemetry.pianoRollPatterns[0]
  );

  let selectedNote = $derived<PianoRollNoteUiState | null>(
    activePattern?.notes.find((n) => n.id === telemetry.selectedNoteId) ??
      activePattern?.notes[0] ??
      null
  );

  let splineSvgPath = $derived.by(() => {
    const samples = 96;
    const coords: string[] = [];
    for (let i = 0; i <= samples; i++) {
      const tick = (i / samples) * TIMELINE_TICKS;
      const normY = telemetry.evaluateTensionSplineAt(tick);
      const xPct = (tick / TIMELINE_TICKS) * 100;
      const yPct = (1.0 - Math.max(0, Math.min(1, normY))) * 100;
      coords.push(`${i === 0 ? 'M' : 'L'} ${xPct.toFixed(2)} ${yPct.toFixed(2)}`);
    }
    return coords.join(' ');
  });

  function handleGridCellClick(key: number, tickStep: number) {
    if (!activePattern) return;
    const startTick = tickStep * 24;
    const existing = activePattern.notes.find(
      (n) => n.key === key && startTick >= n.startTick && startTick < n.startTick + n.durationTicks
    );
    if (existing) {
      telemetry.selectedNoteId = existing.id;
      return;
    }
    telemetry.addPianoRollNote(activePattern.patternId, {
      startTick,
      durationTicks: 48,
      key,
      velocity: 0.88,
      pan: 0.0,
      finePitchCents: 0.0,
      isSlide: telemetry.pianoRollSlideTool,
      colorGroup: telemetry.pianoRollColorGroup,
    });
  }

  function startNoteDrag(
    e: MouseEvent,
    note: PianoRollNoteUiState,
    mode: 'move' | 'resize'
  ) {
    e.stopPropagation();
    telemetry.selectedNoteId = note.id;
    draggingNoteId = note.id;
    dragMode = mode;
    dragStartX = e.clientX;
    dragOrigStartTick = note.startTick;
    dragOrigDuration = note.durationTicks;
  }

  function handlePianoRollMouseMove(e: MouseEvent) {
    if (draggingNoteId === null || !activePattern || !dragMode) return;
    const deltaPx = e.clientX - dragStartX;
    const deltaTicks = Math.round(deltaPx / 12) * 12;
    if (dragMode === 'move') {
      const nextStart = Math.max(0, Math.min(TIMELINE_TICKS - 12, dragOrigStartTick + deltaTicks));
      telemetry.updatePianoRollNote(activePattern.patternId, draggingNoteId, {
        startTick: nextStart,
      });
    } else {
      const nextDur = Math.max(12, Math.min(TIMELINE_TICKS, dragOrigDuration + deltaTicks));
      telemetry.updatePianoRollNote(activePattern.patternId, draggingNoteId, {
        durationTicks: nextDur,
      });
    }
  }

  function stopNoteDrag() {
    draggingNoteId = null;
    dragMode = null;
  }

  function handleSplineSvgDblClick(e: MouseEvent) {
    const svg = e.currentTarget as SVGSVGElement;
    const rect = svg.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const relY = Math.max(0, Math.min(1, 1.0 - (e.clientY - rect.top) / rect.height));
    telemetry.addTensionSplinePoint(Math.round(relX * TIMELINE_TICKS), Number(relY.toFixed(3)), 0.0);
  }

  function handleSplineMouseMove(e: MouseEvent) {
    if (draggingSplineIdx === null) return;
    const svg = e.currentTarget as SVGSVGElement;
    const rect = svg.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const relY = Math.max(0, Math.min(1, 1.0 - (e.clientY - rect.top) / rect.height));
    telemetry.updateTensionSplinePoint(draggingSplineIdx, {
      x: Math.round(relX * TIMELINE_TICKS),
      y: Number(relY.toFixed(3)),
    });
  }
</script>

<section class="panel sequencer-panel">
  <header class="panel-header">
    <div>
      <h2>Hybrid PPQ (96–960) Playlist Arrangement, TensionSpline Automation & FL26 Piano Roll</h2>
      <span class="subtitle">
        Sample-Accurate Event Slicer · Point 1 Generator Lookahead APDC · Slide/Portamento Glide · 30-Bit Fixed-Point (0x40000000) Spline
      </span>
    </div>

    <div class="toolbar">
      <label class="ppq-select">
        PPQ Resolution
        <select
          value={telemetry.ppqResolution}
          onchange={(e) =>
            telemetry.setPpqResolution(
              parseInt((e.currentTarget as HTMLSelectElement).value, 10) as any
            )}
        >
          {#each PPQ_OPTIONS as ppq}
            <option value={ppq}>{ppq} PPQ</option>
          {/each}
        </select>
      </label>

      <button
        class={telemetry.pianoRollSlideTool ? 'tool-btn active' : 'tool-btn'}
        onclick={() => (telemetry.pianoRollSlideTool = !telemetry.pianoRollSlideTool)}
      >
        ⊿ Slide / Portamento ({telemetry.pianoRollSlideTool ? 'ON' : 'OFF'})
      </button>

      <label class="color-select">
        Color Group
        <select
          value={telemetry.pianoRollColorGroup}
          onchange={(e) =>
            (telemetry.pianoRollColorGroup = parseInt(
              (e.currentTarget as HTMLSelectElement).value,
              10
            ))}
        >
          {#each [0, 1, 2, 3] as grp}
            <option value={grp}>Group {grp + 1}</option>
          {/each}
        </select>
      </label>
    </div>
  </header>

  <div class="sequencer-layout">
    <!-- 1. PLAYLIST ARRANGEMENT & INTERACTIVE TENSIONSPLINE AUTOMATION TIMELINE -->
    <div class="playlist-box">
      <div class="sub-header">
        <h3>Playlist Arrangement & TensionSpline Automation Clip Canvas</h3>
        <span class="fixed30-readout">
          30-Bit Bus #{telemetry.activeControllerId}:
          <strong>0x{telemetry.activeControllerFixed30.toString(16).toUpperCase()}</strong>
          ({((telemetry.activeControllerFixed30 / FIXED_POINT_30BIT_MAX) * 100).toFixed(1)}%)
        </span>
      </div>

      <div class="playlist-tracks">
        {#each telemetry.playlistClips as clip (clip.clipId)}
          <div class="playlist-row">
            <div class="row-meta">
              <strong>Track {clip.trackRow + 1} ({clip.kind})</strong>
              <button
                class={clip.muted ? 'mute-btn muted' : 'mute-btn'}
                onclick={() =>
                  telemetry.upsertPlaylistClip({ ...clip, muted: !clip.muted })}
              >
                {clip.muted ? 'Muted' : 'Active'}
              </button>
            </div>
            <div class="row-timeline">
              <div
                class="clip-block kind-{clip.kind.toLowerCase()}"
                style="left: {(clip.startTick / TIMELINE_TICKS) * 100}%; width: {(clip.durationTicks / TIMELINE_TICKS) * 100}%;"
              >
                <span>{clip.kind} #{clip.sourceId} ({clip.startTick}..{clip.startTick + clip.durationTicks} tk)</span>
              </div>
            </div>
          </div>
        {/each}
      </div>

      <!-- Interactive TensionSpline Curve Editor -->
      <div class="spline-canvas-wrap">
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <svg
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          class="spline-svg"
          ondblclick={handleSplineSvgDblClick}
          onmousemove={handleSplineMouseMove}
          onmouseup={() => (draggingSplineIdx = null)}
          onmouseleave={() => (draggingSplineIdx = null)}
        >
          <path d={splineSvgPath} fill="none" stroke="#38bdf8" stroke-width="1.6" />
          {#each telemetry.tensionSplinePoints as pt, idx}
            <circle
              cx={(pt.x / TIMELINE_TICKS) * 100}
              cy={(1.0 - pt.y) * 100}
              r="2.4"
              fill={draggingSplineIdx === idx ? '#f59e0b' : '#10b981'}
              stroke="#090d16"
              stroke-width="0.6"
              onmousedown={(e) => {
                e.stopPropagation();
                draggingSplineIdx = idx;
              }}
            />
          {/each}
        </svg>

        <div class="spline-points-bar">
          {#each telemetry.tensionSplinePoints as pt, idx}
            <div class="spline-pt-chip">
              <span>P{idx + 1} ({pt.x}tk, {(pt.y * 100).toFixed(0)}%)</span>
              <label>
                Tension {pt.tension.toFixed(2)}
                <input
                  type="range"
                  min="-0.95"
                  max="0.95"
                  step="0.05"
                  value={pt.tension}
                  oninput={(e) =>
                    telemetry.updateTensionSplinePoint(idx, {
                      tension: parseFloat((e.currentTarget as HTMLInputElement).value),
                    })}
                />
              </label>
              {#if telemetry.tensionSplinePoints.length > 2}
                <button
                  class="del-pt-btn"
                  onclick={() => telemetry.removeTensionSplinePoint(idx)}
                >
                  ×
                </button>
              {/if}
            </div>
          {/each}
        </div>
      </div>
    </div>

    <!-- 2. FL STUDIO 26 PIANO ROLL EDITOR -->
    <div class="pianoroll-box">
      <div class="sub-header">
        <h3>FL Studio 26 Piano Roll — {activePattern?.name}</h3>
        <div class="pattern-tabs">
          {#each telemetry.pianoRollPatterns as pat}
            <button
              class={telemetry.selectedPatternId === pat.patternId ? 'active' : ''}
              onclick={() => (telemetry.selectedPatternId = pat.patternId)}
            >
              Pattern {pat.patternId} ({pat.notes.length} notes)
            </button>
          {/each}
        </div>
      </div>

      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div
        class="pianoroll-grid"
        onmousemove={handlePianoRollMouseMove}
        onmouseup={stopNoteDrag}
        onmouseleave={stopNoteDrag}
      >
        {#each PIANO_KEYS as row}
          <div class="pr-row">
            <div class="pr-key-label">{row.label}</div>
            <div class="pr-cells">
              {#each Array(16) as _, stepIdx}
                <!-- svelte-ignore a11y_click_events_have_key_events -->
                <div
                  class="pr-cell"
                  onclick={() => handleGridCellClick(row.key, stepIdx)}
                ></div>
              {/each}

              {#if activePattern}
                {#each activePattern.notes.filter((n) => n.key === row.key) as note (note.id)}
                  <div
                    class="pr-note {telemetry.selectedNoteId === note.id ? 'selected' : ''} {note.isSlide ? 'slide-note' : ''}"
                    style="
                      left: {(note.startTick / TIMELINE_TICKS) * 100}%;
                      width: {(note.durationTicks / TIMELINE_TICKS) * 100}%;
                      background: {COLOR_GROUP_PALETTE[note.colorGroup % COLOR_GROUP_PALETTE.length]};
                    "
                    onmousedown={(e) => startNoteDrag(e, note, 'move')}
                    oncontextmenu={(e) => {
                      e.preventDefault();
                      telemetry.removePianoRollNote(activePattern.patternId, note.id);
                    }}
                  >
                    <span class="note-caption">
                      {note.isSlide ? '⊿ Slide' : row.label}
                    </span>
                    <span
                      class="resize-handle"
                      onmousedown={(e) => startNoteDrag(e, note, 'resize')}
                    ></span>
                  </div>
                {/each}
              {/if}
            </div>
          </div>
        {/each}
      </div>

      <!-- Per-Note Expression Inspector (Velocity, Pan, Fine Pitch, Slide, Delete) -->
      {#if selectedNote && activePattern}
        <div class="note-inspector">
          <span>
            Selected Note #{selectedNote.id} (Key {selectedNote.key}, Start {selectedNote.startTick}tk, Dur {selectedNote.durationTicks}tk)
          </span>
          <label>
            Velocity {(selectedNote.velocity * 100).toFixed(0)}%
            <input
              type="range"
              min="0.05"
              max="1.0"
              step="0.01"
              value={selectedNote.velocity}
              oninput={(e) =>
                telemetry.updatePianoRollNote(activePattern.patternId, selectedNote.id, {
                  velocity: parseFloat((e.currentTarget as HTMLInputElement).value),
                })}
            />
          </label>
          <label>
            Pan {selectedNote.pan.toFixed(2)}
            <input
              type="range"
              min="-1.0"
              max="1.0"
              step="0.05"
              value={selectedNote.pan}
              oninput={(e) =>
                telemetry.updatePianoRollNote(activePattern.patternId, selectedNote.id, {
                  pan: parseFloat((e.currentTarget as HTMLInputElement).value),
                })}
            />
          </label>
          <label>
            Fine Pitch {selectedNote.finePitchCents.toFixed(0)} ct
            <input
              type="range"
              min="-200"
              max="200"
              step="5"
              value={selectedNote.finePitchCents}
              oninput={(e) =>
                telemetry.updatePianoRollNote(activePattern.patternId, selectedNote.id, {
                  finePitchCents: parseFloat((e.currentTarget as HTMLInputElement).value),
                })}
            />
          </label>
          <label>
            <input
              type="checkbox"
              checked={selectedNote.isSlide}
              onchange={(e) =>
                telemetry.updatePianoRollNote(activePattern.patternId, selectedNote.id, {
                  isSlide: (e.currentTarget as HTMLInputElement).checked,
                })}
            />
            ⊿ Slide Note
          </label>
          <button
            class="delete-note-btn"
            onclick={() => telemetry.removePianoRollNote(activePattern.patternId, selectedNote.id)}
          >
            Delete Note
          </button>
        </div>
      {/if}
    </div>
  </div>
</section>

<style>
  .panel { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
  h2 { margin: 0; font-size: 1rem; color: #38bdf8; }
  .subtitle { font-size: 0.75rem; color: #94a3b8; }
  .toolbar { display: flex; align-items: center; gap: 8px; font-size: 0.76rem; }
  .toolbar select { background: #090d16; color: #f8fafc; border: 1px solid #334155; border-radius: 4px; padding: 4px 8px; }
  .tool-btn { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 4px; padding: 5px 10px; cursor: pointer; font-weight: 600; font-size: 0.75rem; }
  .tool-btn.active { background: #f59e0b; color: #090d16; border-color: #f59e0b; }
  .sequencer-layout { display: grid; grid-template-columns: 1fr 1.25fr; gap: 12px; }
  .playlist-box, .pianoroll-box { background: #090d16; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; display: flex; flex-direction: column; gap: 8px; }
  .sub-header { display: flex; justify-content: space-between; align-items: center; }
  h3 { margin: 0; font-size: 0.84rem; color: #e2e8f0; }
  .fixed30-readout { font-size: 0.74rem; font-family: monospace; color: #94a3b8; }
  .fixed30-readout strong { color: #38bdf8; }
  .playlist-tracks { display: flex; flex-direction: column; gap: 5px; }
  .playlist-row { display: grid; grid-template-columns: 150px 1fr; gap: 8px; align-items: center; font-size: 0.74rem; }
  .row-meta { display: flex; justify-content: space-between; align-items: center; background: #0f172a; padding: 4px 8px; border-radius: 4px; }
  .mute-btn { background: #10b981; color: #090d16; border: none; border-radius: 3px; padding: 2px 6px; font-size: 0.68rem; font-weight: 700; cursor: pointer; }
  .mute-btn.muted { background: #475569; color: #cbd5e1; }
  .row-timeline { position: relative; height: 24px; background: #0f172a; border: 1px solid #1e293b; border-radius: 4px; overflow: hidden; }
  .clip-block { position: absolute; top: 2px; bottom: 2px; border-radius: 3px; padding: 2px 6px; font-size: 0.7rem; font-weight: 600; color: #090d16; display: flex; align-items: center; }
  .clip-block.kind-pattern { background: #38bdf8; }
  .clip-block.kind-audiostem { background: #a78bfa; }
  .clip-block.kind-automation { background: #10b981; }
  .spline-canvas-wrap { display: flex; flex-direction: column; gap: 6px; }
  .spline-svg { width: 100%; height: 92px; background: #0f172a; border: 1px solid #1e293b; border-radius: 4px; cursor: crosshair; }
  .spline-points-bar { display: flex; flex-wrap: wrap; gap: 6px; }
  .spline-pt-chip { background: #0f172a; border: 1px solid #1e293b; border-radius: 4px; padding: 4px 6px; font-size: 0.7rem; display: flex; align-items: center; gap: 6px; }
  .del-pt-btn { background: #ef4444; color: #fff; border: none; border-radius: 3px; cursor: pointer; padding: 0 5px; }
  .pattern-tabs { display: flex; gap: 4px; }
  .pattern-tabs button { background: #1e293b; color: #94a3b8; border: none; border-radius: 4px; padding: 3px 8px; font-size: 0.72rem; cursor: pointer; }
  .pattern-tabs button.active { background: #38bdf8; color: #090d16; font-weight: 700; }
  .pianoroll-grid { display: flex; flex-direction: column; border: 1px solid #1e293b; border-radius: 4px; overflow: hidden; }
  .pr-row { display: grid; grid-template-columns: 46px 1fr; height: 22px; border-bottom: 1px solid #131c31; }
  .pr-key-label { background: #0f172a; color: #94a3b8; font-size: 0.68rem; font-family: monospace; display: flex; align-items: center; justify-content: center; border-right: 1px solid #1e293b; }
  .pr-cells { position: relative; display: grid; grid-template-columns: repeat(16, 1fr); background: #090d16; }
  .pr-cell { border-right: 1px solid #111827; cursor: pointer; }
  .pr-note { position: absolute; top: 2px; bottom: 2px; border-radius: 3px; color: #090d16; font-size: 0.66rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between; padding: 0 4px; cursor: grab; box-sizing: border-box; }
  .pr-note.selected { outline: 2px solid #f8fafc; }
  .pr-note.slide-note { border-left: 4px solid #090d16; }
  .resize-handle { width: 5px; height: 100%; cursor: ew-resize; }
  .note-inspector { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; background: #0f172a; border: 1px solid #1e293b; border-radius: 4px; padding: 6px 8px; font-size: 0.72rem; }
  .delete-note-btn { background: #ef4444; color: #ffffff; border: none; border-radius: 4px; padding: 4px 8px; font-size: 0.72rem; cursor: pointer; font-weight: 600; }
</style>
