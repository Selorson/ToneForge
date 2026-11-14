<script lang="ts">
  import {
    DawTelemetryView,
    FIXED_POINT_30BIT_MAX,
    type PianoRollNoteUiState,
  } from '../state/daw_master_store.svelte';
  import Guide from './Guide.svelte';

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

  let selectedClip = $derived(
    telemetry.playlistClips.find((c) => c.clipId === telemetry.selectedClipId) ??
      telemetry.playlistClips[0] ??
      null
  );
  // Stretch / Pitch trigger a phase-vocoder render: commit on release (onchange), preview the
  // value while dragging.
  let stretchPreview = $state<number | null>(null);
  let pitchPreview = $state<number | null>(null);

  let ghostNotes = $derived(
    telemetry.pianoRollShowGhostNotes
      ? telemetry.pianoRollPatterns
          .filter((p) => p.patternId !== telemetry.selectedPatternId)
          .flatMap((p) => p.notes)
      : []
  );

  // Viewport culling: only notes intersecting the visible tick window are turned into DOM nodes,
  // bucketed by key once per change (O(notes)), so dense patterns / full MIDI scores stay at 120 FPS.
  let viewStartTick = $state(0);
  const MAX_GHOST_NOTES_RENDERED = 2_000;
  function inView(startTick: number, durationTicks: number): boolean {
    return startTick < viewStartTick + TIMELINE_TICKS && startTick + durationTicks > viewStartTick;
  }
  let visibleNotesByKey = $derived.by(() => {
    const m = new Map<number, PianoRollNoteUiState[]>();
    for (const n of activePattern?.notes ?? []) {
      if (!inView(n.startTick, n.durationTicks)) continue;
      const list = m.get(n.key);
      if (list) list.push(n);
      else m.set(n.key, [n]);
    }
    return m;
  });
  let visibleGhostsByKey = $derived.by(() => {
    const m = new Map<number, PianoRollNoteUiState[]>();
    let count = 0;
    for (const n of ghostNotes) {
      if (count >= MAX_GHOST_NOTES_RENDERED) break;
      if (!inView(n.startTick, n.durationTicks)) continue;
      count++;
      const list = m.get(n.key);
      if (list) list.push(n);
      else m.set(n.key, [n]);
    }
    return m;
  });
  let visibleNotes = $derived((activePattern?.notes ?? []).filter((n) => inView(n.startTick, n.durationTicks)));
  function scrollView(deltaTicks: number) {
    const maxStart = Math.max(0, (activePattern?.lengthTicks ?? TIMELINE_TICKS) - TIMELINE_TICKS);
    viewStartTick = Math.max(0, Math.min(maxStart, viewStartTick + deltaTicks));
  }

  let lassoStart = $state<{ xPct: number; key: number } | null>(null);
  let lassoCurrent = $state<{ xPct: number; key: number } | null>(null);

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

  async function handleGridCellClick(key: number, tickStep: number) {
    let targetPattern = activePattern;
    if (!targetPattern) {
      const createdId = await telemetry.createPatternAndOptionalClip(true);
      targetPattern =
        telemetry.pianoRollPatterns.find((p) => p.patternId === createdId) ??
        telemetry.pianoRollPatterns[0];
    }
    if (!targetPattern) return;
    const startTick = viewStartTick + tickStep * 24;
    const existing = targetPattern.notes.find(
      (n) => n.key === key && startTick >= n.startTick && startTick < n.startTick + n.durationTicks
    );
    if (existing) {
      telemetry.selectedNoteId = existing.id;
      telemetry.auditionNote(existing.key, existing.velocity, existing.finePitchCents, existing.pan);
      return;
    }
    await telemetry.addPianoRollNote(targetPattern.patternId, {
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
    telemetry.auditionNote(note.key, note.velocity, note.finePitchCents, note.pan);
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
      const patLen = Math.max(TIMELINE_TICKS, activePattern.lengthTicks || TIMELINE_TICKS);
      const nextStart = Math.max(0, Math.min(patLen - 12, dragOrigStartTick + deltaTicks));
      telemetry.updatePianoRollNote(activePattern.patternId, draggingNoteId, {
        startTick: nextStart,
      });
    } else {
      const patLen = Math.max(TIMELINE_TICKS, activePattern.lengthTicks || TIMELINE_TICKS);
      const nextDur = Math.max(12, Math.min(patLen, dragOrigDuration + deltaTicks));
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
  <Guide id="playlist" />
  <header class="panel-header">
    <div>
      <h2>Hybrid PPQ (96–960) Playlist Arrangement, TensionSpline Automation & FL26 Piano Roll</h2>
      <span class="subtitle">
        Sample-Accurate Event Slicer · Point 1 Generator Lookahead APDC · Slide/Portamento Glide · 30-Bit Fixed-Point (0x40000000) Spline
      </span>
    </div>

    <div class="toolbar">
      <label class="ppq-select">
        FL26 Sample & Synth Kit
        <select
          value={telemetry.fl26SelectedSampleKit}
          onchange={(e) =>
            telemetry.loadFl26SampleKit((e.currentTarget as HTMLSelectElement).value)}
        >
          {#each [
            '808 Studio Kit (Real .wav)',
            '909 Analog Kit (Real .wav)',
            'MA Skipper Snare Kit',
            'Symphonic Strings (SYM_STR_C_2)',
            'Sytrus / Harmor Lead',
          ] as kit}
            <option value={kit}>{kit}</option>
          {/each}
        </select>
      </label>

      <label class="ppq-select">
        FL26 Piano Roll Score (.fsc)
        <select
          value={telemetry.fl26SelectedScorePreset}
          onchange={(e) =>
            telemetry.loadFl26ScorePreset((e.currentTarget as HTMLSelectElement).value)}
        >
          {#each [
            'ToneForge Lead Hook (Slide)',
            'FL26 Trance Arp (.fsc)',
            'FL26 Oxygene Progression (.fsc)',
            'FL26 4-Steps Groove (.fsc)',
            'FL26 DrumRoll (.fsc)',
          ] as score}
            <option value={score}>{score}</option>
          {/each}
        </select>
      </label>

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
        <div class="playlist-pro-tools">
          {#each [
            ['draw', '✏ Draw'],
            ['razor', '✂ Razor Slice'],
            ['slip', '↔ Slip Trim'],
          ] as [tKey, tLabel]}
            <button
              class={telemetry.playlistTool === tKey ? 'mini-tool active' : 'mini-tool'}
              onclick={() => (telemetry.playlistTool = tKey as 'draw' | 'razor' | 'slip')}
            >
              {tLabel}
            </button>
          {/each}
          {#if selectedClip}
            <button
              class="mini-tool"
              onclick={() => telemetry.sliceOrEditPlaylistClip(selectedClip.clipId, 'duplicate')}
              title="Nhân bản Clip đang chọn"
            >
              ⧉ Duplicate
            </button>
            <button
              class="mini-tool"
              onclick={() =>
                telemetry.sliceOrEditPlaylistClip(selectedClip.clipId, 'slice', {
                  splitTick: selectedClip.startTick + Math.floor(selectedClip.durationTicks / 2),
                })}
              title="Cắt đôi Clip đang chọn với 2ms Equal-Power Micro-Fade"
            >
              ✂ Split ½
            </button>
          {/if}
        </div>
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
              <strong>
                Track {clip.trackRow + 1} ({clip.kind})
                {#if (clip.takeLaneId ?? 0) > 0}
                  <span class="take-pill">Take {clip.takeLaneId}</span>
                {/if}
              </strong>
              <button
                class={clip.muted ? 'mute-btn muted' : 'mute-btn'}
                onclick={() =>
                  telemetry.upsertPlaylistClip({ ...clip, muted: !clip.muted })}
              >
                {clip.muted ? 'Muted' : 'Active'}
              </button>
            </div>
            <div class="row-timeline">
              <!-- svelte-ignore a11y_click_events_have_key_events -->
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <div
                class="clip-block kind-{clip.kind.toLowerCase()} {telemetry.selectedClipId === clip.clipId ? 'selected-clip' : ''}"
                style="left: {(clip.startTick / TIMELINE_TICKS) * 100}%; width: {(clip.durationTicks / TIMELINE_TICKS) * 100}%;"
                onclick={(e) => {
                  telemetry.selectedClipId = clip.clipId;
                  if (telemetry.playlistTool === 'razor') {
                    const rect = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
                    const clickTick = Math.round(((e.clientX - rect.left) / Math.max(1, rect.width)) * TIMELINE_TICKS);
                    telemetry.sliceOrEditPlaylistClip(clip.clipId, 'slice', { splitTick: clickTick });
                  }
                }}
              >
                <span>
                  {clip.kind} #{clip.sourceId} ({clip.startTick}..{clip.startTick + clip.durationTicks} tk)
                  {#if (clip.stretchRatio ?? 1) !== 1}
                    · {clip.stretchRatio?.toFixed(2)}×
                  {/if}
                  {#if (clip.pitchSemitones ?? 0) !== 0}
                    · {(clip.pitchSemitones ?? 0) > 0 ? '+' : ''}{clip.pitchSemitones}st
                  {/if}
                </span>
              </div>
            </div>
          </div>
        {:else}
          <div class="playlist-empty-state">
            <span>Playlist trống — chưa có Pattern, Audio hay Automation Clip nào.</span>
            <div class="playlist-empty-actions">
              <button
                class="mini-tool"
                onclick={() => telemetry.placeActivePatternOnPlaylist()}
                title="Tạo hoặc đặt Pattern lên Track 1 của Playlist"
              >
                + Thêm Pattern Clip
              </button>
              <button
                class="mini-tool"
                onclick={() => telemetry.addAutomationClipToPlaylist()}
                title="Đặt Automation Clip cho đường TensionSpline lên Playlist"
              >
                + Thêm Automation Clip
              </button>
            </div>
          </div>
        {/each}
      </div>

      {#if selectedClip}
        <div class="clip-inspector-bar">
          <strong>Clip #{selectedClip.clipId} ({selectedClip.kind}):</strong>
          <label>
            Gain {(selectedClip.gain * 100).toFixed(0)}%
            <input
              type="range"
              min="0"
              max="2"
              step="0.05"
              value={selectedClip.gain}
              oninput={(e) =>
                telemetry.sliceOrEditPlaylistClip(selectedClip.clipId, 'update_props', {
                  clipGainLinear: parseFloat((e.currentTarget as HTMLInputElement).value),
                })}
            />
          </label>
          <label>
            Fade In {selectedClip.fadeInSamples ?? 0} smp
            <input
              type="range"
              min="0"
              max="4800"
              step="96"
              value={selectedClip.fadeInSamples ?? 0}
              oninput={(e) =>
                telemetry.sliceOrEditPlaylistClip(selectedClip.clipId, 'update_props', {
                  fadeInSamples: parseInt((e.currentTarget as HTMLInputElement).value, 10),
                })}
            />
          </label>
          <label>
            Fade Out {selectedClip.fadeOutSamples ?? 0} smp
            <input
              type="range"
              min="0"
              max="4800"
              step="96"
              value={selectedClip.fadeOutSamples ?? 0}
              oninput={(e) =>
                telemetry.sliceOrEditPlaylistClip(selectedClip.clipId, 'update_props', {
                  fadeOutSamples: parseInt((e.currentTarget as HTMLInputElement).value, 10),
                })}
            />
          </label>
          <label>
            Stretch {(stretchPreview ?? selectedClip.stretchRatio ?? 1.0).toFixed(2)}×
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.05"
              value={selectedClip.stretchRatio ?? 1.0}
              oninput={(e) => (stretchPreview = parseFloat((e.currentTarget as HTMLInputElement).value))}
              onchange={(e) => {
                stretchPreview = null;
                telemetry.sliceOrEditPlaylistClip(selectedClip.clipId, 'update_props', {
                  stretchRatio: parseFloat((e.currentTarget as HTMLInputElement).value),
                });
              }}
            />
          </label>
          <label>
            Pitch {(pitchPreview ?? selectedClip.pitchSemitones ?? 0) >= 0 ? '+' : ''}{(pitchPreview ?? selectedClip.pitchSemitones ?? 0).toFixed(0)} st
            <input
              type="range"
              min="-12"
              max="12"
              step="1"
              value={selectedClip.pitchSemitones ?? 0}
              oninput={(e) => (pitchPreview = parseFloat((e.currentTarget as HTMLInputElement).value))}
              onchange={(e) => {
                pitchPreview = null;
                telemetry.sliceOrEditPlaylistClip(selectedClip.clipId, 'update_props', {
                  pitchSemitones: parseFloat((e.currentTarget as HTMLInputElement).value),
                });
              }}
            />
          </label>
          {#if telemetry.derivedRendersPending > 0}
            <span class="render-pending" title="Stretch / Pitch được tính ở luồng nền; bản cũ vẫn phát cho tới khi xong">
              ⏳ Đang tính Stretch/Pitch ({telemetry.derivedRendersPending} clip)…
            </span>
          {/if}
        </div>
      {/if}

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
          <path d={splineSvgPath} fill="none" stroke="#76D2DB" stroke-width="1.6" />
          {#each telemetry.tensionSplinePoints as pt, idx}
            <circle
              cx={(pt.x / TIMELINE_TICKS) * 100}
              cy={(1.0 - pt.y) * 100}
              r="2.4"
              fill={draggingSplineIdx === idx ? '#fbbf24' : '#4ade80'}
              stroke="#081112"
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
          <button
            class={telemetry.pianoRollTool === 'draw' ? 'active' : ''}
            onclick={() => (telemetry.pianoRollTool = 'draw')}
          >
            ✏ Draw
          </button>
          <button
            class={telemetry.pianoRollTool === 'lasso' ? 'active' : ''}
            onclick={() => (telemetry.pianoRollTool = 'lasso')}
            title="Kéo chuột quét vùng chọn nhiều nốt (Box Lasso)"
          >
            ⬚ Lasso
          </button>
          <button
            class={telemetry.pianoRollShowGhostNotes ? 'active' : ''}
            onclick={() => (telemetry.pianoRollShowGhostNotes = !telemetry.pianoRollShowGhostNotes)}
            title="Hiển thị bóng mờ nốt nhạc từ các Pattern khác (Ghost Notes)"
          >
            👻 Ghost
          </button>
          {#if activePattern}
            <button
              onclick={() =>
                telemetry.transformPianoRollNotes(activePattern.patternId, 'quantize', {
                  noteIds: telemetry.selectedNoteIds.length > 0 ? telemetry.selectedNoteIds : undefined,
                  gridTicks: 24,
                  strength: 1.0,
                })}
              title="Căn chỉnh nốt vào đúng vạch nhịp 1/16 (Quantize)"
            >
              🧲 Quantize
            </button>
            <button
              onclick={() =>
                telemetry.transformPianoRollNotes(activePattern.patternId, 'strum', {
                  noteIds: telemetry.selectedNoteIds.length > 0 ? telemetry.selectedNoteIds : undefined,
                  strumOffsetTicks: 8,
                })}
              title="Rải hợp âm tự nhiên như gảy đàn Guitar/Harp (Strum)"
            >
              🎸 Strum
            </button>
            <button
              onclick={() =>
                telemetry.transformPianoRollNotes(activePattern.patternId, 'arpeggiate', {
                  noteIds: telemetry.selectedNoteIds.length > 0 ? telemetry.selectedNoteIds : undefined,
                  gridTicks: 24,
                })}
              title="Băm hợp âm thành chuỗi nốt chạy Arpeggio"
            >
              🎹 Arp
            </button>
          {/if}
          {#each telemetry.pianoRollPatterns as pat}
            <button
              class={telemetry.selectedPatternId === pat.patternId ? 'active' : ''}
              onclick={() => (telemetry.selectedPatternId = pat.patternId)}
            >
              Pattern {pat.patternId} ({pat.notes.length} notes)
            </button>
          {/each}
          <button
            onclick={() => telemetry.createPatternAndOptionalClip(true)}
            title="Tạo Pattern mới và tự động đặt lên Playlist"
          >
            + Pattern mới
          </button>
          {#if telemetry.pianoRollPatterns.length > 0}
            <button
              onclick={() => telemetry.placeActivePatternOnPlaylist()}
              title="Đặt thêm một bản clip của Pattern đang chọn lên Playlist"
            >
              + Đặt Pattern lên Playlist
            </button>
          {/if}
        </div>
      </div>

      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div
        class="pianoroll-grid"
        onmousedown={(e) => {
          if (telemetry.pianoRollTool !== 'lasso') return;
          const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
          const xPct = Math.max(0, Math.min(100, ((e.clientX - rect.left - 46) / Math.max(1, rect.width - 46)) * 100));
          const rowIdx = Math.max(0, Math.min(PIANO_KEYS.length - 1, Math.floor(((e.clientY - rect.top) / Math.max(1, rect.height)) * PIANO_KEYS.length)));
          lassoStart = { xPct, key: PIANO_KEYS[rowIdx].key };
          lassoCurrent = { xPct, key: PIANO_KEYS[rowIdx].key };
        }}
        onmousemove={(e) => {
          if (lassoStart && telemetry.pianoRollTool === 'lasso') {
            const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
            const xPct = Math.max(0, Math.min(100, ((e.clientX - rect.left - 46) / Math.max(1, rect.width - 46)) * 100));
            const rowIdx = Math.max(0, Math.min(PIANO_KEYS.length - 1, Math.floor(((e.clientY - rect.top) / Math.max(1, rect.height)) * PIANO_KEYS.length)));
            lassoCurrent = { xPct, key: PIANO_KEYS[rowIdx].key };
            return;
          }
          handlePianoRollMouseMove(e);
        }}
        onmouseup={() => {
          if (lassoStart && lassoCurrent && activePattern) {
            const minTick = viewStartTick + (Math.min(lassoStart.xPct, lassoCurrent.xPct) / 100) * TIMELINE_TICKS;
            const maxTick = viewStartTick + (Math.max(lassoStart.xPct, lassoCurrent.xPct) / 100) * TIMELINE_TICKS;
            const minKey = Math.min(lassoStart.key, lassoCurrent.key);
            const maxKey = Math.max(lassoStart.key, lassoCurrent.key);
            const picked = activePattern.notes
              .filter(
                (n) =>
                  n.key >= minKey &&
                  n.key <= maxKey &&
                  n.startTick + n.durationTicks >= minTick &&
                  n.startTick <= maxTick
              )
              .map((n) => n.id);
            telemetry.selectedNoteIds = picked;
            if (picked.length > 0) {
              telemetry.selectedNoteId = picked[0];
            }
          }
          lassoStart = null;
          lassoCurrent = null;
          stopNoteDrag();
        }}
        onmouseleave={() => {
          lassoStart = null;
          lassoCurrent = null;
          stopNoteDrag();
        }}
      >
        <div class="pr-view-nav">
          <button onclick={() => scrollView(-TIMELINE_TICKS / 2)} title="Cuộn trái">◀</button>
          <span>{viewStartTick}–{viewStartTick + TIMELINE_TICKS} tick · {visibleNotes.length}/{activePattern?.notes.length ?? 0} nốt hiển thị</span>
          <button onclick={() => scrollView(TIMELINE_TICKS / 2)} title="Cuộn phải">▶</button>
        </div>
        {#each PIANO_KEYS as row}
          <div class="pr-row">
            <div class="pr-key-label">{row.label}</div>
            <div class="pr-cells">
              {#each Array(16) as _, stepIdx}
                <!-- svelte-ignore a11y_click_events_have_key_events -->
                <div
                  class="pr-cell"
                  onclick={() => {
                    if (telemetry.pianoRollTool === 'draw') {
                      handleGridCellClick(row.key, stepIdx);
                    }
                  }}
                ></div>
              {/each}

              <!-- Ghost Notes from Inactive Patterns -->
              {#each visibleGhostsByKey.get(row.key) ?? [] as gNote ('ghost-' + gNote.id)}
                <div
                  class="pr-ghost-note"
                  style="
                    left: {((gNote.startTick - viewStartTick) / TIMELINE_TICKS) * 100}%;
                    width: {(gNote.durationTicks / TIMELINE_TICKS) * 100}%;
                  "
                ></div>
              {/each}

              {#if activePattern}
                {#each visibleNotesByKey.get(row.key) ?? [] as note (note.id)}
                  <div
                    class="pr-note {telemetry.selectedNoteId === note.id || telemetry.selectedNoteIds.includes(note.id) ? 'selected' : ''} {note.isSlide ? 'slide-note' : ''}"
                    style="
                      left: {((note.startTick - viewStartTick) / TIMELINE_TICKS) * 100}%;
                      width: {(note.durationTicks / TIMELINE_TICKS) * 100}%;
                      background: {COLOR_GROUP_PALETTE[note.colorGroup % COLOR_GROUP_PALETTE.length]};
                    "
                    onmousedown={(e) => {
                      telemetry.selectedNoteIds = [note.id];
                      startNoteDrag(e, note, 'move');
                    }}
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

      <!-- Bottom Per-Note Expression Lane (Velocity / Pan / Fine Pitch) -->
      {#if activePattern}
        <div class="expression-lane-box">
          <div class="expr-tabs">
            {#each [
              ['velocity', 'Velocity'],
              ['pan', 'Pan'],
              ['finePitch', 'Fine Pitch'],
            ] as [eKey, eLabel]}
              <button
                class={telemetry.expressionLaneTarget === eKey ? 'active' : ''}
                onclick={() =>
                  (telemetry.expressionLaneTarget = eKey as 'velocity' | 'pan' | 'finePitch')}
              >
                {eLabel}
              </button>
            {/each}
          </div>
          <div class="expr-bars-canvas">
            {#each visibleNotes as note (note.id)}
              {@const normVal =
                telemetry.expressionLaneTarget === 'velocity'
                  ? note.velocity
                  : telemetry.expressionLaneTarget === 'pan'
                    ? (note.pan + 1.0) * 0.5
                    : (note.finePitchCents + 200.0) / 400.0}
              <!-- svelte-ignore a11y_click_events_have_key_events -->
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <div
                class="expr-stalk {telemetry.selectedNoteId === note.id ? 'selected' : ''}"
                style="left: {((note.startTick - viewStartTick) / TIMELINE_TICKS) * 100}%; height: {Math.max(8, normVal * 100)}%;"
                title="Note #{note.id}: {telemetry.expressionLaneTarget} = {normVal.toFixed(2)}"
                onclick={() => (telemetry.selectedNoteId = note.id)}
              ></div>
            {/each}
          </div>
        </div>
      {/if}

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
  .render-pending { font-size: 0.72rem; color: var(--muted); align-self: center; }
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
  h2 { margin: 0; font-size: 1rem; color: var(--accent); font-weight: 600; }
  .subtitle { font-size: 0.75rem; color: var(--muted); }
  .toolbar { display: flex; align-items: center; gap: 8px; font-size: 0.76rem; color: var(--muted); }
  .toolbar select { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; margin-left: 4px; }
  .tool-btn { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; cursor: pointer; font-weight: 600; font-size: 0.75rem; }
  .tool-btn.active { background: var(--warn); color: #0b2b1e; border-color: var(--warn); }
  .sequencer-layout { display: grid; grid-template-columns: 1fr 1.25fr; gap: 12px; }
  .playlist-box, .pianoroll-box { background: var(--sub-surface-2); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px; display: flex; flex-direction: column; gap: 8px; }
  .sub-header { display: flex; justify-content: space-between; align-items: center; }
  h3 { margin: 0; font-size: 0.84rem; color: var(--text); font-weight: 600; }
  .fixed30-readout { font-size: 0.74rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; color: var(--muted); }
  .fixed30-readout strong { color: var(--accent); }
  .playlist-tracks { display: flex; flex-direction: column; gap: 5px; }
  .playlist-empty-state { display: flex; justify-content: space-between; align-items: center; gap: 8px; padding: 10px 12px; background: var(--sub-surface); border: 1px dashed var(--border); border-radius: 6px; font-size: 0.74rem; color: var(--muted); }
  .playlist-empty-actions { display: flex; gap: 6px; flex-wrap: wrap; }
  .playlist-row { display: grid; grid-template-columns: 150px 1fr; gap: 8px; align-items: center; font-size: 0.74rem; }
  .row-meta { display: flex; justify-content: space-between; align-items: center; background: var(--surface); border: 1px solid var(--border); padding: 4px 8px; border-radius: 6px; }
  .mute-btn { background: var(--ok); color: #0b2b1e; border: none; border-radius: 4px; padding: 2px 6px; font-size: 0.68rem; font-weight: 700; cursor: pointer; }
  .mute-btn.muted { background: var(--surface-2); color: var(--muted); border: 1px solid var(--border); }
  .row-timeline { position: relative; height: 24px; background: var(--sub-surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
  .clip-block { position: absolute; top: 2px; bottom: 2px; border-radius: 4px; padding: 2px 6px; font-size: 0.7rem; font-weight: 600; color: #0b2b1e; display: flex; align-items: center; }
  .clip-block.kind-pattern { background: var(--accent); }
  .clip-block.kind-audiostem { background: var(--warn); }
  .clip-block.kind-automation { background: var(--ok); }
  .clip-block.selected-clip { outline: 2px solid var(--text); z-index: 2; }
  .playlist-pro-tools { display: flex; gap: 4px; flex-wrap: wrap; }
  .mini-tool { background: var(--surface-2); color: var(--muted); border: 1px solid var(--border); border-radius: 5px; padding: 2px 7px; font-size: 0.69rem; font-weight: 600; cursor: pointer; }
  .mini-tool.active { background: var(--warn); color: #0b2b1e; border-color: var(--warn); }
  .take-pill { background: var(--warn); color: #0b2b1e; border-radius: 4px; padding: 1px 4px; font-size: 0.62rem; margin-left: 4px; }
  .clip-inspector-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 5px 8px; font-size: 0.69rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  .pr-ghost-note { position: absolute; top: 4px; bottom: 4px; border-radius: 3px; background: rgba(148, 163, 184, 0.22); border: 1px dashed rgba(148, 163, 184, 0.45); pointer-events: none; }
  .expression-lane-box { display: grid; grid-template-columns: 84px 1fr; gap: 6px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 5px 8px; height: 52px; align-items: stretch; }
  .expr-tabs { display: flex; flex-direction: column; gap: 2px; justify-content: center; }
  .expr-tabs button { background: var(--surface-2); color: var(--muted); border: 1px solid var(--border); border-radius: 4px; padding: 2px 5px; font-size: 0.64rem; cursor: pointer; text-align: left; }
  .expr-tabs button.active { background: var(--accent); color: #0b2b1e; border-color: var(--accent); font-weight: 700; }
  .expr-bars-canvas { position: relative; background: var(--sub-surface); border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }
  .expr-stalk { position: absolute; bottom: 0; width: 6px; background: var(--accent); border-radius: 2px 2px 0 0; cursor: pointer; opacity: 0.82; }
  .expr-stalk.selected { background: var(--warn); opacity: 1; width: 8px; }
  .spline-canvas-wrap { display: flex; flex-direction: column; gap: 6px; }
  .spline-svg { width: 100%; height: 92px; background: var(--sub-surface); border: 1px solid var(--border); border-radius: 6px; cursor: crosshair; }
  .spline-points-bar { display: flex; flex-wrap: wrap; gap: 6px; }
  .spline-pt-chip { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; font-size: 0.7rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; display: flex; align-items: center; gap: 6px; }
  .del-pt-btn { background: var(--err); color: #0b2b1e; font-weight: 700; border: none; border-radius: 4px; cursor: pointer; padding: 0 6px; }
  .pattern-tabs { display: flex; gap: 4px; }
  .pattern-tabs button { background: var(--surface-2); color: var(--muted); border: 1px solid var(--border); border-radius: 6px; padding: 3px 8px; font-size: 0.72rem; cursor: pointer; }
  .pattern-tabs button.active { background: var(--accent); color: #0b2b1e; border-color: var(--accent); font-weight: 600; }
  .pianoroll-grid { display: flex; flex-direction: column; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
  .pr-row { display: grid; grid-template-columns: 46px 1fr; height: 22px; border-bottom: 1px solid rgba(118, 210, 219, 0.08); }
  .pr-key-label { background: var(--surface); color: var(--muted); font-size: 0.68rem; font-family: var(--font-mono); display: flex; align-items: center; justify-content: center; border-right: 1px solid var(--border); }
  .pr-cells { position: relative; display: grid; grid-template-columns: repeat(16, 1fr); background: var(--sub-surface); }
  .pr-cell { border-right: 1px solid rgba(118, 210, 219, 0.06); cursor: pointer; }
  .pr-note { position: absolute; top: 2px; bottom: 2px; border-radius: 4px; color: #0b2b1e; font-size: 0.66rem; font-weight: 700; display: flex; align-items: center; justify-content: space-between; padding: 0 4px; cursor: grab; box-sizing: border-box; }
  .pr-note.selected { outline: 2px solid var(--text); }
  .pr-note.slide-note { border-left: 4px solid #0b2b1e; }
  .resize-handle { width: 5px; height: 100%; cursor: ew-resize; }
  .note-inspector { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; font-size: 0.72rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  .delete-note-btn { background: var(--err); color: #0b2b1e; border: none; border-radius: 6px; padding: 4px 10px; font-size: 0.72rem; cursor: pointer; font-weight: 700; }
  .pr-view-nav { display: flex; align-items: center; gap: 8px; font-size: 0.7rem; color: var(--muted); margin-bottom: 4px; }
  .pr-view-nav button { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 1px 8px; cursor: pointer; }
</style>

