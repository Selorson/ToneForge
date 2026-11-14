<script lang="ts">
  import { DawTelemetryView, VISUAL_TELEMETRY_FRAME_BYTES } from './lib/state/daw_master_store.svelte';
  import type { WorkspaceTabKey } from './lib/state/guides';
  import Guide from './lib/components/Guide.svelte';
  import HandbookView from './lib/components/HandbookView.svelte';
  import MixerRack502 from './lib/components/MixerRack502.svelte';
  import PatcherNodeCanvas from './lib/components/PatcherNodeCanvas.svelte';
  import ProQ4InteractiveAnalyzer from './lib/components/ProQ4InteractiveAnalyzer.svelte';
  import Soothe3ResonanceView from './lib/components/Soothe3ResonanceView.svelte';
  import Ozone12MasterChain from './lib/components/Ozone12MasterChain.svelte';
  import Uvr5EnsembleStudio from './lib/components/Uvr5EnsembleStudio.svelte';
  import PlaylistPianoRollEditor from './lib/components/PlaylistPianoRollEditor.svelte';
  import OfflineRenderStudio from './lib/components/OfflineRenderStudio.svelte';
  import VocalPitchEditor from './lib/components/VocalPitchEditor.svelte';
  import Vst3ClapHostBrowser from './lib/components/Vst3ClapHostBrowser.svelte';

  const sharedBuffer = new ArrayBuffer(VISUAL_TELEMETRY_FRAME_BYTES);
  const telemetry = new DawTelemetryView(sharedBuffer);

  const initialParams =
    typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null;
  const initialTabParam = initialParams?.get('tab') as WorkspaceTabKey | null;
  const openGuideByDefault = initialParams?.get('guide') === 'open';
  const startPausedByParam = initialParams?.get('paused') === '1';

  if (startPausedByParam) {
    telemetry.setTransport(false, telemetry.tempoBpm);
  }

  let activeTab = $state<WorkspaceTabKey>(initialTabParam ?? 'all');
  let handbookSectionId = $state(initialParams?.get('sec') ?? '');
  let fileInputEl = $state<HTMLInputElement | null>(null);

  async function handleUserFileSelected(e: Event): Promise<void> {
    const input = e.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    await telemetry.loadUserAudioFile(file);
    input.value = '';
  }

  async function handleOpenAudioClick(): Promise<void> {
    if (telemetry.isDesktopTauri) {
      await telemetry.openNativeAudioFilePicker();
    } else {
      fileInputEl?.click();
    }
  }

  $effect(() => {
    telemetry.refreshGpuAcceleratorStatus();
    telemetry.refreshWasapiDevicesAndStatus();
    telemetry.refreshAudioMidiInputs();
    if (telemetry.isDesktopTauri) {
      // Desktop: the engine starts stopped with an empty session; mirror its project state.
      telemetry.isPlaying = false;
      telemetry.hydrateFromEngine();
    } else {
      telemetry.refreshUndoRedoStatus();
    }
  });

  $effect(() => {
    const onGlobalKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) {
          telemetry.redoProjectAction();
        } else {
          telemetry.undoProjectAction();
        }
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        telemetry.redoProjectAction();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        telemetry.saveProject(e.shiftKey);
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'o') {
        e.preventDefault();
        telemetry.openProject();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'n') {
        e.preventDefault();
        telemetry.newProject();
      }
    };
    window.addEventListener('keydown', onGlobalKeyDown);
    return () => window.removeEventListener('keydown', onGlobalKeyDown);
  });

  $effect(() => {
    if (telemetry.isDesktopTauri) return;
    const unlockSpeakerAudio = () => {
      telemetry.syncDspToSpeakerEngine();
      if (telemetry.isPlaying) {
        telemetry.speakerEngine.setPlaying(true, telemetry);
      }
    };
    window.addEventListener('pointerdown', unlockSpeakerAudio, { once: true });
    return () => window.removeEventListener('pointerdown', unlockSpeakerAudio);
  });

  $effect(() => {
    const onNavigate = (e: Event) => {
      const custom = e as CustomEvent<{ tab: WorkspaceTabKey; sectionId?: string }>;
      if (custom.detail?.tab) {
        activeTab = custom.detail.tab;
        handbookSectionId = custom.detail.sectionId ?? '';
        if (!custom.detail.sectionId) {
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }
      }
    };
    window.addEventListener('toneforge:navigate', onNavigate);
    return () => window.removeEventListener('toneforge:navigate', onNavigate);
  });

  $effect(() => {
    let rafId = 0;
    let active = true;

    const tick120Hz = (timestampMs: number) => {
      if (!active) return;
      telemetry.pollTelemetry120Hz(timestampMs);
      rafId = requestAnimationFrame(tick120Hz);
    };

    rafId = requestAnimationFrame(tick120Hz);
    return () => {
      active = false;
      cancelAnimationFrame(rafId);
    };
  });
</script>

<main class="toneforge-shell">
  {#if telemetry.lastEngineError}
    <div class="engine-error-banner" role="alert">
      <span>⚠ Lỗi engine: {telemetry.lastEngineError}</span>
      <button onclick={() => telemetry.clearEngineError()} title="Ẩn thông báo">✕</button>
    </div>
  {/if}
  <header class="transport-bar">
    <div class="brand">
      <img class="brand-logo" src="/favicon.png" alt="ToneForge" width="34" height="34" />
      <h1>ToneForge DAW</h1>
      <span class="badge">120 FPS WebGPU · 128 KB Zero-Copy SHM</span>
      <span class="badge" title={telemetry.gpuAcceleratorDetail}>{telemetry.gpuAcceleratorBadge}</span>
    </div>

    <div class="transport-buttons">
      <button
        class={telemetry.isPlaying ? 'playing' : 'paused-btn'}
        onclick={() => telemetry.setTransport(!telemetry.isPlaying, telemetry.tempoBpm)}
        title={telemetry.isPlaying
          ? 'Bấm để Tạm dừng nhạc & tắt đường sóng (chỉ hiện đường EQ vàng)'
          : 'Bấm để Phát nhạc ra Loa máy tính & bật đường sóng thời gian thực'}
      >
        {telemetry.isPlaying ? '⏸ Pause' : '▶ Play'}
      </button>
      <button onclick={() => telemetry.seekTransport(0, 0)}>⏮ Reset</button>

      {#if telemetry.isDesktopTauri}
        <button
          onclick={() => telemetry.newProject()}
          title="Tạo dự án trắng mới — Ctrl+N"
        >
          📄 Dự án mới
        </button>
        <button
          onclick={() => telemetry.openProject()}
          title="Mở dự án ToneForge (.tfproj) — Ctrl+O"
        >
          📂 Mở dự án
        </button>
        <button
          onclick={() => telemetry.saveProject(false)}
          title={telemetry.projectFilePath
            ? `Lưu dự án vào ${telemetry.projectFilePath} — Ctrl+S (Ctrl+Shift+S: Lưu thành)`
            : 'Lưu dự án (.tfproj) — Ctrl+S'}
        >
          💾 Lưu dự án
        </button>
      {/if}

      <button
        class="undo-btn"
        disabled={!telemetry.undoAvailable && telemetry.undoCount === 0}
        onclick={() => telemetry.undoProjectAction()}
        title={telemetry.nextUndoLabel
          ? `Hoàn tác: ${telemetry.nextUndoLabel} (Ctrl+Z)`
          : 'Hoàn tác thao tác vừa thực hiện (Ctrl+Z)'}
      >
        ↶ Undo ({telemetry.undoCount})
      </button>
      <button
        class="undo-btn"
        disabled={!telemetry.redoAvailable && telemetry.redoCount === 0}
        onclick={() => telemetry.redoProjectAction()}
        title={telemetry.nextRedoLabel
          ? `Làm lại: ${telemetry.nextRedoLabel} (Ctrl+Y / Ctrl+Shift+Z)`
          : 'Làm lại thao tác vừa hoàn tác (Ctrl+Y / Ctrl+Shift+Z)'}
      >
        ↷ Redo ({telemetry.redoCount})
      </button>

      <button
        class={telemetry.isDiskRecording ? 'rec-btn active' : 'rec-btn'}
        onclick={() => telemetry.toggleDiskRecording(384)}
        title="Thu âm trực tiếp Microphone / Nhạc cụ vào Track đang chọn với Tự động bù độ trễ phần cứng & Loop Take Comping"
      >
        {telemetry.isDiskRecording
          ? `⏹ Stop REC (T#${telemetry.armedTrackIndex})`
          : `● REC (T#${telemetry.armedTrackIndex}${telemetry.completedTakes.length > 0 ? ` · ${telemetry.completedTakes.length} takes` : ''})`}
      </button>

      <div class="hw-rec-box" title="Chế độ Driver (WASAPI / Steinberg ASIO), Ngõ vào thu âm Microphone & Bàn phím MIDI rời">
        <select
          class="wasapi-select"
          value={telemetry.selectedDriverMode}
          onchange={(e) =>
            telemetry.configureHardwareRecording(
              (e.currentTarget as HTMLSelectElement).value as 'WASAPI_Shared' | 'WASAPI_Exclusive' | 'ASIO',
              telemetry.selectedWasapiInputId,
              telemetry.selectedMidiDeviceId
            )}
          aria-label="Audio Driver Mode"
        >
          <option value="WASAPI_Shared">🔊 WASAPI Shared</option>
          <option value="WASAPI_Exclusive">⚡ WASAPI Exclusive</option>
          <option value="ASIO">🎛️ Steinberg ASIO</option>
        </select>

        <select
          class="wasapi-select"
          value={telemetry.selectedWasapiInputId}
          onchange={(e) =>
            telemetry.configureHardwareRecording(
              telemetry.selectedDriverMode,
              (e.currentTarget as HTMLSelectElement).value,
              telemetry.selectedMidiDeviceId
            )}
          aria-label="Microphone Audio Input"
        >
          <option value="none">🎤 Mic In: Off</option>
          <option value="">🎤 Mic In: Default Windows</option>
          {#each telemetry.wasapiInputs as mic (mic.id)}
            <option value={mic.id}>🎤 {mic.friendlyName}</option>
          {/each}
        </select>

        <select
          class="wasapi-select"
          value={String(telemetry.selectedMidiDeviceId)}
          onchange={(e) =>
            telemetry.configureHardwareRecording(
              telemetry.selectedDriverMode,
              telemetry.selectedWasapiInputId,
              parseInt((e.currentTarget as HTMLSelectElement).value, 10)
            )}
          aria-label="Hardware MIDI Input"
        >
          <option value="-1">🎹 MIDI In: Off</option>
          {#each telemetry.midiInputs as md (md.device_id)}
            <option value={String(md.device_id)}>🎹 {md.name}</option>
          {/each}
        </select>
      </div>

      <input
        bind:this={fileInputEl}
        type="file"
        accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.aiff"
        style="display: none;"
        onchange={handleUserFileSelected}
      />
      <button
        class="open-audio-btn"
        onclick={handleOpenAudioClick}
        title="Mở bài hát (.wav, .mp3, .flac, .ogg) từ đường dẫn gốc ổ cứng để nghe qua Loa, tinh chỉnh EQ/Soothe3/Ozone & tách Stem trực tiếp"
      >
        📂 Mở nhạc từ máy (.wav/.mp3/.flac)
      </button>

      {#if telemetry.loadedAudioFileName}
        <span
          class="loaded-audio-badge"
          title={telemetry.loadedAudioFullPath ?? telemetry.loadedAudioFileName}
        >
          🎵 Đang nạp: {telemetry.loadedAudioFileName}
        </span>
      {/if}

      {#if telemetry.audioSourceMode === 'customFile' || telemetry.audioSourceMode === 'stems4'}
        <button
          class="switch-source-btn"
          onclick={() => telemetry.switchToPianoRollSource()}
          title="Đóng bài hát / Stem đang nạp trên Track 1..4 để nghe riêng Piano Roll & Playlist (giữ nguyên toàn bộ chuỗi FX Mixer)"
        >
          ✕ Đóng file nhạc (Về Piano Roll / Playlist)
        </button>
      {/if}

      <div
        class="speaker-box"
        title="Âm lượng đầu ra phần cứng Windows WASAPI (100% Rust Lock-Free DSP/AI Output)"
      >
        <button class="mute-btn" onclick={() => telemetry.toggleSpeakerMute()}>
          {telemetry.speakerMuted || telemetry.speakerVolumePct === 0
            ? '🔇 Tắt loa'
            : `🔊 Loa: ${telemetry.speakerVolumePct}%`}
        </button>
        <input
          type="range"
          min="0"
          max="100"
          step="1"
          value={telemetry.speakerVolumePct}
          oninput={(e) =>
            telemetry.setSpeakerVolume(parseFloat((e.currentTarget as HTMLInputElement).value))}
        />
      </div>

      <div class="wasapi-box" title={telemetry.wasapiStatusMessage}>
        <select
          class="wasapi-select"
          value={telemetry.selectedWasapiDeviceId}
          onchange={(e) =>
            telemetry.configureWasapiOutput(
              (e.currentTarget as HTMLSelectElement).value,
              telemetry.wasapiExclusiveMode
            )}
          aria-label="WASAPI Output Device Selector"
        >
          <option value="">
            🎧 Mặc định Windows ({telemetry.wasapiActiveDeviceName})
          </option>
          {#each telemetry.wasapiDevices as dev (dev.id)}
            <option value={dev.id}>
              {dev.isDefault ? '★ ' : ''}{dev.friendlyName} ({(dev.mixSampleRateHz / 1000).toFixed(1)}kHz)
            </option>
          {/each}
        </select>

        <button
          class={telemetry.wasapiActiveExclusive ? 'wasapi-excl-btn active' : 'wasapi-excl-btn'}
          onclick={() =>
            telemetry.configureWasapiOutput(
              telemetry.selectedWasapiDeviceId,
              !telemetry.wasapiExclusiveMode
            )}
          title="Chuyển đổi giữa WASAPI Shared Event-Driven (mặc định, dùng chung âm thanh hệ thống) và WASAPI Exclusive Low-Latency (độc quyền phần cứng độ trễ cực thấp)"
        >
          {telemetry.wasapiActiveExclusive
            ? '⚡ WASAPI Exclusive'
            : telemetry.wasapiExclusiveMode
              ? '⚡ Excl→Shared'
              : '🔊 WASAPI Shared'}
        </button>

        <button
          class="wasapi-refresh-btn"
          onclick={() => telemetry.refreshWasapiDevicesAndStatus()}
          title="Làm mới danh sách card âm thanh / tai nghe / DAC WASAPI"
        >
          ↻
        </button>
      </div>

      <label class="bpm-box">
        BPM
        <input
          type="number"
          min="10"
          max="999"
          step="0.5"
          value={telemetry.tempoBpm}
          onchange={(e) =>
            telemetry.setTransport(
              telemetry.isPlaying,
              parseFloat((e.currentTarget as HTMLInputElement).value)
            )}
        />
      </label>
    </div>

    <div class="transport-metrics">
      <span title={telemetry.wasapiStatusMessage}>
        WASAPI: <strong>{(telemetry.wasapiHardwareSampleRateHz / 1000).toFixed(1)}kHz / {telemetry.wasapiBufferFrames}f ({telemetry.wasapiSampleFormat})</strong>
      </span>
      <span>Xruns: <strong>{telemetry.xrunCount}</strong></span>
      <span>Seq: <strong>#{telemetry.frameSeq}</strong></span>
      <span>Sample: <strong>{telemetry.samplePos}</strong></span>
      <span>PPQ ({telemetry.ppqResolution}): <strong>{telemetry.playheadPpq}</strong></span>
      <span>CPU: <strong>{telemetry.cpuLoadPercent.toFixed(1)}%</strong></span>
      <span>LUFS-I: <strong>{telemetry.lufsIntegrated.toFixed(1)}</strong></span>
      <span>TP: <strong>{telemetry.truePeakDbL.toFixed(2)} dBTP</strong></span>
    </div>

    <nav class="workspace-tabs">
      {#each [
        ['all', 'Unified Master View'],
        ['playlist', 'Playlist & Piano Roll'],
        ['vocal', '🎤 Vocal Pitch (NewTone)'],
        ['mixer', '502-Track Mixer & Patcher'],
        ['plugins', '🔌 VST3 / CLAP Host'],
        ['proq4', 'Pro-Q 4 (24-Band)'],
        ['soothe3', 'Soothe3 Spectral'],
        ['ozone12', 'Ozone 12 Console'],
        ['uvr5', 'UVR5 Stem Studio'],
        ['render', 'Offline Render & Bounce'],
        ['handbook', '📖 Cẩm nang (Hướng dẫn người mới)'],
      ] as [tabKey, label]}
        <button
          class="{activeTab === tabKey ? 'active' : ''} {tabKey === 'handbook' ? 'handbook-tab' : ''}"
          onclick={() => {
            activeTab = tabKey as WorkspaceTabKey;
            if (tabKey !== 'handbook') handbookSectionId = '';
          }}
        >
          {label}
        </button>
      {/each}
    </nav>
  </header>

  <div class="workspace-grid">
    {#if activeTab === 'all'}
      <Guide id="unified" openByDefault={openGuideByDefault} />
    {/if}

    {#if activeTab === 'handbook'}
      <HandbookView initialSectionId={handbookSectionId} />
    {/if}

    {#if activeTab === 'all' || activeTab === 'playlist'}
      <PlaylistPianoRollEditor {telemetry} />
    {/if}

    {#if activeTab === 'all' || activeTab === 'vocal'}
      <VocalPitchEditor {telemetry} />
    {/if}

    {#if activeTab === 'all' || activeTab === 'proq4'}
      <ProQ4InteractiveAnalyzer {telemetry} />
    {/if}

    {#if activeTab === 'all' || activeTab === 'soothe3'}
      <Soothe3ResonanceView {telemetry} />
    {/if}

    {#if activeTab === 'all' || activeTab === 'ozone12'}
      <Ozone12MasterChain {telemetry} />
    {/if}

    {#if activeTab === 'all' || activeTab === 'mixer'}
      <MixerRack502 {telemetry} />
      <PatcherNodeCanvas {telemetry} />
    {/if}

    {#if activeTab === 'all' || activeTab === 'plugins'}
      <Vst3ClapHostBrowser {telemetry} />
    {/if}

    {#if activeTab === 'all' || activeTab === 'uvr5'}
      <Uvr5EnsembleStudio {telemetry} />
    {/if}

    {#if activeTab === 'all' || activeTab === 'render'}
      <OfflineRenderStudio {telemetry} />
    {/if}
  </div>
</main>

<style>
  .toneforge-shell {
    min-height: 100vh;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    padding: 12px 16px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .transport-bar {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    background: var(--sidebar-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .brand-logo {
    width: 34px;
    height: 34px;
    border-radius: 7px;
    flex-shrink: 0;
  }
  h1 {
    margin: 0;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
  }
  .badge {
    background: var(--surface-2);
    color: var(--accent);
    border: 1px solid var(--border);
    font-size: 0.72rem;
    padding: 3px 9px;
    border-radius: 999px;
    font-weight: 600;
  }
  .transport-buttons {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }
  .transport-buttons button {
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 12px;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.78rem;
    transition: all 0.15s ease;
  }
  .transport-buttons button:hover {
    border-color: var(--accent);
  }
  .transport-buttons button.playing {
    background: var(--ok);
    color: #0b2b1e;
    border-color: var(--ok);
    font-weight: 700;
  }
  .transport-buttons button.paused-btn {
    background: rgba(235, 198, 98, 0.18);
    color: var(--warn);
    border-color: var(--warn);
    font-weight: 700;
  }
  .transport-buttons button.open-audio-btn {
    background: rgba(118, 210, 219, 0.16);
    color: var(--accent);
    border-color: var(--accent);
    font-weight: 700;
  }
  .transport-buttons button.switch-source-btn {
    background: rgba(235, 198, 98, 0.14);
    color: var(--warn);
    border-color: var(--warn);
  }
  .transport-buttons button.undo-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .transport-buttons button.rec-btn {
    background: rgba(239, 68, 68, 0.14);
    color: #fca5a5;
    border-color: #ef4444;
    font-weight: 700;
  }
  .transport-buttons button.rec-btn.active {
    background: #ef4444;
    color: #ffffff;
    border-color: #ef4444;
  }
  .hw-rec-box {
    display: flex;
    align-items: center;
    gap: 5px;
    background: var(--sub-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 6px;
  }
  .loaded-audio-badge {
    background: var(--sub-surface);
    color: var(--ok);
    border: 1px solid var(--ok);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 0.74rem;
    font-weight: 600;
    max-width: 220px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .speaker-box {
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--sub-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 8px;
  }
  .speaker-box .mute-btn {
    padding: 3px 8px;
    font-size: 0.74rem;
    background: transparent;
    border: none;
  }
  .speaker-box input[type='range'] {
    width: 72px;
    accent-color: var(--accent);
    cursor: pointer;
  }
  .wasapi-box {
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--sub-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 6px;
  }
  .wasapi-select {
    max-width: 230px;
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 4px 8px;
    font-size: 0.74rem;
    font-weight: 600;
    cursor: pointer;
  }
  .transport-buttons button.wasapi-excl-btn {
    padding: 4px 9px;
    font-size: 0.73rem;
  }
  .transport-buttons button.wasapi-excl-btn.active {
    background: rgba(118, 210, 219, 0.22);
    color: var(--accent);
    border-color: var(--accent);
    font-weight: 700;
  }
  .transport-buttons button.wasapi-refresh-btn {
    padding: 4px 8px;
    font-size: 0.76rem;
  }
  .bpm-box {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    color: var(--muted);
    font-weight: 500;
  }
  .bpm-box input {
    width: 68px;
    background: var(--sub-surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 8px;
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
  }
  .transport-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    font-size: 0.76rem;
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    color: var(--muted);
    background: var(--sub-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
  }
  .transport-metrics strong {
    color: var(--accent);
    font-weight: 600;
  }
  .workspace-tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    width: 100%;
    padding-top: 6px;
    border-top: 1px solid var(--border);
  }
  .workspace-tabs button {
    background: var(--surface-2);
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.76rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .workspace-tabs button:hover {
    color: var(--text);
    border-color: var(--accent);
  }
  .workspace-tabs button.handbook-tab {
    border-color: var(--warn);
    color: var(--warn);
    font-weight: 600;
  }
  .workspace-tabs button.active {
    background: var(--accent);
    color: #0b2b1e;
    border-color: var(--accent);
    font-weight: 600;
  }
  .workspace-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .engine-error-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 6px 12px;
    background: #4a1717;
    color: #ffd7d7;
    border-bottom: 1px solid #a33;
    font-size: 0.8rem;
  }
  .engine-error-banner button {
    background: transparent;
    border: none;
    color: inherit;
    cursor: pointer;
  }
</style>

