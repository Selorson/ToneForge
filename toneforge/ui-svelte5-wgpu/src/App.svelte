<script lang="ts">
  import { DawTelemetryView, VISUAL_TELEMETRY_FRAME_BYTES } from './lib/state/daw_master_store.svelte';
  import MixerRack502 from './lib/components/MixerRack502.svelte';
  import PatcherNodeCanvas from './lib/components/PatcherNodeCanvas.svelte';
  import ProQ4InteractiveAnalyzer from './lib/components/ProQ4InteractiveAnalyzer.svelte';
  import Soothe3ResonanceView from './lib/components/Soothe3ResonanceView.svelte';
  import Ozone12MasterChain from './lib/components/Ozone12MasterChain.svelte';
  import Uvr5EnsembleStudio from './lib/components/Uvr5EnsembleStudio.svelte';
  import PlaylistPianoRollEditor from './lib/components/PlaylistPianoRollEditor.svelte';
  import OfflineRenderStudio from './lib/components/OfflineRenderStudio.svelte';

  const sharedBuffer = new ArrayBuffer(VISUAL_TELEMETRY_FRAME_BYTES);
  const telemetry = new DawTelemetryView(sharedBuffer);

  let activeTab = $state<'all' | 'playlist' | 'mixer' | 'proq4' | 'soothe3' | 'ozone12' | 'uvr5' | 'render'>('all');

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
  <header class="transport-bar">
    <div class="brand">
      <h1>ToneForge DAW</h1>
      <span class="badge">120 FPS WebGPU · 128 KB Zero-Copy SHM</span>
    </div>

    <div class="transport-buttons">
      <button
        class={telemetry.isPlaying ? 'playing' : ''}
        onclick={() => telemetry.setTransport(!telemetry.isPlaying, telemetry.tempoBpm)}
      >
        {telemetry.isPlaying ? '⏸ Pause' : '▶ Play'}
      </button>
      <button onclick={() => telemetry.seekTransport(0, 0)}>⏮ Reset</button>
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
        ['mixer', '502-Track Mixer & Patcher'],
        ['proq4', 'Pro-Q 4 (24-Band)'],
        ['soothe3', 'Soothe3 Spectral'],
        ['ozone12', 'Ozone 12 Console'],
        ['uvr5', 'UVR5 Stem Studio'],
        ['render', 'Offline Render & Bounce'],
      ] as [tabKey, label]}
        <button
          class={activeTab === tabKey ? 'active' : ''}
          onclick={() => (activeTab = tabKey as any)}
        >
          {label}
        </button>
      {/each}
    </nav>
  </header>

  <div class="workspace-grid">
    {#if activeTab === 'all' || activeTab === 'playlist'}
      <PlaylistPianoRollEditor {telemetry} />
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
</style>

