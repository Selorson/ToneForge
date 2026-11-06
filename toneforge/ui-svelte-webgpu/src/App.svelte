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
    background: #090d16;
    color: #e2e8f0;
    padding: 12px;
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
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 10px 14px;
  }
  .brand { display: flex; align-items: center; gap: 10px; }
  h1 { margin: 0; font-size: 1.15rem; color: #f8fafc; }
  .badge { background: #1e293b; color: #38bdf8; font-size: 0.72rem; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
  .transport-buttons { display: flex; align-items: center; gap: 8px; }
  .transport-buttons button {
    background: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 4px; padding: 6px 12px; cursor: pointer; font-weight: 600;
  }
  .transport-buttons button.playing { background: #10b981; color: #090d16; border-color: #10b981; }
  .bpm-box input { width: 64px; background: #090d16; color: #f8fafc; border: 1px solid #334155; border-radius: 4px; padding: 4px; }
  .transport-metrics { display: flex; gap: 12px; font-size: 0.78rem; font-family: monospace; color: #94a3b8; }
  .transport-metrics strong { color: #f8fafc; }
  .workspace-tabs { display: flex; gap: 4px; }
  .workspace-tabs button {
    background: #131c31; color: #94a3b8; border: 1px solid #1e293b; border-radius: 4px; padding: 5px 10px; font-size: 0.75rem; cursor: pointer;
  }
  .workspace-tabs button.active { background: #38bdf8; color: #090d16; font-weight: 700; }
  .workspace-grid { display: flex; flex-direction: column; gap: 12px; }
</style>
