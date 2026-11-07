<script lang="ts">
  import { DawTelemetryView } from '../state/daw_master_store.svelte';
  import Guide from './Guide.svelte';

  let { telemetry }: { telemetry: DawTelemetryView } = $props();

  const ENSEMBLE_MODES = ['MAX_SPEC', 'MIN_SPEC', 'AVERAGE', 'MEDIAN', 'WEIGHTED'] as const;
  const UVR5_PROFILES = [
    'Ensemble: MDX23C + HTDemucs4 + VR-ASPP (3-Model)',
    'MDX23C-8KFFT-InstVoc_HQ (n_fft=8192, dim_f=4096)',
    'Kim_Vocal_2 (MDX-Net Lead Vocal, n_fft=7680)',
    'UVR-MDX-Net-Inst_HQ_3 (n_fft=6144, dim_f=3072)',
    'UVR-MDX-NET Karaoke / Vocals',
    'HTDemucs_ft 4-Stem (Vocals / Drums / Bass / Other)',
    'VR-Arch 4band_v3_sn (Cascaded ASPP ResNet)',
  ];

  let uvr5FileInputEl = $state<HTMLInputElement | null>(null);

  async function handleUvr5FileSelected(e: Event): Promise<void> {
    const input = e.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    await telemetry.loadUserAudioFile(file);
    input.value = '';
  }

  async function handleOpenUvr5AudioClick(): Promise<void> {
    if (telemetry.isDesktopTauri) {
      await telemetry.openNativeAudioFilePicker();
    } else {
      uvr5FileInputEl?.click();
    }
  }
</script>

<section class="panel uvr5-panel">
  <Guide id="uvr5" />
  <header class="panel-header">
    <div>
      <h2>UVR5 Multi-Architecture Offline Stem Separation & Spectral Ensemble Studio</h2>
      <span class="subtitle">
        {telemetry.uvr5ModelSpecSummary} · SpectralEnsembler
      </span>
    </div>
    <div class="uvr5-header-actions">
      <span class="gpu-badge" title={telemetry.gpuAcceleratorDetail}>
        {telemetry.gpuAcceleratorBadge}
      </span>
      <input
        bind:this={uvr5FileInputEl}
        type="file"
        accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.aiff"
        style="display: none;"
        onchange={handleUvr5FileSelected}
      />
      <button
        class="pick-file-btn"
        onclick={handleOpenUvr5AudioClick}
      >
        📂 Mở nhạc từ máy (.wav/.mp3/.flac)
      </button>
      <label class="profile-select">
        UVR5 Model Profile (143 Real Configs):
        <select
          value={telemetry.uvr5SelectedModelProfile}
          onchange={(e) =>
            telemetry.loadUvr5ModelProfile((e.currentTarget as HTMLSelectElement).value)}
        >
          {#each UVR5_PROFILES as prof}
            <option value={prof}>{prof}</option>
          {/each}
        </select>
      </label>
      <button
        class="run-btn"
        disabled={telemetry.uvr5IsProcessing}
        onclick={() => telemetry.runUvr5SeparationJob()}
      >
        {telemetry.uvr5IsProcessing ? 'Separating Stems...' : 'Run Ensemble & Bounce to Mixer 1..4'}
      </button>
    </div>
  </header>

  <div class="uvr5-grid">
    <div class="uvr5-card">
      <h3>Source Audio & Spectral Ensemble Mode</h3>
      <label>
        Input File:
        <div class="path-row">
          <input type="text" bind:value={telemetry.uvr5AudioPath} />
          <button type="button" class="folder-btn" onclick={handleOpenUvr5AudioClick}>
            📂 Chọn file
          </button>
        </div>
      </label>
      <label>
        Output Folder (Thư mục xuất 4 Stem .wav):
        <div class="path-row">
          <input
            type="text"
            placeholder="Mặc định: xuất ngay tại thư mục bài hát gốc"
            bind:value={telemetry.uvr5OutputDir}
          />
          <button
            type="button"
            class="folder-btn"
            onclick={() => telemetry.pickUvr5OutputDirectory()}
          >
            📁 Chọn thư mục
          </button>
        </div>
      </label>
      <label>
        Ensemble Algorithm:
        <select bind:value={telemetry.uvr5EnsembleMode}>
          {#each ENSEMBLE_MODES as mode}
            <option value={mode}>{mode}</option>
          {/each}
        </select>
      </label>
      <label>
        <input type="checkbox" bind:checked={telemetry.uvr5EnableDenoise2Pass} />
        MDX23C 2-Pass Phase-Inverted Bias Denoising
      </label>
      <label>
        <input type="checkbox" bind:checked={telemetry.uvr5InvertInstrumental} />
        Phase-Inverted Residual Subtraction (invert_audio_waveform)
      </label>
      <label>
        <input type="checkbox" bind:checked={telemetry.uvr5BounceToMixer} />
        Auto-Bounce 4 Stems onto Mixer Insert Tracks 1..4
      </label>
    </div>

    <div class="uvr5-card">
      <h3>Multi-Model Weights & VR ASPP Aggressiveness</h3>
      <label>
        MDX23C (TFC-TDF v3) Weight: {telemetry.uvr5ModelWeights[0].toFixed(2)}
        <input type="range" min="0" max="1" step="0.05" bind:value={telemetry.uvr5ModelWeights[0]} />
      </label>
      <label>
        HTDemucs v4 Weight: {telemetry.uvr5ModelWeights[1].toFixed(2)}
        <input type="range" min="0" max="1" step="0.05" bind:value={telemetry.uvr5ModelWeights[1]} />
      </label>
      <label>
        VR CascadedASPPNet Weight: {telemetry.uvr5ModelWeights[2].toFixed(2)}
        <input type="range" min="0" max="1" step="0.05" bind:value={telemetry.uvr5ModelWeights[2]} />
      </label>
      <label>
        Inter-Model Circular Phase Coherence Denoise: {telemetry.uvr5PhaseDenoiseStrength.toFixed(2)}
        <input type="range" min="0" max="1" step="0.05" bind:value={telemetry.uvr5PhaseDenoiseStrength} />
      </label>
    </div>

    <div class="uvr5-card">
      <h3>Separated Stems Output (Bounced to Tracks 1..4)</h3>
      {#if telemetry.uvr5SeparatedStems.length === 0}
        <p class="empty-stems-hint">
          Đã nạp file gốc <code>{telemetry.uvr5AudioPath}</code>. Bấm <strong>Run Ensemble &amp; Bounce to Mixer 1..4</strong> để tách 4 file stem <code>.wav</code> thật ngay tại thư mục đích.
        </p>
      {:else}
        <ul class="stem-list">
          {#each telemetry.uvr5SeparatedStems as stemPath, idx}
            <li>
              <strong>Insert Track {idx + 1}:</strong> <code>{stemPath}</code>
            </li>
          {/each}
        </ul>
      {/if}
    </div>
  </div>
</section>

<style>
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; gap: 10px; flex-wrap: wrap; }
  h2 { margin: 0; font-size: 1rem; color: var(--accent); font-weight: 600; }
  .subtitle { font-size: 0.75rem; color: var(--muted); }
  .uvr5-header-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .gpu-badge {
    background: rgba(74, 222, 128, 0.12); color: var(--ok); border: 1px solid var(--ok); border-radius: 999px; padding: 4px 10px; font-size: 0.72rem; font-weight: 700; white-space: nowrap;
  }
  .profile-select { font-size: 0.75rem; color: var(--muted); }
  .profile-select select { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; margin-left: 4px; font-size: 0.74rem; }
  .pick-file-btn {
    background: rgba(118, 210, 219, 0.16); color: var(--accent); border: 1px solid var(--accent); border-radius: 6px; padding: 6px 12px; font-weight: 700; cursor: pointer; font-size: 0.76rem;
  }
  .run-btn {
    background: var(--accent); color: #0b2b1e; border: 1px solid var(--accent); border-radius: 6px; padding: 8px 14px; font-weight: 600; cursor: pointer; font-size: 0.78rem;
  }
  .uvr5-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .uvr5-card { background: var(--sub-surface-2); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px; font-size: 0.76rem; display: flex; flex-direction: column; gap: 6px; }
  .uvr5-card input[type="text"], .uvr5-card select { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; width: 100%; box-sizing: border-box; margin-top: 2px; }
  .path-row { display: flex; align-items: center; gap: 6px; margin-top: 2px; }
  .path-row input[type="text"] { margin-top: 0; flex: 1; min-width: 0; }
  .folder-btn {
    background: var(--surface-2); color: var(--accent); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; font-size: 0.72rem; font-weight: 600; cursor: pointer; white-space: nowrap;
  }
  .folder-btn:hover { border-color: var(--accent); }
  h3 { margin: 0 0 4px 0; font-size: 0.82rem; color: var(--accent); font-weight: 600; }
  .stem-list { margin: 0; padding-left: 16px; display: flex; flex-direction: column; gap: 6px; word-break: break-all; }
  .empty-stems-hint { margin: 4px 0 0 0; color: var(--muted); line-height: 1.45; }
</style>

