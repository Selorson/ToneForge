<script lang="ts">
  import { DawTelemetryView } from '../state/daw_master_store.svelte';

  let { telemetry }: { telemetry: DawTelemetryView } = $props();

  const ENSEMBLE_MODES = ['MAX_SPEC', 'MIN_SPEC', 'AVERAGE', 'MEDIAN', 'WEIGHTED'] as const;
</script>

<section class="panel uvr5-panel">
  <header class="panel-header">
    <div>
      <h2>UVR5 Multi-Architecture Offline Stem Separation & Spectral Ensemble Studio</h2>
      <span class="subtitle">
        MDX23C (TFC-TDF v3 CaC + 2-Pass Denoise) · HTDemucs v4 (CrossTransformer) · VR CascadedASPPNet ([1,2,4,8,16]) · SpectralEnsembler
      </span>
    </div>
    <button
      class="run-btn"
      disabled={telemetry.uvr5IsProcessing}
      onclick={() => telemetry.runUvr5SeparationJob()}
    >
      {telemetry.uvr5IsProcessing ? 'Separating Stems...' : 'Run Ensemble & Bounce to Mixer 1..4'}
    </button>
  </header>

  <div class="uvr5-grid">
    <div class="uvr5-card">
      <h3>Source Audio & Spectral Ensemble Mode</h3>
      <label>
        Input File:
        <input type="text" bind:value={telemetry.uvr5AudioPath} />
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
      <ul class="stem-list">
        {#each telemetry.uvr5SeparatedStems as stemPath, idx}
          <li>
            <strong>Insert Track {idx + 1}:</strong> <code>{stemPath}</code>
          </li>
        {/each}
      </ul>
    </div>
  </div>
</section>

<style>
  .panel { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  h2 { margin: 0; font-size: 1rem; color: #a78bfa; }
  .subtitle { font-size: 0.75rem; color: #94a3b8; }
  .run-btn {
    background: #8b5cf6; color: #ffffff; border: none; border-radius: 6px; padding: 8px 14px; font-weight: 700; cursor: pointer; font-size: 0.78rem;
  }
  .uvr5-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .uvr5-card { background: #090d16; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; font-size: 0.76rem; display: flex; flex-direction: column; gap: 6px; }
  h3 { margin: 0 0 4px 0; font-size: 0.82rem; color: #c4b5fd; }
  .stem-list { margin: 0; padding-left: 16px; display: flex; flex-direction: column; gap: 6px; }
</style>
