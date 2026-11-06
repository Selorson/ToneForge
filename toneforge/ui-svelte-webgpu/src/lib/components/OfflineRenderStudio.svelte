<script lang="ts">
  import { DawTelemetryView } from '../state/daw_master_store.svelte';

  let { telemetry }: { telemetry: DawTelemetryView } = $props();

  const SAMPLE_RATES = [44100, 48000, 96000, 192000] as const;
</script>

<section class="panel render-panel">
  <header class="panel-header">
    <div>
      <h2>Offline Bounce & Mastering Render Engine (MBIT+ Dither & ITU-R BS.1770-4 Analysis)</h2>
      <span class="subtitle">
        16-Bit / 24-Bit PCM (Ozone 12 MBIT+ Psychoacoustic Noise-Shaping Dither) & 32-Bit IEEE Float · 44.1 / 48 / 96 / 192 kHz · Master / Split 502 Tracks / UVR5 Stem Mastering
      </span>
    </div>
    <button
      class="render-btn"
      disabled={telemetry.renderIsRunning}
      onclick={() => telemetry.executeOfflineRender()}
    >
      {telemetry.renderIsRunning ? 'Rendering Offline Bounce...' : 'Export Studio WAV & Analyze LUFS/dBTP'}
    </button>
  </header>

  <div class="render-grid">
    <!-- 1. EXPORT FORMAT & MODE CONFIGURATION -->
    <div class="render-card">
      <h3>Render Mode & Studio WAV Format</h3>

      <label>
        Render Mode:
        <select bind:value={telemetry.renderMode}>
          <option value="Master">Render Master (502-Track Sum + Ozone 12)</option>
          <option value="SplitMixerTracks">Split Mixer Tracks (Simultaneous Multi-Track Export)</option>
          <option value="StemMastering">UVR5 4-Stem Mastering (Ensemble → Pro-Q 4 + Soothe3 → Ozone 12)</option>
        </select>
      </label>

      <label>
        Bit Depth & Dither:
        <select bind:value={telemetry.renderBitDepth}>
          <option value="Pcm16MbitPlus">16-Bit PCM + Ozone 12 MBIT+ Noise-Shaping Dither</option>
          <option value="Pcm24MbitPlus">24-Bit PCM + Ozone 12 MBIT+ Noise-Shaping Dither</option>
          <option value="Float32Ieee">32-Bit IEEE Float (0x0003 Unclipped Mastering)</option>
        </select>
      </label>

      <label>
        Sample Rate:
        <select
          value={telemetry.renderSampleRateHz}
          onchange={(e) =>
            (telemetry.renderSampleRateHz = parseInt(
              (e.currentTarget as HTMLSelectElement).value,
              10
            ) as any)}
        >
          {#each SAMPLE_RATES as sr}
            <option value={sr}>{(sr / 1000).toFixed(1)} kHz ({sr} Hz)</option>
          {/each}
        </select>
      </label>

      <label>
        <input type="checkbox" bind:checked={telemetry.renderEnableNoiseShaping} />
        Enable Ozone 12 MBIT+ 2nd-Order Psychoacoustic Error-Feedback Filter
      </label>

      <label>
        <input type="checkbox" bind:checked={telemetry.renderTrimApdcLatency} />
        Exact 5-Point APDC Master Latency Compensation Trim ({telemetry.lastRenderReport.apdcLatencyTrimmedSamples} smp)
      </label>

      <label>
        Output Directory:
        <input type="text" bind:value={telemetry.renderOutputDir} />
      </label>
    </div>

    <!-- 2. POST-RENDER ITU-R BS.1770-4 & 4X TRUE PEAK TELEMETRY -->
    <div class="render-card">
      <h3>Post-Render Acoustic Analysis Report</h3>
      <div class="metrics-table">
        <div class="metric-row">
          <span>Integrated Loudness (BS.1770-4):</span>
          <strong>{telemetry.lastRenderReport.integratedLufs.toFixed(2)} LUFS</strong>
        </div>
        <div class="metric-row">
          <span>Short-Term Max Loudness (3s):</span>
          <strong>{telemetry.lastRenderReport.shortTermMaxLufs.toFixed(2)} LUFS</strong>
        </div>
        <div class="metric-row">
          <span>Momentary Max Loudness (400ms):</span>
          <strong>{telemetry.lastRenderReport.momentaryMaxLufs.toFixed(2)} LUFS</strong>
        </div>
        <div class="metric-row">
          <span>4x Oversampled True Peak (L / R):</span>
          <strong>
            {telemetry.lastRenderReport.truePeakDbtpL.toFixed(2)} / {telemetry.lastRenderReport.truePeakDbtpR.toFixed(2)} dBTP
          </strong>
        </div>
        <div class="metric-row">
          <span>Dynamic Range Crest Factor:</span>
          <strong>{telemetry.lastRenderReport.crestFactorDb.toFixed(2)} dB</strong>
        </div>
        <div class="metric-row">
          <span>Stereo Phase Correlation (-1..+1):</span>
          <strong>{telemetry.lastRenderReport.phaseCorrelation.toFixed(3)}</strong>
        </div>
      </div>
    </div>

    <!-- 3. STREAMING PLATFORM COMPLIANCE & EXPORTED FILES -->
    <div class="render-card">
      <h3>Streaming Broadcast Compliance & Rendered Stems</h3>

      <div class="compliance-badges">
        <div class="platform-badge {telemetry.lastRenderReport.spotifyCompliant ? 'pass' : 'warn'}">
          <span>Spotify (-14 LUFS / -1.0 dBTP)</span>
          <strong>
            {telemetry.lastRenderReport.spotifyCompliant ? 'PASS' : 'TRIM'} ({telemetry.lastRenderReport.spotifyPenaltyDb.toFixed(1)} dB)
          </strong>
        </div>
        <div class="platform-badge {telemetry.lastRenderReport.appleMusicCompliant ? 'pass' : 'warn'}">
          <span>Apple Music (-16 LUFS / -1.0 dBTP)</span>
          <strong>
            {telemetry.lastRenderReport.appleMusicCompliant ? 'PASS' : 'TRIM'} ({telemetry.lastRenderReport.appleMusicPenaltyDb.toFixed(1)} dB)
          </strong>
        </div>
        <div class="platform-badge {telemetry.lastRenderReport.youtubeCompliant ? 'pass' : 'warn'}">
          <span>YouTube (-14 LUFS / -1.0 dBTP)</span>
          <strong>
            {telemetry.lastRenderReport.youtubeCompliant ? 'PASS' : 'TRIM'} ({telemetry.lastRenderReport.youtubePenaltyDb.toFixed(1)} dB)
          </strong>
        </div>
      </div>

      <ul class="rendered-files">
        {#each telemetry.lastRenderReport.renderedFiles as filePath}
          <li><code>{filePath}</code></li>
        {/each}
      </ul>
    </div>
  </div>
</section>

<style>
  .panel { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
  h2 { margin: 0; font-size: 1rem; color: #10b981; }
  .subtitle { font-size: 0.75rem; color: #94a3b8; }
  .render-btn { background: #10b981; color: #090d16; border: none; border-radius: 6px; padding: 8px 14px; font-weight: 700; cursor: pointer; font-size: 0.78rem; }
  .render-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .render-card { background: #090d16; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; font-size: 0.75rem; display: flex; flex-direction: column; gap: 6px; }
  h3 { margin: 0 0 4px 0; font-size: 0.82rem; color: #6ee7b7; }
  .render-card select, .render-card input[type="text"] { background: #0f172a; color: #f8fafc; border: 1px solid #334155; border-radius: 4px; padding: 4px 6px; width: 100%; box-sizing: border-box; margin-top: 2px; }
  .metrics-table { display: flex; flex-direction: column; gap: 6px; font-family: monospace; }
  .metric-row { display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid #131c31; }
  .metric-row strong { color: #38bdf8; }
  .compliance-badges { display: flex; flex-direction: column; gap: 5px; }
  .platform-badge { display: flex; justify-content: space-between; align-items: center; padding: 5px 8px; border-radius: 4px; font-size: 0.72rem; }
  .platform-badge.pass { background: rgba(16, 185, 129, 0.14); border: 1px solid #10b981; color: #6ee7b7; }
  .platform-badge.warn { background: rgba(245, 158, 11, 0.14); border: 1px solid #f59e0b; color: #fcd34d; }
  .rendered-files { margin: 4px 0 0 0; padding-left: 16px; font-size: 0.7rem; color: #cbd5e1; display: flex; flex-direction: column; gap: 3px; }
</style>
