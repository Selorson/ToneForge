<script lang="ts">
  import { DawTelemetryView } from '../state/daw_master_store.svelte';

  let { telemetry }: { telemetry: DawTelemetryView } = $props();

  const STEM_NAMES = ['Vocals', 'Drums', 'Bass', 'Other'];
  const IRC_MODES = ['IRC I', 'IRC II', 'IRC III', 'IRC IV', 'IRC V', 'IRC LL'];
  const EXCITER_MODES = ['Analog', 'Retro', 'Tape', 'Warm', '12AX7 Triode', 'Dual Triode', 'Tube'];
</script>

<section class="panel ozone12-panel">
  <header class="panel-header">
    <div>
      <h2>iZotope Ozone 12 Mastering Console & 120ms Real-Time BSRNN Stem Rebalance</h2>
      <span class="subtitle">
        12AX7 DK-Triode LUT Exciter · IRC V Multi-Band Maximizer (4× True-Peak) · Imager · Stabilizer · 0x09000 Telemetry
      </span>
    </div>
    <div class="lufs-readout">
      <span>M: <strong>{telemetry.lufsMomentary.toFixed(1)}</strong> LUFS</span>
      <span>S: <strong>{telemetry.lufsShortTerm.toFixed(1)}</strong> LUFS</span>
      <span>I: <strong>{telemetry.lufsIntegrated.toFixed(1)}</strong> LUFS</span>
      <span>TP L/R: <strong>{telemetry.truePeakDbL.toFixed(2)} / {telemetry.truePeakDbR.toFixed(2)}</strong> dBTP</span>
      <span>IRC V GR: <strong>{telemetry.masterGrDb.toFixed(1)}</strong> dB</span>
    </div>
  </header>

  <div class="ozone-grid">
    <!-- 1. IRC V Maximizer -->
    <div class="module-card">
      <h3>IRC V Maximizer & Soft Clipper</h3>
      <label>
        IRC Mode:
        <select
          value={telemetry.ozone12IrcMode}
          onchange={(e) => {
            telemetry.ozone12IrcMode = parseInt((e.currentTarget as HTMLSelectElement).value, 10);
            telemetry.syncOzone12ToEngine();
          }}
        >
          {#each IRC_MODES as m, idx}
            <option value={idx}>{m}</option>
          {/each}
        </select>
      </label>
      <label>
        Gain: +{telemetry.ozone12GainDb.toFixed(1)} dB
        <input
          type="range"
          min="0"
          max="18"
          step="0.1"
          bind:value={telemetry.ozone12GainDb}
          oninput={() => telemetry.syncOzone12ToEngine()}
        />
      </label>
      <label>
        Ceiling: {telemetry.ozone12CeilingDbtp.toFixed(2)} dBTP
        <input
          type="range"
          min="-6"
          max="0"
          step="0.05"
          bind:value={telemetry.ozone12CeilingDbtp}
          oninput={() => telemetry.syncOzone12ToEngine()}
        />
      </label>
      <label>
        Upward Comp: {(telemetry.ozone12UpwardCompWet * 100).toFixed(0)}%
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          bind:value={telemetry.ozone12UpwardCompWet}
          oninput={() => telemetry.syncOzone12ToEngine()}
        />
      </label>
      <label>
        Soft Clip Mix ({['Light', 'Medium', 'Heavy'][telemetry.ozone12SoftClipMode]}): {(telemetry.ozone12SoftClipMix * 100).toFixed(0)}%
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          bind:value={telemetry.ozone12SoftClipMix}
          oninput={() => telemetry.syncOzone12ToEngine()}
        />
      </label>
    </div>

    <!-- 2. 12AX7 DK-Triode Exciter -->
    <div class="module-card">
      <h3>12AX7 DK-Triode 4-Band Exciter</h3>
      {#each ['Low', 'Low-Mid', 'High-Mid', 'High'] as bandName, bIdx}
        <div class="band-row">
          <span>{bandName} ({EXCITER_MODES[telemetry.ozone12ExciterModes[bIdx]]})</span>
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={telemetry.ozone12ExciterDrivesPct[bIdx]}
            oninput={(e) => {
              telemetry.ozone12ExciterDrivesPct[bIdx] = parseFloat(
                (e.currentTarget as HTMLInputElement).value
              );
              telemetry.syncOzone12ToEngine();
            }}
          />
          <span>{telemetry.ozone12ExciterDrivesPct[bIdx].toFixed(0)}%</span>
        </div>
      {/each}
    </div>

    <!-- 3. Multiband Imager & Stabilizer -->
    <div class="module-card">
      <h3>Multiband Imager & Stabilizer</h3>
      {#each ['Sub', 'Low-Mid', 'Presence', 'Air'] as bLabel, bIdx}
        <div class="band-row">
          <span>{bLabel} Width</span>
          <input
            type="range"
            min="-100"
            max="100"
            step="1"
            value={telemetry.ozone12ImagerWidthsPct[bIdx]}
            oninput={(e) => {
              telemetry.ozone12ImagerWidthsPct[bIdx] = parseFloat(
                (e.currentTarget as HTMLInputElement).value
              );
              telemetry.syncOzone12ToEngine();
            }}
          />
          <span>{telemetry.ozone12ImagerWidthsPct[bIdx].toFixed(0)}%</span>
        </div>
      {/each}
      <label>
        Stabilizer Adaptive Amount: {telemetry.ozone12StabilizerAmountPct.toFixed(0)}%
        <input
          type="range"
          min="0"
          max="100"
          step="1"
          bind:value={telemetry.ozone12StabilizerAmountPct}
          oninput={() => telemetry.syncOzone12ToEngine()}
        />
      </label>
    </div>

    <!-- 4. Real-Time 120ms BSRNN Master Rebalance -->
    <div class="module-card">
      <h3>120ms Real-Time BSRNN Master Rebalance</h3>
      {#each STEM_NAMES as stemName, sIdx}
        <div class="band-row">
          <span>{stemName} ({telemetry.ozone12StemRmsDb[sIdx]?.toFixed(1) ?? '-12.0'} dB)</span>
          <input
            type="range"
            min="-12"
            max="12"
            step="0.1"
            value={telemetry.bsrnnStemGainsDb[sIdx]}
            oninput={(e) => {
              telemetry.bsrnnStemGainsDb[sIdx] = parseFloat(
                (e.currentTarget as HTMLInputElement).value
              );
              telemetry.syncBsrnnStemsToEngine();
            }}
          />
          <span>{telemetry.bsrnnStemGainsDb[sIdx] >= 0 ? '+' : ''}{telemetry.bsrnnStemGainsDb[sIdx].toFixed(1)} dB</span>
        </div>
      {/each}
    </div>
  </div>
</section>

<style>
  .panel { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px; }
  h2 { margin: 0; font-size: 1rem; color: #34d399; }
  .subtitle { font-size: 0.75rem; color: #94a3b8; }
  .lufs-readout { display: flex; gap: 12px; font-size: 0.78rem; background: #090d16; padding: 6px 10px; border-radius: 6px; border: 1px solid #1e293b; }
  .ozone-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  .module-card { background: #090d16; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; font-size: 0.75rem; display: flex; flex-direction: column; gap: 6px; }
  h3 { margin: 0 0 4px 0; font-size: 0.82rem; color: #38bdf8; }
  .band-row { display: flex; justify-content: space-between; align-items: center; gap: 6px; }
</style>
