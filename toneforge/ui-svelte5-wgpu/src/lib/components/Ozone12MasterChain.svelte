<script lang="ts">
  import { DawTelemetryView } from '../state/daw_master_store.svelte';
  import Guide from './Guide.svelte';

  let { telemetry }: { telemetry: DawTelemetryView } = $props();

  const STEM_NAMES = ['Vocals', 'Drums', 'Bass', 'Other'];
  const IRC_MODES = ['IRC I', 'IRC II', 'IRC III', 'IRC IV', 'IRC V', 'IRC LL'];
  const EXCITER_MODES = ['Analog', 'Retro', 'Tape', 'Warm', '12AX7 Triode', 'Dual Triode', 'Tube'];
  const OZONE12_GENRES = [
    'AllPurpose',
    'Pop',
    'EDM',
    'HipHop',
    'Trap',
    'Rock',
    'LoFi',
    'KPopJPop',
    'RnbSoul',
    'Jazz',
    'Orchestral',
    'Modern',
    'DancePop',
    'House',
    'Techno',
    'DrumAndBass',
    'Dubstep',
    'FutureBass',
    'Metal',
    'Reggaeton',
  ];
  const OZONE12_GLOBAL_PRESETS = [
    'Modern Clarity',
    'Aggro Dance',
    'Clean & Wide',
    'Warm & Open',
    'Punchy Low End',
    'CD Master',
    'Streaming Master (-14 LUFS)',
    'pop-modern-pro.xml',
    'edm-modern-pro.xml',
    'hiphop-modern-pro.xml',
    'rock-modern-pro.xml',
  ];
</script>

<section class="panel ozone12-panel">
  <Guide id="ozone12" />
  <header class="panel-header">
    <div>
      <h2>iZotope Ozone 12 Mastering Console & 120ms Real-Time BSRNN Stem Rebalance</h2>
      <span class="subtitle">
        12AX7 DK-Triode LUT Exciter · IRC V Multi-Band Maximizer (4× True-Peak) · Imager · Stabilizer · {telemetry.ozone12AiAssistantStatus}
      </span>
    </div>
    <div class="ozone-preset-bar">
      <label>
        Genre Target Curve (36 Real IZRESEMBED):
        <select
          value={telemetry.ozone12SelectedGenre}
          onchange={(e) =>
            telemetry.loadOzone12GenrePreset((e.currentTarget as HTMLSelectElement).value)}
        >
          {#each OZONE12_GENRES as g}
            <option value={g}>{g === 'AllPurpose' ? 'AllPurpose (Neutral Reference)' : g}</option>
          {/each}
        </select>
      </label>
      <label>
        Global Mastering Preset (82 Real XML):
        <select
          value={telemetry.ozone12SelectedGlobalXml}
          onchange={(e) =>
            telemetry.loadOzone12GlobalXmlPreset((e.currentTarget as HTMLSelectElement).value)}
        >
          {#each OZONE12_GLOBAL_PRESETS as gp}
            <option value={gp}>{gp}</option>
          {/each}
        </select>
      </label>
      <button class="ai-assist-btn" onclick={() => telemetry.runOzone12AiMasterAssistant()}>
        ✨ AI Master Assistant (Auto-Match)
      </button>
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
          <div class="stem-sm-btns">
            <button
              class="sm-btn solo {telemetry.bsrnnStemSolo[sIdx] ? 'active' : ''}"
              onclick={() => telemetry.toggleBsrnnStemSolo(sIdx)}
              title="Solo {stemName} Stem"
            >
              S
            </button>
            <button
              class="sm-btn mute {telemetry.bsrnnStemMute[sIdx] ? 'active' : ''}"
              onclick={() => telemetry.toggleBsrnnStemMute(sIdx)}
              title="Mute {stemName} Stem"
            >
              M
            </button>
          </div>
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

    <!-- 5. Low End Focus (20–300 Hz Envelope Contrast) -->
    <div class="module-card">
      <div class="card-head-inline">
        <h3>Low End Focus (20–300 Hz)</h3>
        <label class="chk-inline">
          <input
            type="checkbox"
            bind:checked={telemetry.ozone12LefEnabled}
            onchange={() => telemetry.syncOzone12LowEndFocus()}
          />
          Active
        </label>
      </div>
      <div class="mode-pair">
        <button
          class="sm-btn {telemetry.ozone12LefPunchy ? 'solo active' : ''}"
          onclick={() => {
            telemetry.ozone12LefPunchy = true;
            telemetry.ozone12LefEnabled = true;
            telemetry.syncOzone12LowEndFocus();
          }}
        >
          🥁 Punchy (Transient)
        </button>
        <button
          class="sm-btn {!telemetry.ozone12LefPunchy ? 'solo active' : ''}"
          onclick={() => {
            telemetry.ozone12LefPunchy = false;
            telemetry.ozone12LefEnabled = true;
            telemetry.syncOzone12LowEndFocus();
          }}
        >
          🌊 Smooth (Glue)
        </button>
        <button
          class="sm-btn solo {telemetry.ozone12LefSolo ? 'active' : ''}"
          onclick={() => {
            telemetry.ozone12LefSolo = !telemetry.ozone12LefSolo;
            telemetry.syncOzone12LowEndFocus();
          }}
          title="Nghe riêng dải trầm Low-End dưới ngưỡng Cutoff"
        >
          Solo Low
        </button>
      </div>
      <label>
        Cutoff: {telemetry.ozone12LefUpperHz.toFixed(0)} Hz
        <input
          type="range"
          min="20"
          max="300"
          step="5"
          bind:value={telemetry.ozone12LefUpperHz}
          oninput={() => telemetry.syncOzone12LowEndFocus()}
        />
      </label>
      <label>
        Contrast: {telemetry.ozone12LefContrastPct >= 0 ? '+' : ''}{telemetry.ozone12LefContrastPct.toFixed(0)}% (Δ {telemetry.ozone12LefDeltaDb.toFixed(2)} dB)
        <input
          type="range"
          min="-100"
          max="100"
          step="2"
          bind:value={telemetry.ozone12LefContrastPct}
          oninput={() => telemetry.syncOzone12LowEndFocus()}
        />
      </label>
      <label>
        Low Gain: {telemetry.ozone12LefGainDb >= 0 ? '+' : ''}{telemetry.ozone12LefGainDb.toFixed(1)} dB
        <input
          type="range"
          min="-12"
          max="12"
          step="0.2"
          bind:value={telemetry.ozone12LefGainDb}
          oninput={() => telemetry.syncOzone12LowEndFocus()}
        />
      </label>
    </div>

    <!-- 6. Psychoacoustic Codec Preview (MDCT + 24-Bark Masking) -->
    <div class="module-card">
      <div class="card-head-inline">
        <h3>Psychoacoustic Codec Preview</h3>
        <label class="chk-inline">
          <input
            type="checkbox"
            bind:checked={telemetry.ozone12CodecEnabled}
            onchange={() => telemetry.syncOzone12CodecPreview()}
          />
          Audition
        </label>
      </div>
      <label>
        Streaming Codec Target:
        <select
          value={telemetry.ozone12CodecMode}
          onchange={(e) => {
            telemetry.ozone12CodecMode = parseInt(
              (e.currentTarget as HTMLSelectElement).value,
              10
            ) as 0 | 1 | 2 | 3 | 4 | 5;
            telemetry.ozone12CodecEnabled = telemetry.ozone12CodecMode !== 0;
            telemetry.syncOzone12CodecPreview();
          }}
        >
          <option value={0}>PCM Lossless Bypass</option>
          <option value={1}>MP3 128 kbps (MDCT 24-Bark)</option>
          <option value={2}>MP3 320 kbps (High Bitrate)</option>
          <option value={3}>AAC 128 kbps (YouTube / Web)</option>
          <option value={4}>AAC 256 kbps (Apple Music)</option>
          <option value={5}>Opus 128 kbps (Spotify / VoIP)</option>
        </select>
      </label>
      <div class="band-row">
        <span>Artifact Solo (Nghe riêng nhiễu nén):</span>
        <button
          class="sm-btn {telemetry.ozone12CodecArtifactSolo ? 'mute active' : ''}"
          onclick={() => {
            telemetry.ozone12CodecArtifactSolo = !telemetry.ozone12CodecArtifactSolo;
            telemetry.ozone12CodecEnabled = true;
            telemetry.syncOzone12CodecPreview();
          }}
        >
          {telemetry.ozone12CodecArtifactSolo ? '🔊 Artifact ON' : 'Artifact Solo'}
        </button>
      </div>
      <div
        class="codec-tp-box {Math.max(telemetry.ozone12PostCodecTpL, telemetry.ozone12PostCodecTpR) > -0.1 ? 'clip-warn' : ''}"
      >
        <span>
          Post-Codec 4× TP:
          <strong>
            {telemetry.ozone12PostCodecTpL.toFixed(2)} / {telemetry.ozone12PostCodecTpR.toFixed(2)} dBTP
          </strong>
        </span>
        <span>Artifact Noise: <strong>{telemetry.ozone12CodecArtifactRmsDb.toFixed(1)} dB</strong></span>
      </div>
    </div>
  </div>
</section>

<style>
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px; }
  h2 { margin: 0; font-size: 1rem; color: var(--ok); font-weight: 600; }
  .subtitle { font-size: 0.75rem; color: var(--muted); }
  .ozone-preset-bar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 0.75rem; color: var(--muted); }
  .ozone-preset-bar select { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; margin-left: 4px; font-size: 0.74rem; }
  .ai-assist-btn { background: var(--accent); color: #0b2b1e; border: 1px solid var(--accent); border-radius: 6px; padding: 5px 10px; font-weight: 600; cursor: pointer; font-size: 0.75rem; }
  .lufs-readout { display: flex; gap: 12px; font-size: 0.76rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; background: var(--sub-surface); padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border); color: var(--muted); }
  .lufs-readout strong { color: var(--accent); }
  .ozone-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
  .module-card { background: var(--sub-surface-2); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px; font-size: 0.75rem; font-family: var(--font-mono); font-variant-numeric: tabular-nums; display: flex; flex-direction: column; gap: 6px; }
  .module-card select { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; }
  .card-head-inline { display: flex; justify-content: space-between; align-items: center; gap: 6px; }
  .chk-inline { display: flex; align-items: center; gap: 4px; font-size: 0.71rem; color: var(--ok); font-weight: 600; }
  .mode-pair { display: flex; gap: 5px; flex-wrap: wrap; }
  .codec-tp-box { display: flex; flex-direction: column; gap: 3px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; font-size: 0.71rem; }
  .codec-tp-box strong { color: var(--ok); }
  .codec-tp-box.clip-warn { border-color: var(--err); background: rgba(239, 68, 68, 0.14); }
  .codec-tp-box.clip-warn strong { color: var(--err); }
  h3 { margin: 0 0 4px 0; font-size: 0.82rem; font-family: var(--font-sans); color: var(--accent); font-weight: 600; }
  .band-row { display: flex; justify-content: space-between; align-items: center; gap: 6px; }
  .stem-sm-btns { display: flex; gap: 3px; }
  .sm-btn { background: var(--surface-2); color: var(--muted); border: 1px solid var(--border); border-radius: 4px; padding: 2px 5px; font-size: 0.68rem; font-weight: 700; cursor: pointer; font-family: var(--font-mono); }
  .sm-btn.solo.active { background: var(--warn); color: #0b2b1e; border-color: var(--warn); }
  .sm-btn.mute.active { background: var(--err); color: #0b2b1e; border-color: var(--err); }
</style>
