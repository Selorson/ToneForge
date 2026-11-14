<script lang="ts">
  import { DawTelemetryView } from '../state/daw_master_store.svelte';
  import Guide from './Guide.svelte';

  let { telemetry }: { telemetry: DawTelemetryView } = $props();

  let targetTrack = $state(1);
  let targetSlot = $state(3);
  let loadAsInstrument = $state(false);
  let customBinaryPath = $state('target/release/test_vst3_clap_plugin.dll');
  let customFormatClap = $state(true);
</script>

<section class="panel plugin-host-panel">
  <Guide id="plugins" />
  <header class="panel-header">
    <div>
      <h2>
        🔌 Steinberg VST3 3.7 COM & CLAP 1.2 C ABI External Plugin Host + Crash-Isolated Scanner
      </h2>
      <span class="subtitle">
        Zero-Alloc RT Process Slice · 5-Point APDC Auto-Sync · Win32 HWND Native GUI Hosting (`IPlugView` / `clap_plugin_gui_t`)
      </span>
    </div>
    <div class="scan-controls">
      <button
        class="scan-btn"
        disabled={telemetry.pluginScanRunning}
        onclick={() => telemetry.scanExternalPlugins()}
      >
        {telemetry.pluginScanRunning
          ? '⏳ Đang quét cô lập...'
          : '🔍 Quét Plugin VST3 / CLAP (Crash-Isolated)'}
      </button>
    </div>
  </header>

  <div class="host-toolbar">
    <label>
      Target Mixer Track (0..501):
      <input
        type="number"
        min="0"
        max="501"
        bind:value={targetTrack}
        class="num-input"
      />
    </label>

    <label>
      Insert FX Slot (1..10):
      <select
        value={targetSlot}
        onchange={(e) =>
          (targetSlot = parseInt((e.currentTarget as HTMLSelectElement).value, 10))}
      >
        {#each Array(10) as _, sIdx}
          <option value={sIdx}>Slot {sIdx + 1}</option>
        {/each}
      </select>
    </label>

    <label>
      <input type="checkbox" bind:checked={loadAsInstrument} />
      🎹 Load as Channel Synth Instrument
    </label>

    <div class="direct-load-box">
      <input
        type="text"
        bind:value={customBinaryPath}
        placeholder="Đường dẫn file .vst3 / .clap / .dll..."
        class="path-input"
      />
      <select
        value={customFormatClap ? 'clap' : 'vst3'}
        onchange={(e) => (customFormatClap = (e.currentTarget as HTMLSelectElement).value === 'clap')}
      >
        <option value="clap">CLAP 1.2</option>
        <option value="vst3">VST3 3.7</option>
      </select>
      <button
        class="load-btn"
        onclick={() =>
          telemetry.loadExternalPluginSlot(
            targetTrack,
            targetSlot,
            customBinaryPath,
            null,
            customFormatClap,
            loadAsInstrument
          )}
      >
        ⚡ Nạp trực tiếp
      </button>
    </div>
  </div>

  <div class="status-bar">
    <span>Host Status: <strong>{telemetry.pluginHostStatus}</strong></span>
    <span>
      Discovered: <strong>{telemetry.scannedExternalPlugins.length}</strong> · Active Slots:
      <strong>{telemetry.loadedExternalPlugins.length}</strong> · Quarantined:
      <strong>{telemetry.quarantinedPlugins.length}</strong>
    </span>
  </div>

  {#if telemetry.quarantinedPlugins.length > 0}
    <div class="quarantine-alert">
      <strong>🛡️ Crash-Isolated Quarantine ({telemetry.quarantinedPlugins.length} faulty binaries isolated without crashing DAW):</strong>
      <ul>
        {#each telemetry.quarantinedPlugins as qp}
          <li><code>{qp.binary_path}</code> — {qp.reason}</li>
        {/each}
      </ul>
    </div>
  {/if}

  <div class="host-columns">
    <!-- 1. DISCOVERED VST3 & CLAP CATALOG -->
    <div class="catalog-box">
      <h3>Discovered VST3 & CLAP Plugins ({telemetry.scannedExternalPlugins.length})</h3>
      <div class="plugin-list">
        {#each telemetry.scannedExternalPlugins as plug (plug.format + ':' + plug.plugin_id + ':' + plug.binary_path)}
          <div class="plugin-card">
            <div class="card-top">
              <span class="fmt-badge {plug.format.toLowerCase()}">{plug.format}</span>
              <strong>{plug.name}</strong>
              <span class="type-tag">{plug.is_instrument ? '🎹 Synth/FX' : '🎛️ Audio FX'}</span>
            </div>
            <div class="card-meta">
              <span>{plug.vendor} · v{plug.version} · {plug.param_count} params</span>
              <code>{plug.binary_path}</code>
            </div>
            <button
              class="slot-load-btn"
              onclick={() =>
                telemetry.loadExternalPluginSlot(
                  targetTrack,
                  targetSlot,
                  plug.binary_path,
                  plug.plugin_id,
                  plug.format === 'CLAP',
                  loadAsInstrument
                )}
            >
              ⚡ Load → Track #{targetTrack} · Slot #{targetSlot + 1}
            </button>
          </div>
        {/each}
      </div>
    </div>

    <!-- 2. ACTIVE LOADED EXTERNAL PLUGINS & NATIVE HWND / PARAMETER RACK -->
    <div class="active-rack-box">
      <h3>Active Loaded External Plugins ({telemetry.loadedExternalPlugins.length})</h3>
      {#if telemetry.loadedExternalPlugins.length === 0}
        <p class="empty-hint">
          Chưa có plugin rời nào đang gắn trên Mixer. Bấm <strong>⚡ Load</strong> ở cột bên trái để gắn VST3 hoặc CLAP thực vào Kênh và điều khiển tham số / mở cửa sổ gốc Win32 HWND.
        </p>
      {:else}
        <div class="loaded-list">
          {#each telemetry.loadedExternalPlugins as lp (lp.trackIdx + ':' + lp.slotIdx)}
            <div class="loaded-card">
              <div class="loaded-head">
                <div>
                  <span class="fmt-badge {lp.isClap ? 'clap' : 'vst3'}">
                    {lp.isClap ? 'CLAP 1.2' : 'VST3 3.7'}
                  </span>
                  <strong>Track #{lp.trackIdx} · Slot #{lp.slotIdx + 1}: {lp.name}</strong>
                </div>
                <div class="loaded-badges">
                  <span class="apdc-pill">APDC +{lp.latencySamples} smp</span>
                  <button
                    class="hwnd-btn {lp.guiOpen ? 'open' : ''}"
                    onclick={() => telemetry.toggleExternalPluginGui(lp)}
                  >
                    {lp.guiOpen ? '✕ Đóng GUI gốc' : '🗔 Mở Cửa Sổ Gốc (HWND)'}
                  </button>
                </div>
              </div>

              {#if lp.parameters.length > 0}
                <div class="param-grid">
                  {#each lp.parameters as prm (prm.id)}
                    <label class="param-row">
                      <span>#{prm.id} {prm.name}: {(prm.current_normalized * 100).toFixed(0)}%</span>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={prm.current_normalized}
                        oninput={(e) =>
                          telemetry.setExternalPluginParam(
                            lp.trackIdx,
                            lp.slotIdx,
                            prm.id,
                            parseFloat((e.currentTarget as HTMLInputElement).value)
                          )}
                      />
                    </label>
                  {/each}
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {/if}
    </div>
  </div>
</section>

<style>
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; display: flex; flex-direction: column; gap: 10px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
  h2 { margin: 0; font-size: 1rem; color: var(--accent); font-weight: 600; }
  h3 { margin: 0 0 8px 0; font-size: 0.84rem; color: var(--text); font-weight: 600; }
  .subtitle { font-size: 0.75rem; color: var(--muted); }
  .scan-btn { background: var(--accent); color: #0b2b1e; border: 1px solid var(--accent); border-radius: 6px; padding: 6px 12px; font-weight: 700; font-size: 0.76rem; cursor: pointer; }
  .host-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; background: var(--sub-surface-2); border: 1px solid var(--border); border-radius: var(--radius); padding: 8px 12px; font-size: 0.75rem; }
  .num-input { width: 64px; background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 5px; padding: 4px 6px; font-family: var(--font-mono); margin-left: 4px; }
  .host-toolbar select { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 5px; padding: 4px 8px; margin-left: 4px; font-size: 0.74rem; }
  .direct-load-box { display: flex; align-items: center; gap: 6px; flex: 1; min-width: 280px; }
  .path-input { flex: 1; background: var(--sub-surface); color: var(--text); border: 1px solid var(--border); border-radius: 5px; padding: 4px 8px; font-family: var(--font-mono); font-size: 0.73rem; }
  .load-btn { background: var(--ok); color: #0b2b1e; border: 1px solid var(--ok); border-radius: 5px; padding: 4px 10px; font-weight: 700; font-size: 0.74rem; cursor: pointer; }
  .status-bar { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 0.74rem; font-family: var(--font-mono); color: var(--muted); background: var(--sub-surface); border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; }
  .status-bar strong { color: var(--accent); }
  .quarantine-alert { background: rgba(239, 68, 68, 0.14); border: 1px solid var(--err); border-radius: 6px; padding: 8px 12px; font-size: 0.74rem; color: var(--err); }
  .quarantine-alert ul { margin: 4px 0 0 18px; padding: 0; }
  .host-columns { display: grid; grid-template-columns: 1fr 1.15fr; gap: 12px; }
  .catalog-box, .active-rack-box { background: var(--sub-surface-2); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px; }
  .plugin-list, .loaded-list { display: flex; flex-direction: column; gap: 8px; max-height: 340px; overflow-y: auto; }
  .plugin-card, .loaded-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; display: flex; flex-direction: column; gap: 6px; font-size: 0.74rem; }
  .card-top, .loaded-head { display: flex; justify-content: space-between; align-items: center; gap: 6px; flex-wrap: wrap; }
  .fmt-badge { padding: 2px 7px; border-radius: 4px; font-size: 0.68rem; font-weight: 700; font-family: var(--font-mono); }
  .fmt-badge.clap { background: rgba(74, 222, 128, 0.2); color: var(--ok); border: 1px solid var(--ok); }
  .fmt-badge.vst3 { background: rgba(118, 210, 219, 0.2); color: var(--accent); border: 1px solid var(--accent); }
  .type-tag { font-size: 0.7rem; color: var(--muted); }
  .card-meta { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px; font-size: 0.7rem; color: var(--muted); font-family: var(--font-mono); }
  .slot-load-btn { align-self: flex-start; background: var(--surface-2); color: var(--accent); border: 1px solid var(--accent); border-radius: 5px; padding: 4px 10px; font-size: 0.72rem; font-weight: 700; cursor: pointer; }
  .loaded-badges { display: flex; align-items: center; gap: 6px; }
  .apdc-pill { background: var(--sub-surface); color: var(--warn); border: 1px solid var(--warn); border-radius: 4px; padding: 2px 6px; font-size: 0.68rem; font-family: var(--font-mono); }
  .hwnd-btn { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 5px; padding: 4px 9px; font-size: 0.72rem; font-weight: 600; cursor: pointer; }
  .hwnd-btn.open { background: var(--warn); color: #0b2b1e; border-color: var(--warn); }
  .param-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; padding-top: 4px; border-top: 1px solid var(--border); font-family: var(--font-mono); font-size: 0.71rem; }
  .param-row { display: flex; flex-direction: column; gap: 3px; }
  .empty-hint { font-size: 0.75rem; color: var(--muted); margin: 8px 0; line-height: 1.5; }
</style>
