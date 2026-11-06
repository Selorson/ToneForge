<script lang="ts">
  import { DawTelemetryView } from './lib/state/daw_master_store.svelte';

  const sharedBuffer = new ArrayBuffer(131072);
  const telemetry = new DawTelemetryView(sharedBuffer);
</script>

<main class="toneforge-shell">
  <header class="transport-bar">
    <h1>ToneForge DAW</h1>
    <div class="transport-metrics">
      <span>BPM: {telemetry.tempoBpm.toFixed(2)}</span>
      <span>PPQ: {telemetry.playheadPpq.toFixed(3)}</span>
      <span>CPU: {telemetry.cpuLoadPercent.toFixed(1)}%</span>
      <span>XRuns: {telemetry.xrunCount}</span>
    </div>
  </header>
</main>
