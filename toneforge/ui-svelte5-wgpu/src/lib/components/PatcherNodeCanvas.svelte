<script lang="ts">
  import { DawTelemetryView } from '../state/daw_master_store.svelte';

  let { telemetry }: { telemetry: DawTelemetryView } = $props();

  const totalPatcherLatency = $derived(
    Math.max(...telemetry.patcherNodes.map((n) => n.latencySamples), 0)
  );
</script>

<section class="panel patcher-panel">
  <header class="panel-header">
    <div>
      <h2>Modular Patcher Sub-DAG (Sugiyama Layered Layout & Parallel APDC)</h2>
      <span class="subtitle">
        Automatic Cycle-Free Topological Scheduling · Max Parallel Branch Latency: {totalPatcherLatency} samples
      </span>
    </div>
  </header>

  <svg viewBox="0 0 880 280" class="patcher-svg">
    {#each telemetry.patcherEdges as edge}
      {@const src = telemetry.patcherNodes.find((n) => n.nodeIndex === edge.fromNode)}
      {@const dst = telemetry.patcherNodes.find((n) => n.nodeIndex === edge.toNode)}
      {#if src && dst}
        <path
          d="M {src.xPx + 145} {src.yPx + 24} C {(src.xPx + dst.xPx) * 0.5 + 70} {src.yPx + 24}, {(src.xPx + dst.xPx) * 0.5 + 70} {dst.yPx + 24}, {dst.xPx} {dst.yPx + 24}"
          fill="none"
          stroke="#38bdf8"
          stroke-width="2.5"
        />
      {/if}
    {/each}

    {#each telemetry.patcherNodes as node (node.nodeIndex)}
      <g transform="translate({node.xPx}, {node.yPx})">
        <rect
          width="145"
          height="52"
          rx="6"
          fill="#1e293b"
          stroke={node.pluginKind === 'FixedLatency' ? '#facc15' : '#38bdf8'}
          stroke-width="1.5"
        />
        <text x="10" y="20" fill="#f8fafc" font-size="11" font-weight="600">{node.label}</text>
        <text x="10" y="38" fill="#94a3b8" font-size="10">
          L{node.layer} · {node.latencySamples} smp · CPU {node.cpuPct.toFixed(1)}%
        </text>
      </g>
    {/each}
  </svg>
</section>

<style>
  .panel { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; }
  h2 { margin: 0; font-size: 0.95rem; color: #38bdf8; }
  .subtitle { font-size: 0.75rem; color: #94a3b8; }
  .patcher-svg { width: 100%; height: 220px; background: #090d16; border: 1px solid #1e293b; border-radius: 6px; margin-top: 8px; }
</style>
