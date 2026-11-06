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
          stroke="#76D2DB"
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
          fill="#1a2a2d"
          stroke={node.pluginKind === 'FixedLatency' ? '#fbbf24' : '#76D2DB'}
          stroke-width="1.5"
        />
        <text x="10" y="20" fill="#F7F6E5" font-size="11" font-weight="600">{node.label}</text>
        <text x="10" y="38" fill="rgba(247, 246, 229, 0.5)" font-size="10">
          L{node.layer} · {node.latencySamples} smp · CPU {node.cpuPct.toFixed(1)}%
        </text>
      </g>
    {/each}
  </svg>
</section>

<style>
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; }
  h2 { margin: 0; font-size: 0.95rem; color: var(--accent); font-weight: 600; }
  .subtitle { font-size: 0.75rem; color: var(--muted); }
  .patcher-svg { width: 100%; height: 220px; background: var(--sub-surface); border: 1px solid var(--border); border-radius: var(--radius); margin-top: 8px; }
</style>

