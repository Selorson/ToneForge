<script lang="ts">
  import { DawTelemetryView } from '../state/daw_master_store.svelte';

  let { telemetry }: { telemetry: DawTelemetryView } = $props();

  let svgEl = $state<SVGSVGElement | null>(null);
  let draggingNodeIdx = $state<number | null>(null);
  let dragOffset = $state<{ dx: number; dy: number }>({ dx: 0, dy: 0 });
  let dragMoved = $state(false);
  let newCordFrom = $state(0);
  let newCordTo = $state(4);
  let cordStatusMsg = $state('');

  const criticalPathPatcherLatency = $derived.by(() => {
    const nodes = telemetry.patcherNodes;
    const edges = telemetry.patcherEdges;
    if (!nodes.length) return 0;
    const hostIn = nodes.find((n) => n.pluginKind === 'HostIn');
    const hostOut = nodes.find((n) => n.pluginKind === 'HostOut' || n.pluginKind === 'Ozone12');
    if (!hostIn || !hostOut) {
      return Math.max(
        ...nodes.map((n) => (n.enabled === false ? 0 : n.latencySamples)),
        0
      );
    }
    if (hostIn.enabled === false || hostOut.enabled === false) {
      return 0;
    }

    const nodeMap = new Map(nodes.map((n) => [n.nodeIndex, n]));
    const memo = new Map<number, number>();
    const visiting = new Set<number>();

    const longestToOut = (u: number, depth: number): number => {
      if (u === hostOut.nodeIndex) return 0;
      if (depth > nodes.length || visiting.has(u)) return -1;
      const cached = memo.get(u);
      if (cached !== undefined) return cached;
      visiting.add(u);
      const uNode = nodeMap.get(u);
      const selfLat =
        !uNode || uNode.pluginKind === 'HostIn' || uNode.enabled === false
          ? 0
          : uNode.latencySamples;
      let bestBranch = -1;
      for (const e of edges) {
        if (e.fromNode === u && e.gain > 0.0001) {
          const child = longestToOut(e.toNode, depth + 1);
          if (child >= 0) {
            bestBranch = Math.max(bestBranch, child);
          }
        }
      }
      visiting.delete(u);
      const total = bestBranch < 0 ? -1 : selfLat + bestBranch;
      memo.set(u, total);
      return total;
    };

    return Math.max(0, longestToOut(hostIn.nodeIndex, 0));
  });

  const selectedNode = $derived(
    telemetry.patcherNodes.find((n) => n.nodeIndex === telemetry.selectedPatcherNodeIndex) ??
      telemetry.patcherNodes[0]
  );

  const selectedEdge = $derived(
    telemetry.patcherEdges[telemetry.selectedPatcherEdgeIndex] ?? telemetry.patcherEdges[0]
  );

  function clientToSvgCoords(clientX: number, clientY: number): { x: number; y: number } {
    if (!svgEl) return { x: clientX, y: clientY };
    const ctm = svgEl.getScreenCTM?.();
    if (ctm) {
      const pt = new DOMPoint(clientX, clientY).matrixTransform(ctm.inverse());
      return { x: pt.x, y: pt.y };
    }
    const rect = svgEl.getBoundingClientRect();
    const scale = Math.min(rect.width / 880, rect.height / 280) || 1;
    const offsetX = (rect.width - 880 * scale) * 0.5;
    const offsetY = (rect.height - 280 * scale) * 0.5;
    return {
      x: (clientX - rect.left - offsetX) / scale,
      y: (clientY - rect.top - offsetY) / scale,
    };
  }

  function handleNodePointerDown(e: PointerEvent, nodeIndex: number): void {
    e.stopPropagation();
    telemetry.selectedPatcherNodeIndex = nodeIndex;
    const node = telemetry.patcherNodes.find((n) => n.nodeIndex === nodeIndex);
    if (!node) return;
    const pt = clientToSvgCoords(e.clientX, e.clientY);
    draggingNodeIdx = nodeIndex;
    dragMoved = false;
    dragOffset = { dx: pt.x - node.xPx, dy: pt.y - node.yPx };
    svgEl?.setPointerCapture?.(e.pointerId);
  }

  function handleSvgPointerMove(e: PointerEvent): void {
    if (draggingNodeIdx === null) return;
    const pt = clientToSvgCoords(e.clientX, e.clientY);
    dragMoved = true;
    void telemetry.movePatcherNode(
      draggingNodeIdx,
      pt.x - dragOffset.dx,
      pt.y - dragOffset.dy,
      false
    );
  }

  function handleSvgPointerUp(e: PointerEvent): void {
    if (draggingNodeIdx === null) return;
    const finishedIdx = draggingNodeIdx;
    const moved = dragMoved;
    draggingNodeIdx = null;
    dragMoved = false;
    svgEl?.releasePointerCapture?.(e.pointerId);
    if (moved) {
      const node = telemetry.patcherNodes.find((n) => n.nodeIndex === finishedIdx);
      if (node) {
        void telemetry.movePatcherNode(finishedIdx, node.xPx, node.yPx, true);
      }
    }
  }

  async function handleAddCord(): Promise<void> {
    const ok = await telemetry.addPatcherEdge(newCordFrom, newCordTo, 0.707);
    cordStatusMsg = ok
      ? ''
      : '⚠ Không thể nối dây: phát hiện vòng lặp DAG (Cycle) hoặc đầu nối không hợp lệ.';
  }

  async function handleRemoveSelectedCord(): Promise<void> {
    cordStatusMsg = '';
    await telemetry.removePatcherEdge(telemetry.selectedPatcherEdgeIndex);
  }

  function nodeLabelByIndex(idx: number): string {
    return telemetry.patcherNodes.find((n) => n.nodeIndex === idx)?.label ?? `Node #${idx}`;
  }
</script>

<section class="panel patcher-panel">
  <header class="panel-header">
    <div>
      <h2>Modular Patcher Sub-DAG (Sugiyama Layered Layout &amp; Parallel APDC)</h2>
      <span class="subtitle">
        Interactive Node Drag, Bypass &amp; Cord Weights · Critical-Path APDC Latency: {criticalPathPatcherLatency} samples
      </span>
    </div>
    <div class="header-actions">
      <button
        type="button"
        class="layout-btn"
        onclick={() => telemetry.autoLayoutPatcherNodes()}
        title="Tự động sắp xếp lại các Node theo thuật toán phân lớp Sugiyama"
      >
        ⚡ Auto-Layout (Sugiyama)
      </button>
    </div>
  </header>

  <svg
    bind:this={svgEl}
    viewBox="0 0 880 280"
    class="patcher-svg"
    role="application"
    aria-label="Patcher Modular DAG Canvas"
    onpointermove={handleSvgPointerMove}
    onpointerup={handleSvgPointerUp}
    onpointercancel={handleSvgPointerUp}
  >
    {#each telemetry.patcherEdges as edge, edgeIdx}
      {@const src = telemetry.patcherNodes.find((n) => n.nodeIndex === edge.fromNode)}
      {@const dst = telemetry.patcherNodes.find((n) => n.nodeIndex === edge.toNode)}
      {#if src && dst}
        {@const x1 = src.xPx + 145}
        {@const y1 = src.yPx + 26}
        {@const x2 = dst.xPx}
        {@const y2 = dst.yPx + 26}
        {@const cx = (x1 + x2) * 0.5}
        {@const midX = 0.125 * x1 + 0.375 * cx + 0.375 * cx + 0.125 * x2}
        {@const midY = (y1 + y2) * 0.5}
        {@const isSelectedEdge = telemetry.selectedPatcherEdgeIndex === edgeIdx}
        {@const isMutedEdge =
          edge.gain <= 0.0001 ||
          (src.pluginKind === 'HostIn' && src.enabled === false) ||
          ((dst.pluginKind === 'HostOut' || dst.pluginKind === 'Ozone12') && dst.enabled === false)}
        <g
          class="cord-group"
          role="button"
          tabindex="0"
          onclick={() => (telemetry.selectedPatcherEdgeIndex = edgeIdx)}
          onkeydown={(ev) => {
            if (ev.key === 'Enter' || ev.key === ' ') {
              ev.preventDefault();
              telemetry.selectedPatcherEdgeIndex = edgeIdx;
            }
          }}
        >
          <path
            d="M {x1} {y1} C {cx} {y1}, {cx} {y2}, {x2} {y2}"
            fill="none"
            stroke="transparent"
            stroke-width="14"
          />
          <path
            d="M {x1} {y1} C {cx} {y1}, {cx} {y2}, {x2} {y2}"
            fill="none"
            stroke={isSelectedEdge ? '#fbbf24' : isMutedEdge ? 'rgba(247, 246, 229, 0.24)' : '#76D2DB'}
            stroke-width={isSelectedEdge ? 3.4 : Math.max(1.4, Math.min(4.2, 1.4 + edge.gain * 1.3))}
            stroke-dasharray={isMutedEdge ? '5 4' : undefined}
          />
          <g transform="translate({midX - 22}, {midY - 9})">
            <rect
              width="44"
              height="18"
              rx="9"
              fill={isSelectedEdge ? '#2d2410' : '#102124'}
              stroke={isSelectedEdge ? '#fbbf24' : '#76D2DB'}
              stroke-width="1"
            />
            <text
              x="22"
              y="12.5"
              text-anchor="middle"
              fill={isSelectedEdge ? '#fbbf24' : '#F7F6E5'}
              font-size="9.5"
              font-weight="600"
            >
              ×{edge.gain.toFixed(2)}
            </text>
          </g>
        </g>
      {/if}
    {/each}

    {#each telemetry.patcherNodes as node (node.nodeIndex)}
      {@const isSelected = telemetry.selectedPatcherNodeIndex === node.nodeIndex}
      {@const isEnabled = node.enabled !== false}
      <g
        class="node-group"
        transform="translate({node.xPx}, {node.yPx})"
        role="button"
        tabindex="0"
        onpointerdown={(e) => handleNodePointerDown(e, node.nodeIndex)}
        ondblclick={() => telemetry.togglePatcherNodeEnabled(node.nodeIndex)}
        onkeydown={(ev) => {
          if (ev.key === 'Enter' || ev.key === ' ') {
            ev.preventDefault();
            void telemetry.togglePatcherNodeEnabled(node.nodeIndex);
          }
        }}
      >
        <rect
          width="145"
          height="52"
          rx="6"
          fill={isEnabled ? '#1a2a2d' : '#131b1d'}
          opacity={isEnabled ? 1.0 : 0.62}
          stroke={isSelected
            ? '#fbbf24'
            : !isEnabled
              ? 'rgba(247, 246, 229, 0.28)'
              : node.pluginKind === 'FixedLatency'
                ? '#fbbf24'
                : '#76D2DB'}
          stroke-width={isSelected ? '2.4' : '1.5'}
        />
        <circle cx="0" cy="26" r="4" fill="#76D2DB" />
        <circle cx="145" cy="26" r="4" fill="#fbbf24" />
        <text x="9" y="20" fill="#F7F6E5" font-size="10.5" font-weight="600">
          {node.label.length > 18 ? `${node.label.slice(0, 17)}…` : node.label}
        </text>
        <text x="9" y="38" fill="rgba(247, 246, 229, 0.58)" font-size="9.5">
          L{node.layer} · {isEnabled ? node.latencySamples : 0} smp · {node.cpuPct.toFixed(1)}%
        </text>

        <g
          class="node-toggle-pill"
          transform="translate(108, 6)"
          role="button"
          tabindex="0"
          aria-label="Toggle node {node.label}"
          onpointerdown={(e) => e.stopPropagation()}
          onclick={(e) => {
            e.stopPropagation();
            void telemetry.togglePatcherNodeEnabled(node.nodeIndex);
          }}
          onkeydown={(ev) => {
            if (ev.key === 'Enter' || ev.key === ' ') {
              ev.preventDefault();
              ev.stopPropagation();
              void telemetry.togglePatcherNodeEnabled(node.nodeIndex);
            }
          }}
        >
          <rect
            width="31"
            height="15"
            rx="4"
            fill={isEnabled ? 'rgba(118, 210, 219, 0.2)' : 'rgba(239, 68, 68, 0.22)'}
            stroke={isEnabled ? '#76D2DB' : '#ef4444'}
            stroke-width="1"
          />
          <text
            x="15.5"
            y="10.8"
            text-anchor="middle"
            fill={isEnabled ? '#76D2DB' : '#fca5a5'}
            font-size="8.5"
            font-weight="700"
          >
            {isEnabled ? 'ON' : 'BYP'}
          </text>
        </g>
      </g>
    {/each}
  </svg>

  <div class="patcher-inspector">
    {#if selectedNode}
      <div class="inspector-block">
        <span class="inspector-title">
          Node #{selectedNode.nodeIndex}: <strong>{selectedNode.label}</strong>
          <small>({Math.round(selectedNode.xPx)}, {Math.round(selectedNode.yPx)} px)</small>
        </span>
        <button
          type="button"
          class="toggle-node-btn"
          class:bypassed={selectedNode.enabled === false}
          onclick={() => telemetry.togglePatcherNodeEnabled(selectedNode.nodeIndex)}
        >
          {selectedNode.enabled !== false ? '🟢 Đang Bật (ON)' : '⚪ Đang Bypass (0 smp)'}
        </button>
      </div>
    {/if}

    {#if selectedEdge}
      <div class="inspector-block cord-controls">
        <label class="cord-select-label">
          Dây cáp (Cord):
          <select
            value={telemetry.selectedPatcherEdgeIndex}
            onchange={(e) =>
              (telemetry.selectedPatcherEdgeIndex = parseInt(
                (e.currentTarget as HTMLSelectElement).value,
                10
              ))}
          >
            {#each telemetry.patcherEdges as edge, idx}
              <option value={idx}>
                #{edge.fromNode} {nodeLabelByIndex(edge.fromNode)} → #{edge.toNode} {nodeLabelByIndex(edge.toNode)} (×{edge.gain.toFixed(2)})
              </option>
            {/each}
          </select>
        </label>

        <label class="cord-slider-label">
          Trọng số cáp: <strong>×{selectedEdge.gain.toFixed(3)}</strong>
          <input
            type="range"
            min="0"
            max="2"
            step="0.01"
            value={selectedEdge.gain}
            oninput={(e) =>
              telemetry.updatePatcherEdgeGain(
                telemetry.selectedPatcherEdgeIndex,
                parseFloat((e.currentTarget as HTMLInputElement).value)
              )}
          />
        </label>

        <div class="quick-gain-btns">
          <button
            type="button"
            onclick={() => telemetry.updatePatcherEdgeGain(telemetry.selectedPatcherEdgeIndex, 0.0)}
          >
            0.00×
          </button>
          <button
            type="button"
            onclick={() => telemetry.updatePatcherEdgeGain(telemetry.selectedPatcherEdgeIndex, 0.707)}
          >
            0.707×
          </button>
          <button
            type="button"
            onclick={() => telemetry.updatePatcherEdgeGain(telemetry.selectedPatcherEdgeIndex, 1.0)}
          >
            1.00×
          </button>
          <button
            type="button"
            class="danger-cord-btn"
            onclick={handleRemoveSelectedCord}
            title="Xóa dây cáp đang chọn khỏi Patcher DAG"
          >
            ✕ Xóa Dây
          </button>
        </div>
      </div>
    {/if}

    <div class="inspector-block cord-builder">
      <label class="cord-select-label">
        Từ:
        <select
          value={newCordFrom}
          onchange={(e) => (newCordFrom = parseInt((e.currentTarget as HTMLSelectElement).value, 10))}
        >
          {#each telemetry.patcherNodes as n (n.nodeIndex)}
            <option value={n.nodeIndex}>#{n.nodeIndex} {n.label}</option>
          {/each}
        </select>
      </label>
      <label class="cord-select-label">
        Đến:
        <select
          value={newCordTo}
          onchange={(e) => (newCordTo = parseInt((e.currentTarget as HTMLSelectElement).value, 10))}
        >
          {#each telemetry.patcherNodes as n (n.nodeIndex)}
            <option value={n.nodeIndex}>#{n.nodeIndex} {n.label}</option>
          {/each}
        </select>
      </label>
      <button type="button" class="add-cord-btn" onclick={handleAddCord}>
        + Nối Dây
      </button>
      {#if cordStatusMsg}
        <span class="cord-status-err">{cordStatusMsg}</span>
      {/if}
    </div>
  </div>
</section>

<style>
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; }
  .panel-header { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
  h2 { margin: 0; font-size: 0.95rem; color: var(--accent); font-weight: 600; }
  .subtitle { font-size: 0.75rem; color: var(--muted); }
  .header-actions { display: flex; gap: 8px; align-items: center; }
  .layout-btn {
    background: rgba(118, 210, 219, 0.14);
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 0.74rem;
    font-weight: 600;
    cursor: pointer;
  }
  .patcher-svg {
    width: 100%;
    height: 230px;
    background: var(--sub-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-top: 8px;
    touch-action: none;
    user-select: none;
  }
  .node-group { cursor: grab; }
  .node-group:active { cursor: grabbing; }
  .node-toggle-pill { cursor: pointer; }
  .cord-group { cursor: pointer; }
  .patcher-inspector {
    margin-top: 8px;
    padding: 8px 10px;
    background: var(--sub-surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 0.75rem;
  }
  .inspector-block { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .inspector-title small { color: var(--muted); margin-left: 4px; }
  .toggle-node-btn {
    background: rgba(118, 210, 219, 0.16);
    color: var(--text);
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.73rem;
    font-weight: 600;
    cursor: pointer;
  }
  .toggle-node-btn.bypassed {
    background: rgba(239, 68, 68, 0.16);
    border-color: #ef4444;
  }
  .cord-select-label, .cord-slider-label { display: flex; align-items: center; gap: 6px; color: var(--muted); }
  .cord-select-label select {
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 6px;
    font-size: 0.72rem;
    max-width: 240px;
  }
  .cord-slider-label strong { color: var(--accent); min-width: 44px; }
  .quick-gain-btns { display: flex; gap: 4px; }
  .quick-gain-btns button, .add-cord-btn {
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 3px 7px;
    font-size: 0.7rem;
    cursor: pointer;
  }
  .add-cord-btn {
    border-color: var(--accent);
    color: var(--accent);
    font-weight: 600;
  }
  .quick-gain-btns button.danger-cord-btn {
    border-color: rgba(224, 112, 112, 0.55);
    color: var(--err);
  }
  .cord-status-err {
    color: var(--err);
    font-size: 0.71rem;
    font-weight: 600;
  }
</style>
