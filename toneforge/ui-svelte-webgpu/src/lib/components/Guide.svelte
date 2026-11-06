<script lang="ts">
  import { GUIDES, type WorkspaceTabKey } from '../state/guides';

  let {
    id,
    openByDefault = false,
  }: {
    id: string;
    openByDefault?: boolean;
  } = $props();

  const g = $derived(GUIDES[id]);

  function navigateToTab(tab: WorkspaceTabKey, sectionId?: string) {
    if (typeof window !== 'undefined') {
      window.dispatchEvent(
        new CustomEvent('toneforge:navigate', {
          detail: { tab, sectionId },
        })
      );
    }
  }
</script>

{#if g}
  <details class="guide" open={openByDefault}>
    <summary>
      <b>?</b>
      <span>Hướng dẫn sử dụng nhanh — {g.title}</span>
    </summary>
    <div class="body">
      <p class="what">{g.what}</p>

      {#if g.steps && g.steps.length > 0}
        <ol>
          {#each g.steps as step, i (i)}
            <li>{step}</li>
          {/each}
        </ol>
      {/if}

      {#if g.gotcha}
        <p class="gotcha"><b>Dễ sai:</b> {g.gotcha}</p>
      {/if}

      <div class="next-actions">
        {#if g.next}
          <button
            type="button"
            class="next-btn"
            onclick={() => navigateToTab(g.next.tab, g.next.tab === 'handbook' ? g.handbookAnchor?.sectionId : undefined)}
          >
            {g.next.label} →
          </button>
        {/if}

        {#if g.handbookAnchor && g.next?.tab !== 'handbook'}
          <button
            type="button"
            class="handbook-btn"
            onclick={() => navigateToTab('handbook', g.handbookAnchor.sectionId)}
          >
            📖 {g.handbookAnchor.label} →
          </button>
        {/if}
      </div>
    </div>
  </details>
{/if}

<style>
  .guide {
    margin: 0 0 10px 0;
  }

  summary {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    width: auto;
    padding: 0.26rem 0.78rem;
    border: 1px solid color-mix(in srgb, var(--accent) 45%, var(--border));
    border-radius: 999px;
    background: var(--surface-2);
    color: var(--accent);
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    user-select: none;
    list-style: none;
    transition: all 0.15s ease;
  }

  summary::-webkit-details-marker {
    display: none;
  }

  summary::marker {
    content: '';
  }

  summary:hover {
    background: var(--sidebar-bg);
    border-color: var(--accent);
  }

  summary b {
    display: inline-grid;
    place-items: center;
    width: 1.1rem;
    height: 1.1rem;
    border-radius: 999px;
    background: var(--accent);
    color: #0b2b1e;
    font-size: 0.72rem;
    font-weight: 800;
    flex-shrink: 0;
  }

  .guide[open] summary {
    margin-bottom: 0.45rem;
    background: var(--sidebar-bg);
    border-color: var(--accent);
  }

  .body {
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    background: var(--sub-surface-2);
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    line-height: 1.58;
  }

  .body p {
    margin: 0 0 0.55rem;
  }

  .body p:last-child {
    margin-bottom: 0;
  }

  .what {
    color: var(--text);
    font-weight: 500;
  }

  .body ol {
    margin: 0 0 0.6rem;
    padding-left: 1.25rem;
    color: rgba(247, 246, 229, 0.82);
  }

  .body li {
    margin-bottom: 0.28rem;
  }

  .gotcha {
    color: rgba(247, 246, 229, 0.78);
    border-top: 1px solid var(--border);
    padding-top: 0.5rem;
  }

  .gotcha b {
    color: var(--warn);
  }

  .next-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-top: 0.55rem;
    padding-top: 0.45rem;
    border-top: 1px dashed var(--border);
  }

  .next-btn {
    background: var(--accent);
    color: #0b2b1e;
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 5px 11px;
    font-size: 0.76rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .next-btn:hover {
    filter: brightness(1.08);
  }

  .handbook-btn {
    background: var(--surface-2);
    color: var(--accent);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 11px;
    font-size: 0.75rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .handbook-btn:hover {
    border-color: var(--accent);
    color: var(--text);
  }
</style>
