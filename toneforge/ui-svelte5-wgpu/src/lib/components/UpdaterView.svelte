<script lang="ts">
  import type { Update } from '@tauri-apps/plugin-updater';
  import { onMount } from 'svelte';

  let { isRecording = false, isDesktopTauri = false }: { isRecording?: boolean; isDesktopTauri?: boolean } = $props();

  const RELEASES_URL = 'https://github.com/Selorson/ToneForge-Release/releases';

  let version = $state('0.1.1');
  let checking = $state(false);
  let downloading = $state(false);
  let hasUpdate = $state(false);
  let nextVersion = $state('');
  let nextDate = $state<string | null>(null);
  let nextBody = $state<string | null>(null);
  let message = $state('');
  let isError = $state(false);
  let downloadedBytes = $state(0);
  let totalBytes = $state(0);

  let updateHandle: Update | null = null;

  const progressPct = $derived(
    totalBytes > 0 ? Math.min(100, Math.round((downloadedBytes / totalBytes) * 100)) : 0
  );

  onMount(async () => {
    if (isDesktopTauri) {
      try {
        const { getVersion } = await import('@tauri-apps/api/app');
        version = await getVersion();
      } catch {
        // giữ mặc định 0.1.1
      }
    }
  });

  function formatUpdateError(raw: unknown): { text: string; err: boolean } {
    const s = raw instanceof Error ? raw.message : String(raw);
    if (/404|not found|valid release json|release.*not exist/i.test(s)) {
      return {
        text: `Chưa có bản phát hành mới hơn trên kênh Selorson/ToneForge-Release — đang dùng bản v${version} mới nhất.`,
        err: false
      };
    }
    return {
      text: `Không kiểm tra được bản cập nhật: ${s}`,
      err: true
    };
  }

  async function checkUpdateNow() {
    if (!isDesktopTauri) {
      isError = false;
      message = 'Tính năng tự cập nhật chỉ hoạt động trong cửa sổ ứng dụng desktop (.exe).';
      return;
    }
    checking = true;
    isError = false;
    message = 'Đang kết nối kênh phát hành Selorson/ToneForge-Release…';
    updateHandle = null;
    hasUpdate = false;

    try {
      const { check } = await import('@tauri-apps/plugin-updater');
      const upd = await check();
      if (upd) {
        updateHandle = upd;
        nextVersion = upd.version.replace(/^v/i, '');
        nextDate = upd.date ?? null;
        nextBody = upd.body ?? null;
        hasUpdate = true;
        message = `Tìm thấy phiên bản mới v${nextVersion}!`;
      } else {
        message = `ToneForge Studio đang ở phiên bản mới nhất (v${version}).`;
      }
    } catch (e) {
      const formatted = formatUpdateError(e);
      isError = formatted.err;
      message = formatted.text;
    } finally {
      checking = false;
    }
  }

  async function downloadAndInstallUpdate() {
    if (!updateHandle) return;
    if (isRecording) {
      isError = true;
      message = 'Đang thu âm trực tiếp xuống đĩa — vui lòng dừng thu âm trước khi cập nhật ứng dụng.';
      return;
    }

    downloading = true;
    isError = false;
    downloadedBytes = 0;
    totalBytes = 0;
    message = 'Đang chuẩn bị tải bản cập nhật…';

    try {
      await updateHandle.download((ev) => {
        if (ev.event === 'Started') {
          totalBytes = ev.data.contentLength ?? 0;
          message = 'Bắt đầu tải bộ cài cập nhật…';
        } else if (ev.event === 'Progress') {
          downloadedBytes += ev.data.chunkLength;
          const doneMb = (downloadedBytes / (1024 * 1024)).toFixed(1);
          if (totalBytes > 0) {
            const totMb = (totalBytes / (1024 * 1024)).toFixed(1);
            const pct = Math.min(100, Math.round((downloadedBytes / totalBytes) * 100));
            message = `Đang tải: ${pct}% (${doneMb}/${totMb} MB)`;
          } else {
            message = `Đã tải: ${doneMb} MB`;
          }
        } else if (ev.event === 'Finished') {
          message = 'Đã tải xong & xác thực chữ ký số Ed25519! Đang cài đặt…';
        }
      });

      await updateHandle.install();
      message = 'Cài đặt hoàn tất! Ứng dụng sẽ tự khởi động lại…';
      const { relaunch } = await import('@tauri-apps/plugin-process');
      setTimeout(() => {
        void relaunch();
      }, 1500);
    } catch (e) {
      isError = true;
      downloading = false;
      message = `Lỗi tải hoặc cài đặt bản cập nhật: ${e instanceof Error ? e.message : String(e)}`;
    }
  }

  async function openReleasesPage() {
    if (!isDesktopTauri) {
      window.open(RELEASES_URL, '_blank', 'noopener');
      return;
    }
    try {
      const { openUrl } = await import('@tauri-apps/plugin-opener');
      await openUrl(RELEASES_URL);
    } catch {
      window.open(RELEASES_URL, '_blank', 'noopener');
    }
  }
</script>

<section class="upd-panel">
  <div class="upd-card upd-header">
    <div>
      <div class="upd-title">
        <strong>ToneForge Studio</strong>
        <span class="upd-badge">v{version}</span>
      </div>
      <p class="upd-desc">
        120 FPS WebGPU Mastering, Pro-Q 4, Soothe3, Ozone 12 &amp; UVR5 Stem Workstation · Hệ sinh thái Selorson
      </p>
    </div>
    <button class="upd-btn" onclick={openReleasesPage}>↗ Trang phát hành GitHub</button>
  </div>

  <div class="upd-card">
    <h3>Tự cập nhật qua kênh phát hành chính thức</h3>
    <p class="upd-desc">
      Kênh công khai: <code>Selorson/ToneForge-Release</code> (xác thực chữ ký số Ed25519).
      Các trọng số AI lớn (<code>UVR5</code>, <code>gpu_runtime</code>) và dự án âm thanh nằm tách biệt bên ngoài nên không bị ghi đè khi nâng cấp.
    </p>

    <div class="upd-row">
      <button class="upd-btn primary" onclick={checkUpdateNow} disabled={checking || downloading}>
        {checking ? '↻ Đang kiểm tra…' : '↻ Kiểm tra phiên bản mới'}
      </button>
    </div>

    {#if message}
      <div class="upd-msg {isError ? 'err' : hasUpdate ? 'ok' : ''}">
        {message}
      </div>
    {/if}

    {#if downloading}
      <div class="upd-progress">
        <div class="upd-progress-bar" style="width: {totalBytes > 0 ? progressPct : 100}%"></div>
      </div>
    {/if}

    {#if hasUpdate && !downloading}
      <div class="upd-ready">
        <div class="upd-ready-top">
          <strong>Bản cập nhật v{nextVersion} sẵn sàng cài đặt</strong>
          {#if nextDate}
            <code>{nextDate}</code>
          {/if}
        </div>

        {#if nextBody}
          <div class="upd-notes">{nextBody}</div>
        {/if}

        {#if isRecording}
          <p class="upd-warn">Đang thu âm trực tiếp — hãy dừng REC trước khi cài đặt bản mới.</p>
        {/if}

        <button class="upd-btn primary" onclick={downloadAndInstallUpdate} disabled={isRecording}>
          ⬇ Tải và cài đặt ngay
        </button>
      </div>
    {/if}
  </div>
</section>

<style>
  .upd-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
    max-width: 920px;
  }
  .upd-card {
    background: var(--surface, #152224);
    border: 1px solid var(--border, rgba(118, 210, 219, 0.2));
    border-radius: 10px;
    padding: 16px 18px;
  }
  .upd-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }
  .upd-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.08rem;
  }
  .upd-badge {
    font-family: monospace;
    font-size: 0.78rem;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid rgba(74, 222, 128, 0.45);
    color: #4ade80;
    background: rgba(74, 222, 128, 0.1);
  }
  h3 {
    margin: 0 0 6px;
    font-size: 0.98rem;
  }
  .upd-desc {
    margin: 4px 0 0;
    color: var(--muted, rgba(247, 246, 229, 0.65));
    font-size: 0.86rem;
    line-height: 1.5;
  }
  .upd-row {
    margin-top: 12px;
  }
  .upd-btn {
    padding: 7px 14px;
    border-radius: 8px;
    border: 1px solid var(--border, rgba(118, 210, 219, 0.25));
    background: #1a2a2d;
    color: #f7f6e5;
    cursor: pointer;
    font-weight: 500;
  }
  .upd-btn.primary {
    background: #76d2db;
    border-color: #76d2db;
    color: #0b2b1e;
    font-weight: 700;
  }
  .upd-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .upd-msg {
    margin-top: 12px;
    padding: 10px 12px;
    border-radius: 8px;
    background: #1a2a2d;
    border: 1px solid rgba(118, 210, 219, 0.2);
    border-left: 3px solid #76d2db;
    font-size: 0.86rem;
  }
  .upd-msg.ok {
    border-left-color: #4ade80;
    color: #4ade80;
  }
  .upd-msg.err {
    border-left-color: #e07070;
    color: #e07070;
  }
  .upd-progress {
    margin-top: 10px;
    height: 8px;
    border-radius: 999px;
    background: #0e1b1d;
    overflow: hidden;
  }
  .upd-progress-bar {
    height: 100%;
    background: #4ade80;
    transition: width 0.2s;
  }
  .upd-ready {
    margin-top: 12px;
    padding: 12px 14px;
    border-radius: 8px;
    background: #1a2a2d;
    border: 1px solid rgba(74, 222, 128, 0.45);
    display: flex;
    flex-direction: column;
    gap: 10px;
    align-items: flex-start;
  }
  .upd-ready-top {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #4ade80;
  }
  .upd-notes {
    width: 100%;
    max-height: 150px;
    overflow-y: auto;
    padding: 10px;
    border-radius: 6px;
    background: #0e1b1d;
    white-space: pre-wrap;
    font-size: 0.84rem;
  }
  .upd-warn {
    margin: 0;
    color: #fbbf24;
    font-size: 0.82rem;
  }
</style>
