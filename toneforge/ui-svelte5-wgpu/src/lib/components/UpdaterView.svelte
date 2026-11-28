<script lang="ts">
  import type { Update } from '@tauri-apps/plugin-updater';
  import { onMount } from 'svelte';
  import Guide from './Guide.svelte';

  interface AddonPackInfo {
    id: string;
    title: string;
    subtitle: string;
    targetAudience: 'standard_pc' | 'nvidia_gpu' | 'all_pcs' | string;
    audienceBadge: string;
    recommended: boolean;
    estimatedSizeLabel: string;
    installed: boolean;
    installedSource: 'not_installed' | 'addon_pack' | 'bundled_local' | string;
    installedBytes: number;
    canUninstall: boolean;
    statusDetail: string;
  }

  interface AddonSystemOverview {
    addonsRootPath: string;
    hasNvidiaGpu: boolean;
    nvidiaDeviceName: string | null;
    nvidiaVramMib: number;
    activeTier: string;
    activeBadge: string;
    activeProvider: string;
    restartRecommended: boolean;
    restartReason: string | null;
    downloadInProgress: boolean;
    packs: AddonPackInfo[];
  }

  interface AddonDownloadProgress {
    packId: string;
    phase: 'starting' | 'downloading' | 'extracting' | 'verifying' | 'completed' | 'cancelled' | 'error' | string;
    stepCurrent: number;
    stepTotal: number;
    downloadedBytes: number;
    totalBytes: number;
    percent: number;
    message: string;
  }

  let {
    isRecording = false,
    isDesktopTauri = false,
    onPacksChanged = () => {}
  }: {
    isRecording?: boolean;
    isDesktopTauri?: boolean;
    onPacksChanged?: () => void;
  } = $props();

  let version = $state('0.1.3');
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

  let overview = $state<AddonSystemOverview | null>(null);
  let inspectingPacks = $state(false);
  let activePackId = $state<string | null>(null);
  let packProgress = $state<AddonDownloadProgress | null>(null);
  let packNotice = $state('');
  let packNoticeErr = $state(false);
  let confirmUninstallId = $state<string | null>(null);

  let updateHandle: Update | null = null;

  const progressPct = $derived(
    totalBytes > 0 ? Math.min(100, Math.round((downloadedBytes / totalBytes) * 100)) : 0
  );

  function sanitizeText(raw: unknown): string {
    const s = raw instanceof Error ? raw.message : String(raw);
    return s
      .replace(/https?:\/\/\S+/gi, '[máy-chủ-cập-nhật]')
      .replace(/github\.com\S*/gi, '[máy-chủ-cập-nhật]')
      .replace(/selorson\/\S*/gi, 'kênh chính thức');
  }

  async function refreshAddonPacks() {
    if (!isDesktopTauri) return;
    inspectingPacks = true;
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const res = await invoke<AddonSystemOverview>('cmd_inspect_addon_packs');
      overview = res;
    } catch (e) {
      packNoticeErr = true;
      packNotice = `Không kiểm tra được danh sách gói mở rộng: ${sanitizeText(e)}`;
    } finally {
      inspectingPacks = false;
    }
  }

  onMount(() => {
    let unlistenFn: (() => void) | null = null;

    void (async () => {
      if (!isDesktopTauri) return;
      try {
        const { getVersion } = await import('@tauri-apps/api/app');
        version = await getVersion();
      } catch {
        // giữ mặc định
      }

      try {
        const { listen } = await import('@tauri-apps/api/event');
        unlistenFn = await listen<AddonDownloadProgress>('addon-pack-progress', (ev) => {
          packProgress = ev.payload;
          if (ev.payload.phase === 'downloading' || ev.payload.phase === 'extracting' || ev.payload.phase === 'starting') {
            activePackId = ev.payload.packId;
          }
        });
      } catch {
        // bỏ qua nếu chạy chế độ trình duyệt
      }

      await refreshAddonPacks();
    })();

    return () => {
      unlistenFn?.();
    };
  });

  async function installPack(packId: string) {
    if (!isDesktopTauri) {
      packNoticeErr = true;
      packNotice = 'Chức năng tải gói bổ sung chỉ hoạt động trên ứng dụng Desktop (.exe).';
      return;
    }
    if (isRecording) {
      packNoticeErr = true;
      packNotice = 'Đang thu âm trực tiếp — vui lòng dừng thu âm (REC) trước khi tải gói lớn.';
      return;
    }

    confirmUninstallId = null;
    activePackId = packId;
    packNotice = '';
    packNoticeErr = false;
    packProgress = {
      packId,
      phase: 'starting',
      stepCurrent: 0,
      stepTotal: 1,
      downloadedBytes: 0,
      totalBytes: 0,
      percent: 0,
      message: 'Đang kết nối máy chủ tải gói bổ sung…'
    };

    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const updated = await invoke<AddonSystemOverview>('cmd_install_addon_pack', { packId });
      overview = updated;
      packNoticeErr = false;
      packNotice = 'Đã tải, kiểm tra toàn vẹn và cài đặt gói thành công!';
      onPacksChanged();
    } catch (e) {
      const msg = sanitizeText(e);
      packNoticeErr = !msg.includes('Đã hủy tải');
      packNotice = msg;
      await refreshAddonPacks();
    } finally {
      activePackId = null;
    }
  }

  async function cancelPackDownload() {
    if (!isDesktopTauri) return;
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke('cmd_cancel_addon_download');
    } catch {
      // ignore
    }
  }

  async function uninstallPack(packId: string) {
    if (!isDesktopTauri) return;
    confirmUninstallId = null;
    packNotice = '';
    packNoticeErr = false;
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const updated = await invoke<AddonSystemOverview>('cmd_uninstall_addon_pack', { packId });
      overview = updated;
      packNoticeErr = false;
      packNotice = 'Đã gỡ gói bổ sung khỏi thư mục người dùng.';
      onPacksChanged();
    } catch (e) {
      packNoticeErr = true;
      packNotice = `Không gỡ được gói: ${sanitizeText(e)}`;
    }
  }

  async function openAddonsFolder() {
    if (!isDesktopTauri) return;
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke('cmd_open_addons_folder');
    } catch (e) {
      packNoticeErr = true;
      packNotice = sanitizeText(e);
    }
  }

  async function restartAppNow() {
    if (!isDesktopTauri) return;
    try {
      const { relaunch } = await import('@tauri-apps/plugin-process');
      await relaunch();
    } catch (e) {
      packNoticeErr = true;
      packNotice = `Không thể tự khởi động lại: ${sanitizeText(e)}`;
    }
  }

  function formatUpdateError(raw: unknown): { text: string; err: boolean } {
    const s = sanitizeText(raw);
    if (/404|not found|valid release json|release.*not exist/i.test(s)) {
      return {
        text: `Chưa có bản phát hành mới hơn — ứng dụng đang ở phiên bản v${version} mới nhất.`,
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
    message = 'Đang kết nối máy chủ cập nhật chính thức…';
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
          message = 'Bắt đầu tải gói cập nhật ứng dụng…';
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
      message = `Lỗi tải hoặc cài đặt bản cập nhật: ${sanitizeText(e)}`;
    }
  }
</script>

<section class="upd-panel">
  <Guide id="update" />

  <!-- 1. GIỚI THIỆU (ABOUT) & TRẠNG THÁI PHẦN CỨNG -->
  <div class="upd-card upd-header">
    <div class="about-main">
      <div class="upd-title">
        <strong>ToneForge Studio</strong>
        <span class="upd-badge">v{version}</span>
        {#if overview}
          <span class="tier-pill">{overview.activeBadge}</span>
        {/if}
      </div>
      <p class="upd-desc">
        Trạm làm nhạc, Thu âm, Chỉnh phô giọng (NewTone), Pro-Q 4, Soothe3, Ozone 12 &amp; Tách nhạc AI UVR5 120 FPS WebGPU.
        Bộ cài gốc được tối giản dung lượng — các nhân tăng tốc AI và mô hình lớn được tách thành gói phụ tải theo nhu cầu máy.
      </p>

      {#if overview}
        <div class="hw-detect-bar">
          {#if overview.hasNvidiaGpu}
            <span class="hw-chip nvidia">
              🚀 Phát hiện GPU rời NVIDIA: <strong>{overview.nvidiaDeviceName}</strong>
              {#if overview.nvidiaVramMib > 0}
                ({Math.max(1, Math.round(overview.nvidiaVramMib / 1024))} GB VRAM)
              {/if}
            </span>
          {:else}
            <span class="hw-chip standard">
              💻 Phát hiện cấu hình: <strong>Máy thường / Không có card rời NVIDIA</strong> (Tương thích DirectML DX12 &amp; CPU đa nhân)
            </span>
          {/if}
        </div>
      {/if}
    </div>

    {#if isDesktopTauri}
      <div class="header-actions">
        <button class="upd-btn" onclick={refreshAddonPacks} disabled={inspectingPacks || activePackId !== null}>
          {inspectingPacks ? '↻ Đang quét…' : '↻ Kiểm tra gói cài'}
        </button>
        <button class="upd-btn" onclick={openAddonsFolder} title="Mở thư mục lưu các gói tải thêm trên máy tính">
          📂 Thư mục gói phụ
        </button>
      </div>
    {/if}
  </div>

  {#if overview?.restartRecommended && overview.restartReason}
    <div class="upd-card restart-banner">
      <div>
        <strong>⚡ Sẵn sàng chuyển sang nhân tăng tốc mới</strong>
        <p class="upd-desc">{overview.restartReason}</p>
      </div>
      <button class="upd-btn primary" onclick={restartAppNow} disabled={isRecording}>
        ↻ Khởi động lại ngay
      </button>
    </div>
  {/if}

  <!-- 2. HƯỚNG DẪN CHỌN GÓI THEO LOẠI MÁY -->
  <div class="upd-card">
    <h3>Hướng dẫn chọn gói tải thêm theo cấu hình máy</h3>
    <p class="upd-desc">
      Các tính năng DAW, Thu âm, Cắt ghép, Preset mẫu, EQ Pro-Q 4, Soothe3 và Mastering chỉnh tay <strong>đã hoạt động ngay sau khi cài app</strong>.
      Khi cần dùng <strong>AI Tách nhạc (UVR5)</strong> hoặc <strong>AI Mastering tự động</strong>, hãy chọn đúng tổ hợp gói cho máy của bạn:
    </p>

    <div class="guide-grid">
      <div class="guide-box {overview && !overview.hasNvidiaGpu ? 'matched' : ''}">
        <div class="guide-box-head">
          <strong>💻 Máy thường / Laptop văn phòng (Không có NVIDIA)</strong>
          {#if overview && !overview.hasNvidiaGpu}
            <span class="match-tag">Phù hợp với máy này</span>
          {/if}
        </div>
        <p class="guide-sub">Dành cho máy dùng card onboard Intel (UHD / Iris Xe), AMD Radeon hoặc chạy thuần CPU:</p>
        <ul class="guide-list">
          <li>Tải <strong>Gói 1 (Nhân AI Máy thường — DirectML)</strong> <code>~17 MB</code>.</li>
          <li>Tải <strong>Gói 3 (Mô hình Tách nhạc UVR5)</strong> <code>~128 MB</code> để tách Vocal / Beat chuẩn phòng thu.</li>
          <li><em>Tùy chọn:</em> Tải thêm <strong>Gói 4 (Ozone 12 AI)</strong> nếu muốn dùng trợ lý tự động cân chỉnh nhạc.</li>
          <li class="skip-note">✕ <strong>Không cần</strong> tải Gói 2 NVIDIA (tiết kiệm <code>~2.5 GB</code> ổ cứng).</li>
        </ul>
      </div>

      <div class="guide-box {overview?.hasNvidiaGpu ? 'matched' : ''}">
        <div class="guide-box-head">
          <strong>🚀 Máy có Card đồ họa rời NVIDIA (GTX / RTX)</strong>
          {#if overview?.hasNvidiaGpu}
            <span class="match-tag">Phù hợp với máy này</span>
          {/if}
        </div>
        <p class="guide-sub">Dành cho PC / Laptop Gaming hoặc Đồ họa có card rời NVIDIA:</p>
        <ul class="guide-list">
          <li>Tải <strong>Gói 2 (Nhân Tăng tốc NVIDIA CUDA 12 + cuDNN 9)</strong> <code>~1.1 GB</code> để chạy bằng nhân Tensor Cores.</li>
          <li>Tải <strong>Gói 3 (Mô hình Tách nhạc UVR5)</strong> <code>~128 MB</code> để tách Vocal / Beat tốc độ cao (vài giây/bài).</li>
          <li><em>Tùy chọn:</em> Tải thêm <strong>Gói 4 (Ozone 12 AI)</strong> cho Master Assistant &amp; 4-Stem trực tiếp.</li>
          <li class="skip-note">✓ Có thể bỏ qua Gói 1 (hoặc giữ làm nhân dự phòng đều được).</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- 3. DANH SÁCH KIỂM TRA & CÀI ĐẶT GÓI PHỤ (ADD-ON PACKS) -->
  <div class="upd-card">
    <div class="section-head">
      <div>
        <h3>Quản lý Gói cài đặt bổ sung (Add-on Packs)</h3>
        <p class="upd-desc">
          Các gói tải thêm được lưu độc lập trong thư mục người dùng nên <strong>không bị xóa hay phải tải lại</strong> mỗi khi cập nhật phiên bản mới.
        </p>
      </div>
    </div>

    {#if packNotice}
      <div class="upd-msg {packNoticeErr ? 'err' : 'ok'}">
        {packNotice}
      </div>
    {/if}

    {#if !isDesktopTauri}
      <div class="upd-msg">
        Đang mở ở chế độ trình duyệt xem trước — hãy chạy ứng dụng Desktop (<code>ToneForge_Studio.exe</code>) để kiểm tra và cài đặt gói phụ.
      </div>
    {:else if !overview}
      <div class="upd-msg">Đang kiểm tra trạng thái các gói bổ sung trên ổ cứng…</div>
    {:else}
      <div class="pack-list">
        {#each overview.packs as pack (pack.id)}
          {@const isThisDownloading = activePackId === pack.id}
          {@const otherDownloading = activePackId !== null && activePackId !== pack.id}
          <div class="pack-item {pack.installed ? 'installed' : ''} {pack.recommended && !pack.installed ? 'recommended' : ''}">
            <div class="pack-top">
              <div class="pack-title-wrap">
                <strong class="pack-title">{pack.title}</strong>
                <span class="pack-audience {pack.recommended ? 'rec' : 'optional'}">{pack.audienceBadge}</span>
                <span class="pack-size">{pack.estimatedSizeLabel}</span>
              </div>

              <div class="pack-status-badges">
                {#if pack.installed}
                  <span class="state-pill ok">
                    ✓ {pack.installedSource === 'bundled_local' ? 'Đã có sẵn' : 'Đã cài đặt'}
                  </span>
                {:else}
                  <span class="state-pill missing">Chưa cài đặt</span>
                {/if}
              </div>
            </div>

            <p class="pack-sub">{pack.subtitle}</p>
            <div class="pack-detail-row">
              <span class="pack-detail-text">{pack.statusDetail}</span>

              <div class="pack-actions">
                {#if isThisDownloading}
                  <button class="upd-btn danger" onclick={cancelPackDownload}>
                    ✕ Hủy tải
                  </button>
                {:else if pack.installed}
                  {#if pack.canUninstall}
                    {#if confirmUninstallId === pack.id}
                      <span class="confirm-txt">Xác nhận gỡ?</span>
                      <button class="upd-btn danger" onclick={() => uninstallPack(pack.id)} disabled={otherDownloading}>
                        Gỡ ngay
                      </button>
                      <button class="upd-btn" onclick={() => (confirmUninstallId = null)}>
                        Hủy
                      </button>
                    {:else}
                      <button
                        class="upd-btn"
                        onclick={() => (confirmUninstallId = pack.id)}
                        disabled={otherDownloading}
                        title="Xóa gói này khỏi ổ cứng để giải phóng dung lượng"
                      >
                        🗑 Gỡ gói
                      </button>
                    {/if}
                  {/if}
                {:else}
                  <button
                    class="upd-btn {pack.recommended ? 'primary' : ''}"
                    onclick={() => installPack(pack.id)}
                    disabled={otherDownloading || isRecording}
                  >
                    ⬇ {pack.recommended ? 'Tải & Cài đặt' : 'Tải gói này'}
                  </button>
                {/if}
              </div>
            </div>

            {#if isThisDownloading && packProgress && packProgress.packId === pack.id}
              <div class="pack-progress-box">
                <div class="pack-progress-msg">{packProgress.message}</div>
                <div class="upd-progress">
                  <div
                    class="upd-progress-bar"
                    style="width: {packProgress.percent > 0 ? packProgress.percent : 8}%"
                  ></div>
                </div>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- 4. CẬP NHẬT ỨNG DỤNG CHÍNH THỨC (ẨN HOÀN TOÀN THÔNG TIN GIT) -->
  <div class="upd-card">
    <h3>Cập nhật phiên bản ứng dụng</h3>
    <p class="upd-desc">
      Tự động kiểm tra và nâng cấp nhân ứng dụng qua máy chủ phát hành chính thức (xác thực chữ ký số an toàn Ed25519).
    </p>

    <div class="upd-row">
      <button class="upd-btn primary" onclick={checkUpdateNow} disabled={checking || downloading || activePackId !== null}>
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
    gap: 14px;
    max-width: 1080px;
  }
  .upd-card {
    background: var(--surface, #152224);
    border: 1px solid var(--border, rgba(118, 210, 219, 0.2));
    border-radius: 10px;
    padding: 16px 18px;
  }
  .upd-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 14px;
    flex-wrap: wrap;
  }
  .about-main {
    flex: 1 1 540px;
  }
  .header-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .upd-title {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
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
  .tier-pill {
    font-size: 0.76rem;
    padding: 2px 9px;
    border-radius: 999px;
    border: 1px solid rgba(118, 210, 219, 0.4);
    color: #76d2db;
    background: rgba(118, 210, 219, 0.1);
  }
  .hw-detect-bar {
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .hw-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    padding: 5px 11px;
    border-radius: 8px;
    background: #0e1b1d;
    border: 1px solid rgba(118, 210, 219, 0.25);
  }
  .hw-chip.nvidia {
    border-color: rgba(74, 222, 128, 0.45);
    color: #d9f99d;
  }
  .hw-chip.standard {
    border-color: rgba(118, 210, 219, 0.4);
    color: #e0f2fe;
  }
  .restart-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    border-color: rgba(251, 191, 36, 0.55);
    background: rgba(251, 191, 36, 0.08);
  }
  h3 {
    margin: 0 0 6px;
    font-size: 0.98rem;
  }
  .upd-desc {
    margin: 4px 0 0;
    color: var(--muted, rgba(247, 246, 229, 0.7));
    font-size: 0.85rem;
    line-height: 1.52;
  }
  .guide-grid {
    margin-top: 12px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 12px;
  }
  .guide-box {
    background: #0e1b1d;
    border: 1px solid rgba(118, 210, 219, 0.18);
    border-radius: 9px;
    padding: 13px 15px;
  }
  .guide-box.matched {
    border-color: rgba(74, 222, 128, 0.55);
    box-shadow: inset 0 0 0 1px rgba(74, 222, 128, 0.18);
  }
  .guide-box-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 0.9rem;
  }
  .match-tag {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(74, 222, 128, 0.16);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.45);
  }
  .guide-sub {
    margin: 6px 0 8px;
    font-size: 0.8rem;
    color: rgba(247, 246, 229, 0.65);
  }
  .guide-list {
    margin: 0;
    padding-left: 18px;
    font-size: 0.82rem;
    line-height: 1.58;
    color: rgba(247, 246, 229, 0.88);
  }
  .guide-list code {
    font-size: 0.77rem;
    padding: 1px 5px;
    border-radius: 4px;
    background: rgba(118, 210, 219, 0.12);
    color: #76d2db;
  }
  .skip-note {
    color: rgba(247, 246, 229, 0.68);
  }
  .pack-list {
    margin-top: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .pack-item {
    background: #0e1b1d;
    border: 1px solid rgba(118, 210, 219, 0.18);
    border-radius: 9px;
    padding: 13px 15px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .pack-item.recommended {
    border-color: rgba(118, 210, 219, 0.42);
  }
  .pack-item.installed {
    border-color: rgba(74, 222, 128, 0.4);
  }
  .pack-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
  }
  .pack-title-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .pack-title {
    font-size: 0.92rem;
  }
  .pack-audience {
    font-size: 0.73rem;
    padding: 2px 8px;
    border-radius: 999px;
    font-weight: 600;
  }
  .pack-audience.rec {
    background: rgba(118, 210, 219, 0.15);
    color: #76d2db;
    border: 1px solid rgba(118, 210, 219, 0.35);
  }
  .pack-audience.optional {
    background: rgba(247, 246, 229, 0.07);
    color: rgba(247, 246, 229, 0.65);
    border: 1px solid rgba(247, 246, 229, 0.16);
  }
  .pack-size {
    font-family: monospace;
    font-size: 0.76rem;
    color: rgba(247, 246, 229, 0.6);
  }
  .state-pill {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 999px;
  }
  .state-pill.ok {
    background: rgba(74, 222, 128, 0.14);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.42);
  }
  .state-pill.missing {
    background: rgba(251, 191, 36, 0.1);
    color: #fbbf24;
    border: 1px solid rgba(251, 191, 36, 0.32);
  }
  .pack-sub {
    margin: 0;
    font-size: 0.82rem;
    line-height: 1.48;
    color: rgba(247, 246, 229, 0.72);
  }
  .pack-detail-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 2px;
  }
  .pack-detail-text {
    font-size: 0.8rem;
    color: rgba(118, 210, 219, 0.9);
  }
  .pack-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .confirm-txt {
    font-size: 0.8rem;
    color: #f87171;
  }
  .pack-progress-box {
    margin-top: 4px;
    padding-top: 8px;
    border-top: 1px solid rgba(118, 210, 219, 0.14);
  }
  .pack-progress-msg {
    font-size: 0.8rem;
    color: #76d2db;
    margin-bottom: 6px;
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
    font-size: 0.83rem;
  }
  .upd-btn.primary {
    background: #76d2db;
    border-color: #76d2db;
    color: #0b2b1e;
    font-weight: 700;
  }
  .upd-btn.danger {
    background: rgba(224, 112, 112, 0.15);
    border-color: rgba(224, 112, 112, 0.5);
    color: #fca5a5;
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
    font-size: 0.85rem;
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
    height: 8px;
    border-radius: 999px;
    background: #081214;
    overflow: hidden;
  }
  .upd-progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #76d2db, #4ade80);
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
