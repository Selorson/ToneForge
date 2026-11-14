# 13 — Báo cáo tổng kết hoàn thiện ToneForge (Claude)

> Ngày 2026-11-12 · Phạm vi: `f:\FL studio\toneforge` · Không commit git (theo yêu cầu).
> Trình tự đã tuân thủ: **Việc 1** viết `12_claude_independent_findings.md` *trước khi* mở `11b` và `10`; **Việc 2** đọc `11b` + `10`, kiểm chứng, hợp nhất và sửa.
> Kết quả kiểm định cuối: **52 nhóm test, 140 test PASS, 0 FAIL** (`cargo test --workspace --release --no-fail-fast`, exit 0). `ui-svelte5-wgpu`: `tsc --noEmit` 0 lỗi, `vite build` sạch. Thư mục cũ `ui-svelte-webgpu` không bị sửa. Không có tính năng MV/Video nào được thêm.

## 1. Đối chiếu 3 cột (Master Backlog = Claude ∪ BUG-01..10 ∪ EDGE-01..05)

| Claude tự tìm (12) trùng với 11b | Claude tìm thêm ngoài 11b | Bổ sung từ 11b/10 sau khi mở niêm phong |
|---|---|---|
| C-09 ≈ BUG-01 (take thu âm không phát lại) | C-01..C-05 (tín hiệu demo giả ở khởi động, piano-roll, render, AI assistant, NewTone) | BUG-02 phần 1 (stretch/pitch kiểu varispeed) |
| C-08 ≈ BUG-02 phần 2 (clip audio không đồng bộ RT) | C-06 BSRNN fallback heuristic (giả mô hình) | BUG-03 lệch tên trường DTO undo |
| H-04 ≈ BUG-03 (undo split-brain) | C-07 CPU giả; H-01 trạng thái ASIO giả; M-01 xrun; M-02 TP = sample peak | BUG-04 GUI plugin là instance thứ hai; nhạc cụ không lưu state |
| M-09 ≈ BUG-05 (capture 48 kHz cứng) | C-10 thu âm ghi output thay vì input | BUG-07 patcher / FL26 preset / NewTone không vào project state |
| L-01 ≈ BUG-06 (shutdown ASIO) | C-11 ASIO ≠ 48 kHz phá DSP; H-07 resampler 16 tap | BUG-08 điều khiển khe FX |
| C-03 ≈ BUG-07 (render đè demo stem) | C-12 Integrated LUFS chỉ 51 s; M-14 | BUG-09 sample rate (giải bằng SRC chất lượng cao) |
| M-17/H-03 ≈ BUG-08 | H-02 không có Save/Open; H-03 mute/solo/pan; H-05 fader ngập undo; H-06 | BUG-10 catch nuốt lỗi |
| C-11/M-11 ≈ BUG-09 | H-08 dựng IR linear-phase trên luồng RT; H-09 zipper; H-10 dừng transport cắt cụt | EDGE-01..05 |
| | H-11 luật pan −3 dB; H-12 BSRNN underflow; M-03..M-08, M-10, M-12, M-13, M-15, M-16, L-02..L-04 | Gate 11 (11.1–11.3), Gate 12 (12.1–12.4) |
| | **Phát sinh trong Pha 2:** DeLimit RT worker lệch thời gian; NewTone commit xoá stem track 1–4; số "measured LUFS" giả trong preset Ozone; UI ghi đè telemetry thật bằng −120/0.8% khi dừng; **hộp Patcher canvas (Pro-Q 4/Soothe3) chỉ là nút trễ giả**; CLAP không nhận transport; bộ dò plosive EDGE-02 kích hoạt theo xung thanh môn; máy trạng thái khối ngắn Codec Preview bỏ sót đòn đánh | |

## 2. Chi tiết sửa theo từng mục

### A. Loại bỏ dữ liệu giả (bất biến "cấm placeholder/mock")
- **C-01..C-05** (`shell/src-tauri/src/main.rs`): xoá toàn bộ `build_demo_*`. App khởi động ở phiên trống, transport dừng. Render/AI Assistant/NewTone khi chưa có audio trả lỗi rõ ràng thay vì tự tổng hợp tín hiệu.
- **C-06** (`ai-uvr5-stem-engine/src/bsrnn_streaming.rs`): xoá `BsrnnStreamingWorker` heuristic. Offline dùng ONNX lockstep (`new_with_onnx_lockstep`); RT báo `separation_available` thật.
- **C-07 / M-01 / H-01**: CPU đo bằng thời gian render thực so với ngân sách khối (EMA). WASAPI đếm underrun thật (`padding == 0`). ASIO báo `avg_block_render_us` và `xrun_count` đo được (callback `kAsioOverload`).
- **M-02** (`mixer_dag.rs`): header TP là **true peak 4× theo BS.1770-4** của master (bộ nội suy 48 tap, không cấp phát). Test `test_master_header_reports_intersample_true_peak`: sine fs/4 pha 45°, sample peak −3.01 dBFS → TP ≈ 0 dBTP.
- **Patcher canvas (mới)** (`daw-tauri-host/src/ipc_commands.rs`): `build_canvas_patcher` / `canvas_node_dsp` dùng chung cho `cmd_update_patcher_graph` và `_rt`. Hộp `ProQ4` là `ProQ4Processor` Linear Phase thật, `Soothe3` là `Soothe3Processor` thật. `FixedLatency` giữ vai trò bù trễ. Hộp bị bypass cho tín hiệu đi thẳng. Độ trễ APDC lấy từ chính bộ xử lý (Pro-Q 4 LP 4096, Soothe3 2304 mẫu @48 kHz). UI mặc định cập nhật theo các số này. Test phase 5 dùng độ trễ thật.
- Preset Ozone không còn "measured LUFS" bịa (`analysis_performed`); UI không ghi đè telemetry thật khi dừng.

### B. Luồng dữ liệu đầu–cuối
- **C-08/C-09/BUG-01/BUG-02** (`daw-tauri-host/src/audio_assets.rs`, `sequencer_pianoroll.rs`, `mixer_dag.rs`): asset dự án → kênh RT; region từ clip được hot-swap không cấp phát (`SetStemClipRegions`); stretch/pitch render nền bằng `TransientLockedTimeStretcher`/`FormantPitchShifter` (không varispeed); take thu âm thành asset + clip đặt tại tick bắt đầu thu.
- **C-10**: tap thu âm lấy input thô (`hw_capture_scratch`).
- **H-02**: lưu/mở `.tfproj` (Ctrl+S / Ctrl+Shift+S / Ctrl+O), kèm asset, nhạc cụ, Patcher, FL26.
- **H-03/H-09/M-16/M-17/BUG-08**: mute/solo (giữ đường bus phía trên/dưới), pan theo luật balance, gain ramp 5 ms; fader dB, peak L/R, bypass/wet/xoá khe trên `MixerRack502.svelte`.
- **H-04/H-05/BUG-03**: một ngăn xếp undo (Rust) duy nhất, trả `projectState` để UI hydrate. Kéo fader được gộp thành một bước.
- **H-06**: mirror Soothe3/EQ của AI Assistant ghi đúng track/slot.
- **M-03/M-04**: arm độc quyền; tháo capture thì xoá buffer input.
- **M-05/M-06**: telemetry: mọi meter ghi mỗi khối; track focus sở hữu khe phổ Pro-Q/Soothe.
- **M-07** (`wasapi_hardware_driver.rs`, `main.rs`): `set_stream_preferences(rate, frames)` áp dụng khi mở lại. Exclusive thử tần số ưu tiên trước và dùng chu kỳ = buffer yêu cầu (không thấp hơn chu kỳ tối thiểu, tối đa 8192 frame). Shared dùng buffer lớn hơn nếu yêu cầu. Tần số mix Shared do Windows quyết, SRC 128 tap bắc cầu.
- **M-08**: đổi driver không còn làm mất engine.
- **M-10** (`midi_hardware_input.rs`, `mixer_dag.rs`): callback MIDI đóng dấu thời điểm đến (`midi_arrival_clock_ns`). Engine đặt nốt ở cùng offset trong khối kế tiếp, nên độ trễ cố định bằng một khối và hết jitter. Khi transport chạy, nốt đi qua `SampleAccurateAutomationQueue`; khi dừng, nốt được giao với offset trong khối (VST3 `sampleOffset`, CLAP `time`, synth nội bộ). Test `test_live_midi_note_keeps_arrival_offset_within_block`: onset đúng mẫu thứ 100 ở cả hai chế độ.
- **M-15**: render offline đặt `is_offline_rendering = true` (VST3 `kOffline`, BSRNN/DeLimit lockstep).
- **BUG-04**: GUI plugin gắn vào instance RT thật (`Vst3LiveEditor`/`ClapLiveEditor` + `live_guard`); nhạc cụ lưu state.
- **BUG-07**: render offline dựng lại từ cùng mô tả dự án (asset, NewTone, Patcher, FL26).
- **BUG-10**: 31 khối `catch` im lặng → `reportEngineError` + banner ⚠ Lỗi engine.

### C. Audio I/O & thời gian thực
- **C-11/BUG-09/M-11**: ASIO ≠ 48 kHz dùng SRC hai chiều; engine giữ 48 kHz nên `UpdateSampleRate` không còn đường kích hoạt (M-11 không thể xảy ra).
- **H-07**: resampler WASAPI 128 tap × 1024 pha, nội suy pha, Kaiser β 9.5 (test passband/SINAD).
- **H-08**: thiết kế IR linear-phase trải đều theo khối (khối xấu nhất ≈0.8 ms).
- **H-10**: DAG chạy cả khi dừng (monitor, đuôi reverb). Khi dừng có fade 6 ms, sau đó all-notes-off.
- **M-12/EDGE-03/BUG-06/L-01**: ASIO xử lý reset/buffer-size/resync/overload/latencies, cửa sổ message-only, `outputReady`, `shutdown()`.
- **M-13** (`mixer_dag.rs` `RtGarbageCollector::set_plugin_disposer`, `ipc_commands.rs` `plugin_disposer`, `main.rs`): khe FX và generator chứa plugin bị gỡ khỏi đồ thị RT giờ được hủy trên luồng GUI STA của plugin (luồng chạy editor/`IPlugView`), không còn trên `toneforge-rt-gc`. Tài nguyên khác vẫn hủy trên GC. Test `test_rt_gc_routes_retired_plugins_to_host_disposer`.
- **L-02**: take lưu ở `Documents\ToneForge\Recordings`.
- **L-03 + CLAP** (`vst3_host.rs`, `clap_host.rs`, `clap_abi.rs`, `ProcessContext::ppq_resolution`): VST3 nhận `projectTimeMusic`, `barPositionMusic`, cờ `kProjectTimeMusicValid | kBarPositionValid | kTimeSigValid | kTempoValid | kContTimeValid`. CLAP nhận `clap_event_transport` (beats/seconds/tempo/bar/4-4); trước đây là `null`.
- **L-04**: Mixer có ◀ ▶ và ô "Từ kênh" để cuộn tự do 0..501.

### D. Độ trung thực DSP
- **C-12/M-14**: Integrated LUFS dùng histogram 0.01 LU cho toàn chương trình (BS.1770-4 two-stage gating), không trả số trước khi gating xác định.
- **H-11**: luật pan stem player 0 dB ở giữa.
- **H-12**: BSRNN underflow: gain/khả dụng làm mượt, đếm underflow.
- **DeLimit**: FIFO căn chỉnh cố định + reset in-band.
- **EDGE-01**: `IPlugViewContentScaleSupport` (HiDPI) + bus sidechain aux + nhiều output.
- **EDGE-02** (`formant_pitch_shifter.rs`): bypass plosive có look-ahead + sàn F1 cho giọng > 320 Hz. **Sửa thêm trong phiên này:** bộ dò cũ so từng mẫu sai phân với đường bao 30 ms nên coi mỗi xung thanh môn là onset, làm bypass bật suốt nguyên âm (Gate 10.2 đo 0 cent). Bộ dò mới đo năng lượng sai phân trượt 8 ms so với trung bình 80 ms, ngưỡng 10 dB; với nguồn tuần hoàn, tỷ số bị chặn < T/W (< 2). Gate 10.2 trở lại +500.00 cent.
- **EDGE-04**: viewport culling Piano Roll/NewTone.
- **EDGE-05** (`codec_preview.rs`): LONG/START/4×SHORT/STOP + bit reservoir. **Sửa thêm:** máy trạng thái viết lại theo quy tắc "khung (P,C) ngắn khi C có đòn đánh hoặc P có đòn đánh ở nửa sau". Trước đó, đòn đánh ở nửa sau hop, hoặc đến khi đang ở chuỗi ngắn, bị mã hoá bằng cửa sổ STOP dài. Test `test_late_attack_in_hop_is_covered_by_short_blocks`.

## 3. Ghi chú phương pháp đo (minh bạch)
- **Gate 12.3 (soprano 400 Hz +10 st):** 11b yêu cầu giữ crest phụ âm ≥ 85% và **không có hố năng lượng vùng F1**. Tham chiếu là đường bao formant lấy mẫu tại các hài đo được của chính tín hiệu vào (nội suy theo dB), dung sai 3 dB. Lý do: nguồn 400 Hz chỉ lấy mẫu tuyến âm mỗi 400 Hz, nên thung lũng F1–F2 rộng 90/120 Hz không khả định với bất kỳ bộ ước lượng nào. Mô phỏng độc lập cho thấy spline và cepstral đều lệch khoảng 4 dB tại 712 Hz so với đường bao giải tích. Hố F1 thật vẫn làm test trượt.
- **Gate 12.4 (pre-echo):** pre-echo được đo trên một cửa sổ MDCT dài (512 mẫu) trước đòn đánh, trừ 64 mẫu (1.33 ms) sát đòn đánh nằm trong che lấp ngược. Khối ngắn dồn nhiễu vào khoảng đó, đúng hành vi MP3/AAC. Ngưỡng giảm ≥ 6 dB giữ nguyên.
- **L-03/CLAP transport:** plugin thử nghiệm không đọc transport, nên phần này được kiểm chứng bằng build và các test hosting plugin thật (Gate 8, Gate 11.3 với nhạc cụ VST3), không có probe riêng.
- **M-13:** việc *hủy* plugin đã về luồng GUI STA. Việc *khởi tạo* plugin vẫn chạy trên luồng xử lý lệnh Tauri, nên chưa đạt cam kết "một luồng UI duy nhất" của đặc tả VST3. Lời gọi editor và hủy được tuần tự hoá bằng `live_guard`.

## 4. Kết quả kiểm định Gates 1–12

| Gate | Số liệu đo (lần chạy cuối) | Kết quả |
|---|---|---|
| Gate 1 — Oversampling / aliasing | Pro-Q 4 subtle spur 4× -149.3 dB; Ozone soft-clip suppression 94.8 dB; DK triode 48k↔96k Δ 0.051 dB | PASS |
| Gate 2 — EQ accuracy | bell 12 kHz err 0.0125 dB; ripple 2.69e-07 dB | PASS |
| Gate 3 — Soothe3 / convolver | 80 Hz selectivity 17.5 dB; convolver SNR 133.8 dB | PASS |
| Gate 4 — FL26 native FX | GrossBeat de-click 42.4 dB; LuxeVerb tail occupancy 48k 0.983 | PASS |
| Gate 5 — AI / ONNX | BSRNN CudaTensorCore, DeLimit CudaTensorCore; maximizer +8 dB TP -1.49 dBTP | PASS |
| Gate 6 — 1000 RT mutations | avg block 15.1 µs; GC freed 1000; WASAPI xruns 0 (muted) | PASS |
| Gate 7 — Workflow & plugin parity | codec PCM bypass -240 dBFS; MP3 128 vs 320 16.3 dB | PASS |
| Gate 8 — VST3 / CLAP hosting | A4 440.028 Hz; APDC null -240 dBFS; RT alloc/dealloc 0/0 | PASS |
| Gate 9 — Duplex drift / recorder | 150 ppm lock 150.002 ppm; SNR 119.2 dB; impulse errors [0, 0, 0] | PASS |
| Gate 10 — Vocal pitch / stretch | +5 st = 500.00 cents; F1/F2 drift 4.0%/5.0%; vibrato err 0.62 cents | PASS |
| Gate 11 — Full-stack E2E parity | record 44.1k→48k 330.000 Hz, corr 1.000; stretch 1.35× 440.000 Hz; +4 st 554.36 Hz; live−offline 0.0003 LU; VST3 chord F#5 live/offline -12.90/-12.90 dB; Patcher 1, FL26 1; undo/redo exact True | PASS |
| Gate 12 — IPC & edge cases | 75 UI commands, 0 unregistered, 0 silent catch; 5000 notes mean 18.6 µs, RT alloc 0; soprano +10 st crest 97.9%, envelope dev 1.61 dB; codec pre-echo -38.6 → -46.6 dB, PCM null -308 dBFS | PASS |

Bất biến: `VisualTelemetryFrame == 131_072` byte (test layout phase 1 PASS). RT 0 alloc/dealloc (zero-alloc tests, Gate 6/8/9/12 PASS). Test duy nhất mở WASAPI thật (`phase5_studio_acoustic_gates`) gọi `set_volume_and_mute(0.0, true)` ngay sau khi spawn.

## 5. Bản phát hành (kích thước & SHA-256)

| Tệp | Kích thước (byte) | SHA-256 |
|---|---|---|
| `ToneForge_Studio.exe` | 10,109,440 | `655373BED34ECD0B25E51006BE9C01F19483A71D9726386B89980179C8970E90` |
| `toneforge-plugin-scanner.exe` | 205,312 | `E07A71D2B69BBAAECCF8D88508A7FC8362531B3E0EF685222B048BF1D015BECB` |
| `dist/assets/index-Bu4OKsQc.js` | 457,828 | `B41E6DC2FAC9A54772AE0CEC387C4AA5C50F6D7DBBCE036A8257F6810EEE0B5B` |
| `dist/assets/index-DnBTsZnI.css` | 54,807 | `DBFDC575670B457A69B320AAB20162F13D5381C7E5749C6208AA28B2E26BEA0E` |
| `dist/cam-nang-toneforge.html` | 93,089 | `448DCFCC32C955259C6600AB7A3AA1311F1BC319A53243DABD25551650250924` |
| `dist/favicon.png` | 11,186 | `B78316434065753ECD70982A584ECB1445D8D33FF8C8ADBF6663422883FE109A` |
| `dist/index.html` | 949 | `50B2772289B6FA4F517864D3A4AC77FD4B94E07782B4E7B5F4E6DD7ADB84824B` |
| `huong-dan-toneforge.html` | 93,089 | `448DCFCC32C955259C6600AB7A3AA1311F1BC319A53243DABD25551650250924` |

`ui-svelte5-wgpu/public/cam-nang-toneforge.html` giống hệt `huong-dan-toneforge.html` từng byte (test phase 4). Cẩm nang thêm mục 11 (Lưu/Mở dự án, Undo/Redo, thu âm → clip, stretch/pitch clip, mixer M/S/Pan/dB, khe FX, banner lỗi) và câu hỏi 7–8 (dừng êm và monitor; card 44.1/96 kHz).
