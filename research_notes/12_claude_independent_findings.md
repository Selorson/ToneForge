# 12 — Kiểm toán độc lập ToneForge (Claude, Pha 1)

> Ngày: 2026-11-12 · Phạm vi: `f:\FL studio\toneforge` (workspace Rust, `shell/src-tauri`, `ui-svelte5-wgpu`, `assets/`, `tools/`, tests).
> Trạng thái niêm phong: tài liệu này được viết **trước khi** mở `11b_sealed_prior_audit_ledger.md` và `10_toneforge_v3_acceptance_report.md`.
> Baseline: `cargo test --workspace --release --no-fail-fast` → **exit 0, toàn bộ ~170 test pass**. Điều đó cho thấy bộ test hiện tại *không* bắt được các lỗi dưới đây — phần lớn là lỗi luồng dữ liệu đầu–cuối giữa các tầng, mà mỗi tầng riêng lẻ vẫn pass test đơn vị.

## Phương pháp

Đọc trực tiếp mã nguồn theo đường đi của tín hiệu và lệnh, từ nút bấm UI → `invoke` Tauri (`shell/src-tauri/src/main.rs`) → `daw-tauri-host` (IPC, project state, undo) → hàng đợi lock-free `rtrb` → `MixerEngine::process_audio_block` (`daw-core-graph`) → plugin DSP/AI → driver WASAPI/ASIO → telemetry 128 KB → UI. Với mỗi tầng, tôi kiểm tra: (1) dữ liệu có thực sự đi tới tầng tiếp theo không, (2) trạng thái hai phía có khớp nhau không, (3) bất biến RT (0 Mutex / 0 alloc / 0 dealloc / không có đột biến CPU), (4) độ trung thực DSP, (5) có dữ liệu giả/placeholder hay không.

Mức độ: **C** = Critical (sai âm thanh, mất dữ liệu, hoặc vi phạm bất biến cấm placeholder), **H** = High (tính năng DAW cốt lõi bị hỏng hoặc thiếu), **M** = Medium, **L** = Low.

---

## A. Dữ liệu giả / placeholder lọt vào đường âm thanh (vi phạm bất biến "cấm mock")

| ID | Mức | Phát hiện | Bằng chứng |
|---|---|---|---|
| **C-01** | C | Lúc khởi động, engine được **bơm tín hiệu tổng hợp** (sine 55/110/440/880/3520 Hz + pink noise, 4096 mẫu, loop) vào Track 1..4 và route về Master. Người dùng mở app là nghe/đo tín hiệu giả; meter, spectrum, LUFS hiển thị số liệu của tín hiệu giả. | `shell/src-tauri/src/main.rs` `build_demo_harmonic_stem` (≈1431) và `main()` (≈2232-2245) |
| **C-02** | C | `cmd_switch_to_piano_roll_source` không chuyển sang Piano Roll mà **lại bơm 4 bản demo noise/sine** vào Track 1..4. | `main.rs` ≈1708-1723 |
| **C-03** | C | Xuất file offline khi chưa nạp audio → render **demo noise/sine** ra WAV như thể là bản master thật. | `main.rs` `cmd_execute_offline_render` ≈1945-1951 |
| **C-04** | C | AI Master Assistant khi chưa có audio → phân tích **demo noise lặp 5 s** và áp preset vào chuỗi master như kết quả "AI" thật. | `main.rs` `cmd_run_ozone12_ai_master_assistant` ≈2086-2092 |
| **C-05** | C | Vocal Pitch Editor khi chưa có audio → tự **tổng hợp "giọng hát" giả** (C4-E4-G4-A4 có vibrato) rồi phân tích pYIN và **commit/bounce bản giả** vào mixer. | `main.rs` `build_demo_vocal_phrase` ≈927, dùng ở ≈984, ≈994, ≈1031 |
| **C-06** | C | `BsrnnStreamingWorker` (fallback "deterministic" của BSRNN) **không chạy mô hình**: là mặt nạ heuristic (tỷ lệ mid/side, transient ratio, hằng số 3.5/2.2/1.8…) nhân với "weight fingerprint" đọc từ file ONNX, có "hidden state" chỉ mang tính trang trí. Fallback này được dùng (a) khi GPU worker không khởi tạo được và (b) **trong toàn bộ offline render**, vì engine offline không gọi `enable_realtime_onnx_gpu_bsrnn_on_master` → bản export có stem rebalance khác hẳn bản nghe live và là phân tách giả. | `ai-uvr5-stem-engine/src/bsrnn_streaming.rs` ≈175-430; `main.rs` ≈1914 (`AudioStreamController::new` cho render) |
| **C-07** | C | Telemetry CPU giả: `cpu_load_pct = 1.2 + active_slots * 0.85` — không đo gì cả. | `daw-core-graph/src/mixer_dag.rs` ≈2843 |
| **H-01** | H | Trạng thái ASIO trả về số liệu bịa: `avg_block_render_us: 120`, `xrun_count: 0`, `mmcss_pro_audio(_elevated): true`, `exclusive_mode_active: true` cố định. | `main.rs` `ActiveHardwareAudioDriver::status_snapshot` ≈73-98 |
| **M-01** | M | Bộ đếm xrun của WASAPI chỉ đếm lỗi API; nhánh phát hiện underrun (`padding == 0`) để trống → xrun gần như luôn 0. | `audio_io/wasapi_hardware_driver.rs` ≈1878-1880 |
| **M-02** | M | `header.true_peak_db` khi không có Ozone trên master được gán bằng **sample peak** của meter master (không oversample) nhưng vẫn mang nhãn True Peak. | `mixer_dag.rs` ≈2812-2815 |

## B. Đứt gãy luồng dữ liệu đầu–cuối (UI ↔ host ↔ RT engine)

| ID | Mức | Phát hiện | Bằng chứng |
|---|---|---|---|
| **C-08** | C | **Audio clip trên Playlist không ảnh hưởng tới âm thanh.** `PlaylistSequencer::on_ppq_tick` chỉ xử lý clip `Automation` và `Pattern`; clip `AudioStem` bị bỏ qua hoàn toàn. `AudioStemClipPlayerNode` chỉ có 1 region mặc định tạo lúc dựng node; `add_region/clear_regions` chỉ được gọi trong test. Mọi thao tác UI (di chuyển, cắt, fade, gain, mute clip, comping) chỉ đổi state, không đổi tiếng. | `sequencer_pianoroll.rs` ≈918-1028, ≈2010-2080; grep `add_region` |
| **C-09** | C | **Thu âm → phát lại bị đứt.** Sau khi dừng thu, take được thêm thành clip `AudioStem` với `source_id = take_number` (1, 2, …) — nhưng `source_id` của AudioStem là *channel id*, và channel 1 là synth đa âm. File WAV vừa ghi không bao giờ được nạp vào engine. Ngoài ra `start_tick: 0` luôn luôn, bất kể vị trí bắt đầu thu. | `main.rs` `cmd_start_stop_disk_recording` ≈599-628 |
| **C-10** | C | **Thu âm ghi sai nguồn.** `RtRecordTap` nhận `sub_block_buf` của track được arm *sau* FX, *sau* fader, *đã cộng* mọi generator/stem đang phát trên track đó — không phải tín hiệu input sạch. | `mixer_dag.rs` ≈2794-2800 |
| **H-02** | H | **Không có lệnh Save/Open Project** cho UI. `save_project_to_file/load_project_from_file` tồn tại nhưng không được đăng ký trong `generate_handler!`. Một DAW không lưu được dự án. | `audio_stream_controller.rs` ≈1358-1365; `main.rs` ≈2311-2377 |
| **H-03** | H | **Mixer không có Mute / Solo / Pan cho track.** `MixerTrack` chỉ có `gain` + `polarity_inverted`; UI mixer chỉ có fader tuyến tính 0..1.5. | `mixer_dag.rs` ≈380-406; `MixerRack502.svelte` ≈77-87 |
| **H-04** | H | **Hai ngăn xếp undo song song không đồng bộ**: UI có `browserUndoStack` riêng, Rust có `ProjectUndoHistory` riêng; undo pop mỗi bên một mục độc lập → trạng thái UI và engine lệch sau undo/redo. Rust undo cũng không trả lại state để UI đồng bộ. | `daw_master_store.svelte.ts` ≈3366-3425; `main.rs` ≈186-221 |
| **H-05** | H | **Kéo fader làm ngập undo**: mỗi sự kiện `oninput` gọi `cmd_set_track_gain` → `record_undo` (clone toàn bộ project state) → 64 bước undo bị một lần kéo fader chiếm hết; đồng thời clone state nặng ở tần suất cao. | `MixerRack502.svelte` ≈83-86; `main.rs` `cmd_set_track_gain` ≈1393 |
| **H-06** | H | AI Master Assistant phản chiếu các band EQ hiệu chỉnh vào project state **cứng vào Track 1 Slot 0**, bất kể `track_idx/slot_idx` được yêu cầu → state lệch với engine. | `main.rs` ≈2116-2133 |
| **M-03** | M | Arm một track mới **không disarm track cũ**: `record_armed`/`input_source_bus` của track trước giữ nguyên → monitor input trên nhiều track. | `main.rs` ≈477-482; `mixer_dag.rs` ≈2089-2103 |
| **M-04** | M | Khi tháo capture bridge (input = "none") mà track vẫn giữ `input_source_bus = Some(0)` (monitor bật), `hw_input_buf` giữ **khối cuối cùng** và bị cộng lặp lại mỗi block → tiếng ù/buzz tuần hoàn. | `mixer_dag.rs` ≈2450-2469, ≈592-610; `main.rs` ≈467-473, ≈477-482 |
| **M-05** | M | Một khe telemetry Pro-Q 4 / Soothe3 duy nhất: mọi instance đều `export_visual_telemetry` vào cùng vùng nhớ → UI hiển thị instance được xử lý *sau cùng* (thường là Master), không phải instance đang chọn. | `mixer_dag.rs` ≈2817-2842; `telemetry_shm.rs` |
| **M-06** | M | Meter của track rời khỏi tập active level không được ghi lại trong frame → các frame của triple buffer giữ meter cũ, UI hiển thị meter "đóng băng". | `mixer_dag.rs` ≈2828-2842 |
| **M-07** | M | `cmd_configure_wasapi_output` bỏ qua `preferred_sample_rate_hz` và `desired_buffer_frames` do UI gửi. | `main.rs` ≈244-245 |
| **M-08** | M | Chuyển driver: nếu `stop_and_take_engine()` trả `None`, `audio_driver` bị để `None` vĩnh viễn → mất engine, mọi lệnh sau báo "shutting down". | `main.rs` ≈253-258, ≈395-419, ≈424-433 |
| **L-01** | L | `ActiveHardwareAudioDriver::shutdown` không làm gì với ASIO (chỉ dựa vào `Drop`). | `main.rs` ≈110-114 |
| **L-02** | L | Take thu âm mặc định ghi vào `%TEMP%\ToneForge_Recordings` — dễ bị dọn dẹp hệ thống xoá mất. | `main.rs` ≈541-547 |

## C. Audio I/O & bất biến thời gian thực

| ID | Mức | Phát hiện | Bằng chứng |
|---|---|---|---|
| **C-11** | C | **ASIO ở sample rate ≠ 48 kHz phá vỡ toàn bộ DSP và tốc độ phát.** Khi driver ASIO không nhận 48 kHz, code gán thẳng `engine.slicer.sample_rate = actual_sr` mà **không prepare lại plugin** (hệ số EQ, K-filter LUFS, delay… vẫn tính cho 48 kHz) trong khi host vẫn resample audio nạp vào về 48 kHz → audio phát sai cao độ/tốc độ, EQ lệch tần số, LUFS sai. | `audio_io/asio_hardware_driver.rs` ≈802-811 |
| **H-07** | H | **Resampler đầu ra WASAPI chất lượng thấp**: sinc 16 tap × 512 pha, không nội suy giữa các pha, Kaiser β=8.6, cutoff 0.95. Với 16 tap, dải chuyển tiếp rộng ≈0.7×Nyquist → suy giảm treble nhìn thấy được từ ~13–14 kHz và aliasing quanh Nyquist. Áp dụng cho mọi thiết bị 44.1/88.2/96 kHz (rất phổ biến). Đường capture dùng 128 tap × 1024 pha — đầu ra lại kém hơn đầu vào. | `audio_io/wasapi_hardware_driver.rs` ≈36-37, ≈414-473, ≈531-577 |
| **H-08** | H | **Đột biến CPU trên luồng RT**: ở chế độ Linear Phase, Pro-Q 4 dựng lại IR FIR (tới 131 072 điểm, đánh giá đáp ứng phức của 24 band trên mọi bin + FFT) **đồng bộ** bên trong `set_parameter_normalized` (gọi từ `drain_commands` trên luồng RT) và bên trong vòng lặp từng mẫu của `process_slice`. Không cấp phát, nhưng vượt deadline → dropout khi chỉnh EQ. | `dsp-proq4-eq/src/lib.rs` ≈207-232, ≈274-284, ≈603-610 |
| **H-09** | H | **Thay đổi gain không làm mượt** (fader track, gain automation `slot_index == 0xFF`, stem gain BSRNN) → zipper noise / click khi kéo fader. | `mixer_dag.rs` ≈2226-2233, ≈2295-2296, ≈2662-2664; `bsrnn_streaming.rs` ≈851-866 |
| **H-10** | H | **Dừng transport cắt cụt âm thanh**: khi `!is_playing`, master bị điền 0 ngay lập tức (không fade-out → click), đuôi reverb/delay bị cắt, và **input monitor không hoạt động khi dừng** (đồ thị mixer không chạy). Note audition / MIDI live khi dừng **đi tắt qua FX của track và chuỗi master** (cộng thẳng vào output, không qua limiter). | `mixer_dag.rs` ≈2479-2556 |
| **M-09** | M | Capture input luôn khởi tạo `AdaptiveDllDriftResampler::new(48_000.0, engine_sr)` bất kể định dạng thật của thiết bị thu; độ trễ input cố định `480` mẫu, bỏ qua độ trễ ASIO báo cáo (`get_latencies`). | `main.rs` ≈456, ≈466; `asio_hardware_driver.rs` ≈885-906 |
| **M-10** | M | Sự kiện MIDI input bị gán `sample_pos` = đầu block → jitter thời gian tới 1 block, bỏ qua timestamp của cổng MIDI. | `mixer_dag.rs` ≈2437-2441 |
| **M-11** | M | `RtGraphMutation::UpdateSampleRate` chỉ đổi `slicer.sample_rate` + reset state, **không re-prepare plugin** ở rate mới. | `mixer_dag.rs` ≈2077-2088 |
| **M-12** | M | `asio_cb_message` không xử lý `kAsioResetRequest`, `kAsioBufferSizeChange`, `kAsioResyncRequest`, `kAsioLatenciesChanged` → driver đổi cấu hình là stream im lặng mà không khôi phục. | `asio_hardware_driver.rs` ≈523-534 |
| **M-13** | M | VST3 plugin bị `Drop` (terminate/release/`FreeLibrary`) trên luồng `toneforge-rt-gc`, không phải luồng UI/main như đặc tả VST3 yêu cầu → rủi ro crash với plugin bên thứ ba. | `mixer_dag.rs` RtGarbageCollector; `daw-plugin-host/src/vst3_host.rs` `impl Drop` |
| **L-03** | L | VST3 `ProcessContext` thiếu vị trí nhạc (PPQ, `kProjectTimeMusicValid`, `kTimeSigValid`), bỏ qua sidechain bus. | `vst3_host.rs` ≈1882-1891 |

## D. Độ trung thực DSP / đo lường

| ID | Mức | Phát hiện | Bằng chứng |
|---|---|---|---|
| **C-12** | C | **Integrated LUFS chỉ tính ~51 giây gần nhất**: vòng gating chỉ có `MAX_GATING_BLOCKS = 512` khối (hop 100 ms), khối cũ bị ghi đè → với bài 3–4 phút, "Integrated" thực chất là cửa sổ trượt 51 s, sai so với ITU-R BS.1770-4 / EBU R128. Ảnh hưởng trực tiếp quyết định mastering và kiểm tra chuẩn streaming. | `dsp-ozone12-mastering/src/lufs_true_peak_meter.rs` ≈289, ≈470-520 |
| **H-11** | H | **Luật pan của `AudioStemClipPlayerNode` làm stem stereo nhỏ đi 3 dB ở vị trí giữa** (`L·cos(π/4)`, `R·sin(π/4)`) — luật equal-power cho mono bị áp lên nguồn stereo. | `sequencer_pianoroll.rs` ≈2234-2238 |
| **H-12** | H | BSRNN GPU async: khi GPU trễ, `pop_aligned_output_hop` trả về hop rỗng → phần delta rebalance biến mất trong hop đó (output = dry) → stem rebalance "bơm" theo tải GPU. | `bsrnn_streaming.rs` ≈735-768, ≈897-907 |
| **M-14** | M | Trong 100 ms đầu, `lufs_integrated` bị gán bằng loudness tức thời không gating. | `lufs_true_peak_meter.rs` ≈441-452, ≈459-468 |
| **M-15** | M | Offline render chạy `process_audio_block` với `is_offline_rendering: false` cố định → plugin (VST3 `kRealtime`, worker GPU async) xử lý như realtime; kết quả render phụ thuộc tốc độ máy. | `mixer_dag.rs` ≈2490, ≈2620, ≈2708; `offline_render_engine.rs` ≈664-690 |

## E. UX

| ID | Mức | Phát hiện |
|---|---|---|
| **M-16** | M | Fader track tuyến tính 0..1.5 (không theo dB), chỉ hiển thị peak L. |
| **M-17** | M | Không có thao tác Mute/Solo/Pan, không có thêm/xoá/bypass insert từ mixer (chỉ hiển thị tên slot). |
| **L-04** | L | Bank selector cứng (0, 32, 128, 256, 470), không cuộn tự do. |

---

## Tổng kết Pha 1

- **12 Critical**, **12 High**, **17 Medium**, **4 Low** (tính theo ID ở trên).
- Mẫu chung: các module đơn lẻ (EQ, Soothe, Ozone, plugin host, driver) được viết kỹ và có test riêng, nhưng **lớp nối** giữa chúng (sequencer ↔ clip audio, recorder ↔ playlist, undo UI ↔ undo Rust, render offline ↔ engine live) có nhiều đứt gãy mà test hiện tại không phủ. Ngoài ra còn nhiều đường "an toàn" mặc định bơm tín hiệu tổng hợp vào chỗ đáng lẽ phải báo lỗi.
- Ưu tiên sửa ở Pha 2 (theo bất biến chất lượng âm thanh): C-01..C-07 (loại bỏ dữ liệu giả), C-08..C-10 (luồng audio clip và thu âm), C-11 + H-07 (I/O), C-12 (LUFS), H-08/H-09/H-10 (RT và độ mượt), H-02..H-06 (lưu dự án, mixer, undo).
