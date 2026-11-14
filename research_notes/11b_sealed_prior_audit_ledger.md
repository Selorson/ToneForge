# 🔒 SỔ CÁI KIỂM TOÁN PHÁP Y NIÊM PHONG (`v1` + `v2` + `v3`) — CHỈ MỞ Ở PHA 2 SAU KHI ĐÃ GHI XONG `12_claude_independent_findings.md`

> **CẢNH BÁO QUY TRÌNH:** Nếu bạn chưa hoàn tất kiểm toán độc lập và chưa lưu file `f:\FL studio\research_notes\12_claude_independent_findings.md` xuống đĩa, hãy đóng ngay tài liệu này lại và hoàn thành Pha 1 trước!
>
> Khi đã bước sang **Pha 2**, hãy nghiên cứu thật sâu từng mục dưới đây (cùng với Mục 5 của `f:\FL studio\research_notes\10_toneforge_v3_acceptance_report.md`), kiểm chứng trực tiếp trên mã nguồn thực tế, hợp nhất toàn bộ với các phát hiện độc lập của riêng bạn (`Master Backlog = Claude Independent Findings ∪ BUG-01..BUG-10 ∪ EDGE-01..EDGE-05`), và khắc phục triệt để **100%** ở chuẩn sản xuất phòng thu thương mại.

---

## PHẦN I — 10 LỖI HỆ THỐNG, ĐỨT GÃY TÍCH HỢP ĐẦU-CUỐI & DI SẢN XUYÊN THẾ HỆ (`v1` → `v2` → `v3`)

Mặc dù toàn bộ `129/129` bài kiểm định `--release` (`Gates 1–10`) đều vượt ngưỡng toán học cấp module rất xuất sắc, đợt kiểm toán pháp y mã nguồn từ đầu đến cuối xuyên suốt 3 thế hệ (`v1` Genesis, `v2` Studio Mastering, `v3` Full DAW Parity) đã phát hiện **10 đứt gãy nghiêm trọng ở lớp keo tích hợp (`Host Glue / State Sync / RT Bridge / UI Contract`)**:

---

### `BUG-01` [CRITICAL — `v3` Audio Recording E2E] File `.wav` Thu Âm Xong Không Được Nạp Lên Đồ Thị Phát Lại Thời Gian Thực Lẫn Offline Render
- **Vị trí mã nguồn:**
  - `f:\FL studio\toneforge\shell\src-tauri\src\main.rs` (khoảng dòng `588–631`, hàm `cmd_start_stop_disk_recording`)
  - `f:\FL studio\toneforge\crates\daw-core-graph\src\sequencer_pianoroll.rs` (`AudioStemClipPlayerNode`)
- **Cơ chế gây lỗi:**
  - Khi người dùng bấm dừng thu âm (`start == false`), `recorder.finish_and_collect_takes()` đóng gói chính xác các file `32-bit IEEE Float WAV` vào `%TEMP%\ToneForge_Recordings` và `main.rs` đẩy các `PlaylistClipDto { kind: "AudioStem", source_id: ui_take.take_number, ... }` vào `PlaylistSequencer` qua `cmd_upsert_playlist_clip_rt`.
  - **Tuy nhiên**, không hề có bước đọc/giải mã mẫu PCM từ file `.wav` vừa thu (hoặc giữ lại bộ đệm PCM từ `DiskTrackRecorder`), không lưu đường dẫn file `.wav` của từng Take vào `ToneForgeProjectState` (để lưu/mở dự án `.tfproj`), và **không gắn/cập nhật `AudioStemClipPlayerNode` / `ChannelGenerator` trên kênh Mixer tương ứng**!
  - Hệ quả: Người dùng thu âm xong nhìn thấy khối Clip hiện trên Playlist, nhưng bấm `Play` thì kênh đó im lặng hoàn toàn!
- **Vì sao Gate 9.2 không bắt được:** Gate 9.2 kiểm định trực tiếp `DiskTrackRecorder` ghi ra đĩa đúng `0-sample error` và `0` RT alloc, nhưng không đi qua `cmd_start_stop_disk_recording` → `Play` trên `SharedDawController`.
- **Yêu cầu sửa chuẩn phòng thu:**
  1. Lưu danh mục `AudioAssetRegistry` (id tài nguyên âm thanh, đường dẫn `.wav`/stem, sample rate gốc, số kênh, độ dài mẫu, và bộ đệm PCM `Arc<Vec<[f32; 2]>>` đã chuẩn hóa về `engine_sample_rate`) ngay trong `SharedDawController` + metadata đường dẫn trong `ToneForgeProjectState`.
  2. Ngay khi `finish_and_collect_takes()` hoàn tất, nạp PCM của từng Take vào `AudioAssetRegistry`, đồng bộ vùng phát `StemClipRegion` xuống `AudioStemClipPlayerNode` trên kênh đích qua `RtGraphMutation` không khóa (`0` RT alloc, rác cũ đẩy qua `RtGc`), và đảm bảo chỉ Take đang được chọn (`active_take` / comping) phát tiếng khi trùng làn lặp.
  3. Đảm bảo `OfflineRenderEngine` và Lưu/Mở dự án `.tfproj` nạp lại đầy đủ các Take này.

---

### `BUG-02` [CRITICAL — `v3` Playlist DSP Bridge] Kéo Giãn Thời Gian (`stretch_ratio`) & Dịch Tông (`pitch_semitones`) Trên Audio Clip Đang Chạy Sai Thuật Toán (Kiểu Băng Từ `Varispeed` Thay Vì `TransientLockedTimeStretcher`) + Kiểm Tra Đồng Bộ `PlaylistSequencer` ↔ `AudioStemClipPlayerNode`
- **Vị trí mã nguồn:**
  - `f:\FL studio\toneforge\crates\daw-core-graph\src\sequencer_pianoroll.rs` (khoảng dòng `2185–2215`, `AudioStemClipPlayerNode::process_slice` và `PlaylistSequencer`)
  - `f:\FL studio\toneforge\crates\daw-tauri-host\src\ipc_commands.rs` (khoảng dòng `2958–3063`, `cmd_playlist_slice_or_edit_clip_rt` & `cmd_upsert_playlist_clip_rt`)
- **Cơ chế gây lỗi (2 tầng lỗi chồng nhau):**
  1. **Sai bản chất vật lý khi giãn thời gian / dịch tông:** Trong `AudioStemClipPlayerNode::process_slice`, code tính `let rate = pitch_rate / stretch; let src_pos = reg.source_offset_samples + rel * rate;` rồi nội suy Hermite 4 điểm! Công thức này là **đổi tốc độ quay băng từ (`varispeed`)**: nếu chỉ đổi `stretch_ratio = 1.35x` (với `pitch_semitones = 0.0`), tốc độ đọc giảm còn `1 / 1.35 = 0.7407x`, làm **tông nhạc bị tụt thê thảm `-5.2` bán âm** thay vì giữ nguyên cao độ! Trong khi đó, crate `dsp-pitch-time` đã có sẵn `TransientLockedTimeStretcher` và `FormantPitchShifter` đạt chuẩn Elastique ở Gate 10.3 nhưng chưa được nối vào luồng render bộ đệm Audio Clip!
  2. **Đứt mạch đồng bộ vùng phát trên Playlist:** Kiểm tra kỹ mối liên kết giữa `PlaylistSequencer` (nơi lưu `PlaylistClip` với `kind == PlaylistClipKind::AudioStem`) và `AudioStemClipPlayerNode::regions` (`Vec<StemClipRegion>`): khi người dùng thực hiện `Razor Slice`, `Slip` (`source_offset_samples`), `Fade In / Fade Out`, xóa hoặc di chuyển clip trên Playlist qua `cmd_playlist_slice_or_edit_clip_rt` / `cmd_upsert_playlist_clip_rt`, danh sách `StemClipRegion` bên trong `AudioStemClipPlayerNode` trên luồng RT bắt buộc phải được tái tổng hợp và tráo nóng (`hot-swap` qua `RtGraphMutation` + `RtGc`) khớp 100% từng mẫu!
- **Yêu cầu sửa chuẩn phòng thu:**
  1. Khi một Audio Clip có `stretch_ratio != 1.0` hoặc `pitch_semitones != 0.0`, thực hiện xử lý ngoại tuyến trên luồng worker nền bằng `dsp_pitch_time::TransientLockedTimeStretcher` (và `FormantPitchShifter` / khóa pha đa âm) để tạo bộ đệm PCM `Arc<Vec<[f32; 2]>>` đã giãn thời gian / dịch tông chuẩn xác mà không đổi tốc độ/tông chéo nhau, sau đó tráo nóng không khóa vào `AudioStemClipPlayerNode`.
  2. Khép kín hàm đồng bộ tự động `sync_playlist_audio_clips_to_rt_players(...)` mỗi khi clip `AudioStem` được thêm, cắt (`Razor Slice`), di chuyển, chỉnh `fade_in_ticks`/`fade_out_ticks`, `source_offset_samples`, chọn `take_lane_id`, hoặc Hoàn tác/Làm lại (`Undo/Redo`).

---

### `BUG-03` [CRITICAL — `v3` State & IPC Contract] "Split-Brain Undo/Redo" Giữa Ngăn Xếp Rust Và Ngăn Xếp Svelte 5 + Lệch Tên Trường JSON Khiến Bộ Đếm Undo Trên UI Luôn Bằng `0`
- **Vị trí mã nguồn:**
  - `f:\FL studio\toneforge\crates\daw-tauri-host\src\undo_history.rs` (dòng `40–49`, `UndoRedoStatusDto`)
  - `f:\FL studio\toneforge\shell\src-tauri\src\main.rs` (dòng `186–226`, `cmd_undo_project_action` / `cmd_redo_project_action` / `cmd_get_undo_redo_status`)
  - `f:\FL studio\toneforge\ui-svelte5-wgpu\src\lib\state\daw_master_store.svelte.ts` (dòng `3296–3428`)
- **Cơ chế gây lỗi:**
  1. **Lệch tên trường DTO (`Schema Mismatch`):** `UndoRedoStatusDto` gắn `#[serde(rename_all = "camelCase")]` xuất ra các trường `undoDepth`, `redoDepth`, `lastAppliedLabel`, `nextUndoLabel`, `nextRedoLabel`. Thế nhưng `daw_master_store.svelte.ts` (dòng `3357–3359`) lại đọc `st.undoCount`, `st.redoCount`, `st.lastActionLabel`! Kết quả là trong ứng dụng desktop Tauri thật, huy hiệu số bước Undo/Redo trên thanh Topbar luôn kẹt cứng ở số `0`!
  2. **Hai ngăn xếp Undo lệch pha nhau ("Split-Brain"):** Hàng loạt lệnh chỉnh sửa trực tiếp xuống Tauri (`cmd_update_proq4_band`, `cmd_update_proq4_global`, `cmd_set_track_gain`, `cmd_route_mixer_track`, `cmd_set_external_plugin_param`, nạp preset...) đều gọi `record_undo_snapshot(...)` bên phía Rust **mà không đẩy bản chụp vào `browserUndoStack` của TypeScript**. Ngược lại, khi bấm `Ctrl+Z`, `cmd_undo_project_action` khôi phục `MixerEngine` + `ToneForgeProjectState` bên Rust nhưng **chỉ trả về `UndoRedoStatusDto` chứ KHÔNG trả về `ToneForgeProjectState` đã khôi phục cho giao diện Svelte**! Giao diện Svelte hoặc không cập nhật núm vặn/dải EQ theo trạng thái âm thanh vừa hoàn tác, hoặc rút nhầm một bản chụp cũ không liên quan từ `browserUndoStack`!
- **Yêu cầu sửa chuẩn phòng thu:**
  1. Biến `UndoHistoryManager` phía Rust thành **Nguồn Sự Thật Duy Nhất (`Single Source of Truth`)** khi chạy trong Tauri (`browserUndoStack` chỉ được kích hoạt khi chạy thuần trình duyệt không có `__TAURI_INTERNALS__`).
  2. Mở rộng kết quả trả về của `cmd_undo_project_action` và `cmd_redo_project_action` để trả về trọn bộ `{ status: UndoRedoStatusDto, projectState: ToneForgeProjectState }` (hoặc gói DTO đồng bộ đầy đủ Mixer, Playlist, Piano Roll, Patcher, Pro-Q 4, Soothe3, Ozone 12, Vocal Pitch, External Plugins), và viết hàm `hydrateStoreFromProjectState(ps)` trong `daw_master_store.svelte.ts` để cập nhật 100% giao diện khớp tuyệt đối với âm thanh sau mỗi lần `Ctrl+Z` / `Ctrl+Y`.
  3. Đồng bộ chính xác tên trường `undoDepth` / `redoDepth` / `lastAppliedLabel` / `nextUndoLabel` / `nextRedoLabel` giữa Rust và TypeScript, đồng thời tự động cập nhật huy hiệu Undo sau mọi thao tác chỉnh sửa.

---

### `BUG-04` [HIGH — `v3` VST3/CLAP Hosting] Mở Cửa Sổ Đồ Họa `Win32 HWND` Đang Khởi Tạo Một Thực Thể Plugin Thứ Hai Tách Rời + Chế Độ Nhạc Cụ (`Instrument Mode`) Bị Bỏ Quên Trong `ProjectState`, Điều Khiển Tham Số & `Offline Render`
- **Vị trí mã nguồn:**
  - `f:\FL studio\toneforge\crates\daw-tauri-host\src\ipc_commands.rs` (dòng `2276–2367`, `2408–2446`, `2528–2653`)
  - `f:\FL studio\toneforge\crates\daw-plugin-host\src\vst3_host.rs`, `clap_host.rs`, `hwnd_window.rs`
- **Cơ chế gây lỗi:**
  1. **Hai thực thể plugin cô lập khi mở `HWND`:** `ExternalPluginGuiWorker` gọi lại `Vst3HostedPluginNode::load` / `ClapHostedPluginNode::load` để tạo một **thực thể plugin thứ hai hoàn toàn độc lập** chỉ để mở cửa sổ `IPlugView` / `clap_plugin_gui_t`, rồi định kỳ đọc `get_parameter_normalized` gửi sang thực thể RT! Do đó:
     - Mọi tham số đã chỉnh trước khi mở cửa sổ GUI không được nạp vào cửa sổ GUI lúc mở.
     - Nếu người dùng chọn Preset nội bộ ngay bên trong cửa sổ của plugin thương mại, nạp sample/wavetable (lưu trong state chunk `IBStream` / `clap_plugin_state_t` chứ không phơi ra tham số đơn lẻ), thực thể trên luồng RT hoàn toàn không biết gì!
     - Đồng hồ đo tín hiệu / phổ bên trong cửa sổ plugin không nhận được dòng âm thanh đang chạy trên luồng RT.
  2. **Điểm mù ở chế độ Nhạc cụ (`as_channel_instrument == true`):**
     - Trong `ipc_commands.rs:2276` và `2334`, khi nạp VSTi/CLAPi làm nhạc cụ kênh (`ChannelGenerator`, `id = 200 + track_idx`), code **bỏ qua việc ghi vào `ToneForgeProjectState`** → Lưu dự án `.tfproj`, Hoàn tác `Undo/Redo`, và `OfflineRenderEngine` làm mất sạch mọi nhạc cụ VSTi/CLAPi!
     - Trong `cmd_set_external_plugin_param_rt` (`ipc_commands.rs:2420`), lệnh `EngineCommand::SetPluginParamNormalized` chỉ tìm trong các khe hiệu ứng `slots[slot_idx]`, không bao giờ chạm tới `ChannelGenerator`!
- **Yêu cầu sửa chuẩn phòng thu:**
  1. Kiến trúc chuẩn VST3/CLAP: Cho phép gắn `IPlugView` / `clap_plugin_gui_t` trực tiếp vào đúng thực thể plugin đang sống (tách bạch con trỏ giao diện luồng chính STA và hàm `process()` trên luồng RT theo đúng đặc tả thread-safety của Steinberg VST3 `IEditController` + `IComponentHandler::performEdit` qua hàng đợi lock-free `rtrb`, và `clap_host_t::request_callback` trên main thread). Trong trường hợp một plugin VST3 tách rời `IComponent` và `IEditController` không chia sẻ bộ nhớ, phải nối `IConnectionPoint` và đồng bộ đầy đủ luồng `IBStream` hai chiều khi khởi tạo/thay đổi state.
  2. Bổ sung trường lưu cấu hình nhạc cụ kênh (`channel_instrument: Option<HostedExternalPluginStateDto>`) vào từng rãnh/kênh trong `ToneForgeProjectState`, hỗ trợ điều khiển tham số, lưu/nạp state nhị phân Base64, `Undo/Redo`, và khởi tạo lại đầy đủ trong `OfflineRenderEngine`.

---

### `BUG-05` [HIGH — `v3` Hardware Capture Clocking] Đóng Đinh Tần Số Đầu Vào `48,000.0 Hz` Trên `AdaptiveDllDriftResampler` Khi Mở Micro Phần Cứng
- **Vị trí mã nguồn:**
  - `f:\FL studio\toneforge\shell\src-tauri\src\main.rs` (dòng `455–456`)
  - `f:\FL studio\toneforge\crates\daw-core-graph\src\audio_io\wasapi_capture_duplex.rs` (dòng `877–878`)
- **Cơ chế gây lỗi:**
  - `WasapiCaptureStream::spawn` mở thiết bị thu âm theo định dạng `GetMixFormat` thực tế của Windows (`44,100 Hz`, `48,000 Hz`, `96,000 Hz`, hoặc `16,000 Hz` với headset USB/Bluetooth).
  - Thế nhưng ngay dòng kế tiếp (`main.rs:456`), code khởi tạo `AdaptiveDllDriftResampler::new(48_000.0, engine_sr)` với giả định cứng đầu vào luôn là `48,000.0 Hz`! Vì vòng khóa trễ DLL chỉ cho phép bù trôi xung nhịp tinh `±500 ppm` quanh tỷ lệ danh định `input_sr / output_sr`, bất kỳ micro nào chạy ở `44.1 kHz` hay `96 kHz` sẽ bị lệch tông nghiêm trọng và tràn/cạn bộ đệm liên tục!
- **Yêu cầu sửa chuẩn phòng thu:**
  - Cho `WasapiCaptureStream::spawn` báo cáo đồng bộ (qua kênh handshake khởi tạo một lần hoặc đọc trước `mix_sample_rate_hz` của endpoint) đúng tần số lấy mẫu thực tế `actual_capture_sr` và số kênh mà phần cứng vừa thương lượng thành công, rồi mới khởi tạo `AdaptiveDllDriftResampler::new(actual_capture_sr as f64, engine_sr as f64)`.

---

### `BUG-06` [MEDIUM — `v3` Driver Lifecycle] `ActiveHardwareAudioDriver::shutdown` Bỏ Quên Nhánh `Asio` Khi Thoát Ứng Dụng
- **Vị trí mã nguồn:**
  - `f:\FL studio\toneforge\shell\src-tauri\src\main.rs` (dòng `110–114`)
- **Cơ chế gây lỗi:**
  - Phương thức `ActiveHardwareAudioDriver::shutdown(&mut self)` chỉ xử lý `if let Self::Wasapi(s) = self { s.shutdown(); }` và bỏ mặc biến thể `Self::Asio(s)` khi nhận sự kiện `RunEvent::Exit`, khiến một số driver ASIO thương mại không được gọi `IASIO::stop()` / `disposeBuffers()` tường minh trước khi tiến trình kết thúc.
- **Yêu cầu sửa chuẩn phòng thu:**
  - Gọi `s.shutdown()` tường minh cho cả `Self::Wasapi(s)` và `Self::Asio(s)`, đồng thời dọn dẹp an toàn luồng `WasapiCaptureStream`, `MidiHardwareInputStream`, `DiskTrackRecorder` và `ExternalPluginGuiWorker` khi thoát app hoặc chuyển đổi driver.

---

### `BUG-07` [CRITICAL — Di Sản `v1/v2` → `v3`] `OfflineRenderEngine` Và `ToneForgeProjectState` Làm Rơi Rớt Giọng Hát Đã Chỉnh `NewTone`, Sơ Đồ `Patcher`, `Preset FL26`, Và Ghi Đè Kênh `1..4` Bằng Demo Stems Cũ
- **Vị trí mã nguồn:**
  - `f:\FL studio\toneforge\shell\src-tauri\src\main.rs`:
    - `cmd_execute_offline_render` (khoảng dòng `1881–1961`)
    - `cmd_vocal_commit_pitch_edits` (khoảng dòng `1020–1061`)
    - `cmd_update_patcher_graph` (khoảng dòng `2185–2205`)
    - `cmd_load_fl26_fx_preset` (khoảng dòng `2160–2180`)
- **Cơ chế gây lỗi (Di sản tích tụ từ `v1` sang `v3`):**
  1. **Offline Render lệch hoàn toàn với bản phối Live RT:** Trong `cmd_execute_offline_render`, hệ thống tạo một `AudioStreamController` mới, gọi `apply_project_state(snap_ps)`, rồi ngay lập tức chạy đoạn code di sản từ thời `v1` (`dòng 1914–1952`) **đè lên các kênh `1..=4`** bằng `bounced_stems`, `custom_pcm`, hoặc tệ hơn là `build_demo_harmonic_stem()`! Nó bỏ quên hoàn toàn:
     - Giọng hát ca sĩ vừa chỉnh cao độ tỉ mỉ bằng `NewTone` qua `cmd_vocal_commit_pitch_edits` (vì `cmd_vocal_commit_pitch_edits` chỉ đẩy âm thanh xuống luồng RT qua `cmd_bounce_stems_to_mixer_rt` trên `[dst_trk]` mà **không lưu PCM đã chỉnh sửa vào kho tài nguyên của dự án**!).
     - Các đoạn thu âm `.wav` từ `DiskTrackRecorder` (`BUG-01`).
     - Các Audio Clip đã cắt ghép / giãn thời gian trên Playlist (`BUG-02`).
     - Các nhạc cụ `VST3 / CLAP` (`BUG-04`).
  2. **Chỉnh sửa `Patcher` và `FL26 Preset` không lưu vào `ToneForgeProjectState`:**
     - `cmd_update_patcher_graph` cập nhật đồ thị Patcher trên luồng RT nhưng **không ghi `PatcherGraphStateDto` mới vào `state.project_state`** (cũng không ghi nhận snapshot Undo!).
     - `cmd_load_fl26_fx_preset` nạp preset cho `LuxeVerb` / `GrossBeat` / `Pro-Q 4`... xuống luồng RT nhưng **không đồng bộ tham số preset đó vào `state.project_state`**!
     - Hệ quả: Hễ bấm `Undo` hoặc bấm `Offline Render` xuất nhạc ra file `.wav/.flac`, mọi công sức nối dây Patcher, chọn Preset FL26 và chỉnh phô NewTone đều biến mất!
- **Yêu cầu sửa chuẩn phòng thu:**
  1. Thiết lập kiến trúc **Đồng Nhất Tuyệt Đối Giữa Live RT và Offline Render (`Live-Offline Engine Graph Equivalence`)**: viết một hàm dựng trạng thái thống nhất nạp chính xác 100% `ToneForgeProjectState` + `AudioAssetRegistry` (bao gồm Custom Audio, UVR5 Stems, NewTone Committed Stems, Recorded Takes, Stretched Clips, Sequencer Patterns, Automation, Patcher DAG, FL26 Presets, và External VST3/CLAP FX + Instruments). Chỉ nạp `build_demo_harmonic_stem` khi và chỉ khi dự án thực sự đang ở chế độ Demo trống chưa có bất kỳ nốt MIDI hay tài nguyên âm thanh nào!
  2. Cập nhật đầy đủ `state.project_state` + `record_undo_snapshot` trong `cmd_update_patcher_graph`, `cmd_load_fl26_fx_preset`, `cmd_vocal_commit_pitch_edits`, và rà soát toàn bộ các `#[tauri::command]` còn lại trong `main.rs` để đảm bảo **0 lệnh biến đổi âm thanh nào lọt lưới `project_state`**.

---

### `BUG-08` [HIGH — Di Sản `v1/v2` UI & Mixer] Chuỗi Khe Hiệu Ứng Trên `MixerRack502.svelte` Thiếu Điều Khiển Thực Thụ (Bypass / Wet-Dry / Xóa / Đổi Chỗ) & Chưa Phản Ánh Đầy Đủ `ToneForgeProjectState`
- **Vị trí mã nguồn:**
  - `f:\FL studio\toneforge\ui-svelte5-wgpu\src\lib\components\MixerRack502.svelte` (khoảng dòng `120–165`)
  - `f:\FL studio\toneforge\crates\daw-tauri-host\src\ipc_commands.rs` & `project_state.rs`
- **Cơ chế gây lỗi:**
  - Trên giao diện `MixerRack502.svelte`, danh sách `selectedTrack.slots` đang hiển thị nhãn tĩnh sơ khai từ thời `v1`, thiếu nút Bật/Tắt (`Bypass`), thanh kéo độ trộn song song (`Wet/Dry Mix`), nút Xóa plugin khỏi khe (`Remove Slot`), hoặc nạp nhanh plugin nội bộ (`Pro-Q 4`, `Soothe3`, `Ozone 12`, `LuxeVerb`, `GrossBeat`, `Pitcher`, `Patcher`) vào bất kỳ khe nào trong 10 khe của bất kỳ kênh nào trong `502` kênh với đồng bộ hai chiều xuống `MixerEngine` + `ToneForgeProjectState` + `Undo`.
- **Yêu cầu sửa chuẩn phòng thu:**
  - Nâng cấp trọn vẹn quản lý `10` FX Slots trên `MixerRack502.svelte` và tầng IPC tương ứng: cho phép Nạp / Bật-Tắt (`Bypass`) / Chỉnh `Wet Mix` / Xóa khe FX trên mọi kênh Mixer, đồng bộ hai chiều với `ToneForgeProjectState`, `UndoHistoryManager`, `MixerEngine` (qua `RtGraphMutation` + `RtGc` `0` RT alloc) và `OfflineRenderEngine`.

---

### `BUG-09` [MEDIUM-HIGH — Di Sản `v1/v2` Sample Rate] Truyền Động Tần Số Lấy Mẫu Thực Tế (`44.1 / 48 / 88.2 / 96 / 192 kHz`) Thay Vì Mặc Định Tĩnh `48,000.0 Hz`
- **Vị trí mã nguồn:**
  - `f:\FL studio\toneforge\shell\src-tauri\src\main.rs` (khoảng dòng `2224` lúc khởi tạo `SharedDawController`, và các hàm đổi thiết bị `cmd_select_audio_output_device` / `cmd_select_asio_driver`)
- **Cơ chế gây lỗi:**
  - Khi mở ứng dụng hoặc khi người dùng đổi sang DAC/Audio Interface chạy ở `44,100 Hz` hoặc `96,000 Hz`, cần kiểm tra chặt chẽ xem tất cả các bộ phận phụ thuộc tần số lấy mẫu — bao gồm `PlaylistSequencer` (`samples_per_tick`), bộ lọc `Pro-Q 4` (`Natural Phase` / ZDF coefficients), `Soothe3`, `Ozone 12` (chọn lại bảng `exciter-*-<sr>.dat` tương ứng `44100/48000/88200/96000/192000`), `LuxeVerb` (độ dài đường trễ FDN theo giây), `Pitcher` (`pYIN`), và `DiskTrackRecorder` (header `.wav`) — đã nhận lệnh `prepare(new_sample_rate)` đồng bộ 100% chưa!
- **Yêu cầu sửa chuẩn phòng thu:**
  - Đảm bảo mỗi khi phần cứng `WASAPI` hoặc `ASIO` thương lượng tần số lấy mẫu mới `actual_sr != current_engine_sr`, hệ thống cập nhật `engine_sample_rate = actual_sr` và xây dựng lại / `prepare(actual_sr)` toàn bộ đồ thị `MixerEngine` từ `ToneForgeProjectState` (tráo nóng an toàn không rò rỉ bộ nhớ) + resample các tài nguyên trong `AudioAssetRegistry` khớp đúng `actual_sr` để cao độ và tốc độ BPM (`samples_per_tick`) chính xác tuyệt đối ở mọi tần số `44.1k..192k`!

---

### `BUG-10` [HIGH — `v2/v3` Frontend Diagnostics & IPC Schema] Các Khối `catch {}` Nuốt Lỗi Âm Thầm Trong `daw_master_store.svelte.ts` Che Giấu Lỗi Gọi Lệnh Tauri
- **Vị trí mã nguồn:**
  - `f:\FL studio\toneforge\ui-svelte5-wgpu\src\lib\state\daw_master_store.svelte.ts` (rải rác ở các dòng `~3383, 3415, 3464, 3657, 3788, 3856, 3895, 3919` và các hàm `invoke(...)` khác từ thời `v1/v2`)
- **Cơ chế gây lỗi:**
  - Nhiều hàm gọi `try { await invoke('cmd_...', { ... }); } catch { /* fallback mock */ }` không phân biệt giữa **"đang mở trên trình duyệt web thuần (không có Tauri)"** với **"đang chạy trong Tauri nhưng gọi sai tên tham số `camelCase`/`snake_case` hoặc lỗi runtime phía Rust"**! Khi có lỗi thật trong Tauri, khối `catch` lặng lẽ nuốt mất thông báo lỗi và điền mảng dữ liệu giả lập của trình duyệt vào UI, khiến người dùng tưởng tính năng chạy nhưng thực chất backend Rust không thực thi!
- **Yêu cầu sửa chuẩn phòng thu:**
  1. Kiểm tra tường minh `isTauriRuntime()` (`typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window`):
     - Nếu `isTauriRuntime() === false`: dùng nhánh mô phỏng trình duyệt.
     - Nếu `isTauriRuntime() === true`: **TUYỆT ĐỐI KHÔNG rơi về dữ liệu giả lập khi `invoke(...)` ném ngoại lệ!** Phải ghi nhận lỗi rõ ràng vào thanh trạng thái chẩn đoán (`lastEngineError` / toast cảnh báo kỹ thuật trên giao diện) và `console.error` để không một lỗi IPC nào được phép ẩn mình.
  2. Kiểm toán đối chiếu 1-đối-1 toàn bộ chữ ký `invoke('cmd_...', payload)` trong `ui-svelte5-wgpu` với chữ ký hàm `#[tauri::command]` trong `shell/src-tauri/src/main.rs` (quy tắc chuyển đổi `snake_case` của tham số Rust sang `camelCase` trong payload JS của Tauri v2, cùng tên trường struct trả về).

---

## PHẦN II — 5 HẠNG MỤC TINH CHỈNH ĐIỂM BIÊN KIẾN TRÚC & VẬT LÝ (`EDGE-01` → `EDGE-05`)

Song song với 10 lỗi tích hợp ở Phần I, hãy triển khai nâng cấp luôn 5 điểm biên kỹ thuật đã được nhận diện tại Mục 5 của `f:\FL studio\research_notes\10_toneforge_v3_acceptance_report.md` để đưa ứng dụng lên độ chín thương mại cao nhất:

1. **`EDGE-01` (`daw-plugin-host` — HiDPI Scaling & Multi-Bus VST3 Arrangements):**
   - Triển khai COM interface `IPlugViewContentScaleSupport` (`F2AEE70D-0E6E-485B-9D46-0907B968D331` / chuẩn Steinberg SDK) đồng bộ tỷ lệ縮 phóng DPI thực tế của màn hình Windows (`GetDpiForWindow(hwnd) as f32 / 96.0`) để cửa sổ plugin thương mại sắc nét, đúng khung kích thước trên màn hình `125% / 150% / 200%`.
   - Mở rộng `IAudioProcessor::setBusArrangements` linh hoạt nhận diện cấu hình Main Stereo + Aux Sidechain Stereo + các bus đầu ra phụ mà không ép cứng duy nhất một cấu hình cố định.
2. **`EDGE-02` (`dsp-pitch-time` — Bảo Toàn Formant Giọng Nữ Cao `f0 > 320 Hz` & Cửa Sổ Bảo Vệ Phụ Âm Bật Hơi `Plosive Transient Bypass`):**
   - Khi `f0 > 320 Hz` (quãng Soprano nữ cao khiến khoảng cách hài âm thưa hơn bề rộng cộng hưởng `F1`), kết hợp ràng buộc bao hình tiên nghiệm đường thanh đạo sinh học (`Quasi-Harmonic / LPC-Regularized Formant Envelope Floor`) để ngăn suy giảm năng lượng vùng `280–600 Hz` khi dịch quãng lớn.
   - Bổ sung bộ phát hiện đạo hàm năng lượng tần số cao / độ phẳng phổ (`Spectral Flatness + Crest Flux`) để tự động chuyển mượt sang chế độ giữ nguyên pha thời gian thực (`Transient Bypass Crossfade` `5–8 ms`) ngay tại đầu các phụ âm bật hơi (`/p/`, `/t/`, `/k/`), triệt tiêu hiện tượng mờ pha `10–15 ms` ngay cả khi dịch tới `±12 semitones`.
3. **`EDGE-03` (`asio_hardware_driver.rs` — Cửa Sổ `Message-Only HWND` Riêng Trên Luồng STA & Chu Kỳ `ASIOOutputReady()` Ở Buffer `32–64 Samples`):**
   - Tạo một cửa sổ ẩn chuyên dụng `CreateWindowExW(..., HWND_MESSAGE, ...)` ngay trên đúng luồng COM STA khởi tạo driver ASIO và truyền `sys_handle` chuẩn xác đó vào `IASIO::init(sys_handle)` kèm vòng bơm thông điệp nhẹ `PeekMessageW` khi nghỉ.
   - Kiểm tra cờ hỗ trợ của `IASIO::outputReady()` ngay sau khi hoàn tất mỗi callback `bufferSwitch` / `bufferSwitchTimeInfo` để gọi `outputReady()` tức thời, tối ưu độ ổn định phần cứng ở kích thước đệm cực thấp `32–64 samples`.
4. **`EDGE-04` (`PlaylistPianoRollEditor.svelte` & `VocalPitchEditor.svelte` — Ảo Hóa Khung Nhìn / Dựng Hình Tốc Độ Cao Cho `> 2,500` Nốt Nhạc):**
   - Triển khai **cắt tỉa khung nhìn không gian (`Spatial Viewport Culling` theo trục `startTick..endTick` đang hiển thị + lớp Canvas 2D / WebGPU tăng tốc cho các nốt nền & `Ghost Notes`)**, chỉ dựng phần tử tương tác cho các nốt nằm trong cửa sổ nhìn thấy, đảm bảo tốc độ cuộn và kéo thả giữ vững `120 FPS` mượt mà ngay cả khi dự án chứa `5,000+` nốt MIDI giao hưởng hoặc `1,000+` thanh `NewTone`!
5. **`EDGE-05` (`dsp-ozone12-mastering/src/codec_preview.rs` — Chuyển Đổi Khối Ngắn/Dài Thích Ứng `Adaptive Window Switching` & Dự Trữ Bit `Bit-Reservoir`):**
   - Nâng cấp `CodecPreview` với cơ chế phát hiện xung đột ngột (`Transient Attack Detector` theo chuẩn tâm sinh lý học MPEG/AAC/Opus) để chuyển động giữa cửa sổ dài (`512-sample MDCT` cho đoạn ngân hài âm độ phân giải tần số cao) và dãy cửa sổ ngắn (`4 × 128-sample MDCT` với cửa sổ chuyển tiếp `Start/Stop Bridge Windows` vẫn thỏa mãn điều kiện khử răng cưa miền thời gian `TDAC Princen-Bradley` hoàn hảo `-240 dBFS` ở chế độ PCM Bypass!) kèm mô hình ngân sách bit `Bit Reservoir`, giúp tái tạo cực sát hành vi nén thực tế trên trống/nhạc cụ gõ của MP3/AAC/Opus.

---

## PHẦN III — CỔNG KIỂM ĐỊNH ĐẦU-CUỐI BẮT BUỘC BỔ SUNG (`GATES 11 & 12`)

Ngoài việc giữ vững **`100% PASS` trên toàn bộ `Gates 1–10` hiện có** (`target/gate_metrics/gate1.json` .. `gate10.json`), bạn bắt buộc phải viết mới bộ test tích hợp cấp Host/Workflow thực chiến và xuất kết quả đo đạc ra `target/gate_metrics/gate11.json` & `target/gate_metrics/gate12.json`:

### `Gate 11` — Kiểm Định Khép Kín Toàn Vòng Đời Người Dùng & Đồng Nhất Live RT ↔ Offline Render (`Full-Stack Host E2E Parity Gate`)
Kiểm chứng tự động trên chuỗi điều phối Host thực tế (không chỉ gọi rời từng struct DSP):
1. **Kịch bản 11.1 (Record → Comping → Playback → Save/Load `.tfproj`):** Bơm luồng âm thanh thu thử (ở cả tần số lệch như `44,100 Hz` qua `AdaptiveDllDriftResampler` sang engine `48,000 Hz`) qua `DiskTrackRecorder` tạo `2` Take lặp → kết thúc thu → nạp tự động vào `AudioAssetRegistry` & `AudioStemClipPlayerNode` → chạy `MixerEngine` xác nhận ngõ ra phát đúng sóng âm của Take được chọn (`RMS > -20 dBFS`, tương quan dạng sóng `>= 0.99`, không phải im lặng!).
2. **Kịch bản 11.2 (Playlist Audio Clip Time-Stretch `1.35x` & Pitch `+4.0 st` Không Bị Varispeed):** Áp dụng `stretch_ratio = 1.35` với `pitch_semitones = 0.0` lên một Audio Clip phát nốt chuẩn `440 Hz` + tiếng trống gõ nhịp trên Playlist → đo tần số cơ bản ngõ ra từ `AudioStemClipPlayerNode` phải giữ nguyên `440.0 ± 1.5 Hz` (chứng minh tuyệt đối không bị tụt tông kiểu băng từ xuống `325.9 Hz`!), đồng thời cắt đôi `Razor Slice` + `Fade Out` phản ánh chính xác trên ngõ ra RT.
3. **Kịch bản 11.3 (Live RT Mix == Offline Render Mix Parity):** Dựng một dự án phức hợp gồm: (a) 1 Take thu âm thật, (b) 1 đoạn giọng hát đã commit qua `NewTone` (`+3 st`), (c) 1 nhạc cụ `test_vst3_clap_plugin.dll` ở chế độ `ChannelGenerator` chơi hợp âm MIDI, (d) sơ đồ dây `Patcher` tùy chỉnh + `FL26 Preset` + `Pro-Q 4` + `Soothe3` + `Ozone 12` → sau đó thực hiện `Undo` rồi `Redo` (kiểm tra trả về đầy đủ `ToneForgeProjectState` khớp 100%) → cuối cùng so sánh năng lượng phổ / LUFS giữa chạy `MixerEngine` trực tiếp và chạy `OfflineRenderEngine`: sai lệch tích hợp `|LUFS_live - LUFS_offline| <= 0.15 LU` và không một rãnh nào bị mất tiếng hay bị đè bởi demo stem!

### `Gate 12` — Kiểm Định Hợp Đồng Schema IPC Rust ↔ TypeScript & Hiệu Năng Khung Nhìn `5,000` Nốt (`UI-Host Contract & High-Density Gate`)
1. **Kịch bản 12.1 (100% IPC Command & DTO Field Parity):** Viết bài kiểm tra tự động quét mã nguồn `ui-svelte5-wgpu/src` tìm toàn bộ lời gọi `invoke('cmd_...')` và kiểm chứng:
   - Mọi tên lệnh `cmd_*` đều được đăng ký trong `tauri::generate_handler![...]` tại `shell/src-tauri/src/main.rs`.
   - Tên trường JSON của `UndoRedoStatusDto` và các DTO cốt lõi khớp 100% giữa bộ giải mã TypeScript và `serde_json::to_value(...)` phía Rust.
   - Xác nhận `0` khối `catch` nuốt lỗi âm thầm không kiểm tra `isTauriRuntime()` trong `daw_master_store.svelte.ts`.
2. **Kịch bản 12.2 (High-Density Sequencer & Soprano `+12 st` Edge Verification):**
   - Kiểm chứng xử lý `5,000` nốt Piano Roll trên `PlaylistSequencer` (`0` RT alloc, thời gian xử lý mỗi block `< 50 µs`) và thuật toán lọc khung nhìn phía UI.
   - Kiểm chứng `FormantPitchShifter` trên tín hiệu giọng Soprano cao (`f0 = 400 Hz`) kèm phụ âm bật hơi đầu câu khi dịch `+10.0 semitones`: giữ nguyên độ sắc nét đầu âm phụ âm (`Transient Crest Retention >= 85%`) và không bị sụt hố năng lượng vùng `F1` (`300–600 Hz`).
   - Kiểm chứng `CodecPreview` với `Adaptive Short/Long Window Switching`: `PCM Bypass Null <= -120 dBFS` (hoàn nguyên TDAC hoàn hảo ngay cả khi chuyển khối ngắn/dài!) và giảm nhiễu tiền vang (`pre-echo`) rõ rệt trước xung trống sắc.

---

## PHẦN IV — YÊU CẦU BÁO CÁO NGHIỆM THU CUỐI CÙNG (`13_claude_master_completion_report.md`)

Sau khi hoàn tất toàn bộ Pha 1 → Pha 4, hãy trình bày rõ ràng trong `f:\FL studio\research_notes\13_claude_master_completion_report.md`:
1. **Bảng Đối Chiếu Minh Bạch 3 Cột:**
   - Cột 1: Những vấn đề Claude tự tìm ra độc lập ở Pha 1 (`12_claude_independent_findings.md`) trùng khớp với Sổ cái niêm phong (`11b`).
   - Cột 2: Những vấn đề **mới** do chính Claude phát hiện thêm ngoài `11b` (xuyên suốt từ `v1`, `v2` đến `v3`).
   - Cột 3: Những vấn đề từ `11b` + `10` được bổ sung vào sau khi mở niêm phong ở Pha 2.
2. **Chi Tiết Phẫu Thuật Kỹ Thuật Cho Từng Hạng Mục** (liệt kê rõ file, dòng code, thuật toán đã sửa).
3. **Bảng Kết Quả Thực Đo Định Lượng Trọn Bộ `Gates 1–12`** (`--release`, trích xuất từ `gate1.json` .. `gate12.json`), kích thước & mã băm `SHA-256` mới của `ToneForge_Studio.exe`, `toneforge-plugin-scanner.exe`, và `ui-svelte5-wgpu/dist`.
