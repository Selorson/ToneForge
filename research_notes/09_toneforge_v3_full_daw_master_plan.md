# 09 — Kế Hoạch Tổng Thể & Bộ Tiêu Chí Nghiệm Thu Định Lượng ToneForge Studio v3 (Full FL Studio 26 & Plugin Parity)

**Phiên bản kế hoạch:** `v3.0-master-blueprint`  
**Trạng thái nền tảng xuất phát (`ToneForge Studio v2`):** `112/112` bài test `--release` `PASSED` • `6/6` Cổng Kiểm Định Âm Học (`Gates 1–6`) `PASSED` • `ToneForge_Studio.exe` chạy thực tế trên phần cứng `WASAPI Shared / Exclusive` (`0` Mutex, `0` alloc, `0` free trên luồng âm thanh).  
**Mục tiêu tối hậu của `ToneForge Studio v3`:** Khắc phục triệt để **100% khoảng trống kỹ thuật còn lại** để ToneForge đủ năng lực **thay thế hoàn toàn FL Studio 26** trong toàn bộ vòng đời sản xuất âm nhạc (Sáng tác MIDI/Piano Roll → Thu âm Vocal/Nhạc cụ độ trễ thấp qua ASIO/WASAPI → Nạp nhạc cụ & hiệu ứng hãng thứ 3 `VST3`/`CLAP` → Chỉnh phô giọng hát `Pitcher`/`NewTone` & Co giãn nhịp Audio Clip → Cắt ghép Arrangement đa rãnh → Mixing & AI Mastering chuẩn 100% `Pro-Q 4` + `Soothe3` + `Ozone 12`).

---

## 1. Kỷ Luật Kỹ Thuật & Các Bất Biến Kiến Trúc Cứng (Bắt Buộc Tuân Thủ Ở Cửa Sổ `/goal` Mới)

1. **Không khoan nhượng với code giả lập (Zero Toy Code):**
   - Nghiêm cấm viết stub giả vờ nạp `VST3`/`CLAP`, nghiêm cấm dịch cao độ kiểu đổi tốc độ phát đơn thuần gây méo giọng "sóc chuột" (*chipmunk*), nghiêm cấm co giãn thời gian làm bệt tiếng trống (*transient smearing*), nghiêm cấm giả lập Codec Preview chỉ bằng bộ lọc thông thấp đơn giản.
2. **Bảo toàn 100% Bất Biến Luồng Thời Gian Thực (`0` Mutex • `0` `HeapAlloc` • `0` `HeapFree`/`drop()`):**
   - Luồng âm thanh phần cứng (`WASAPI` hoặc `ASIO`) tiếp tục **sở hữu độc quyền `Box<MixerEngine>`**.
   - Mọi thay đổi đồ thị, nạp plugin ngoài, cập nhật tham số, nạp vùng clip kéo giãn, hay định tuyến luồng thu âm phải đi qua `rtrb::RingBuffer<RtGraphMutation>` / `rtrb::RingBuffer<EngineCommand>`.
   - Mọi đối tượng có bộ nhớ động bị thay thế phải được đẩy qua `rtrb::RingBuffer<RetiredRtResource>` sang luồng dọn rác nền `toneforge-rt-gc`.
3. **Bảo toàn kích thước khung bộ nhớ chia sẻ `VisualTelemetryFrame == 131,072` bytes (`128 KB`):**
   - Tuyệt đối không làm lệch bất kỳ hằng số `OFFSET_*` nào trong `crates/daw-core-graph/src/telemetry_shm.rs`.
   - Tận dụng chính xác các trường đã thiết kế sẵn trong `128 KB`:
     - `ProQ4SpectrumFrame::sc_db` (`[f32; 1024]`) và `ProQ4SpectrumFrame::collision_mask` (`[f32; 1024]`) cho phổ Sidechain & mặt nạ va chạm tần số liên rãnh.
     - `Ozone12MasterTelemetry::exciter_drive_gr[16..32]` cho độ nén dải trầm `Low End Focus`, mức đỉnh thực hậu giải mã `Codec Preview True Peak`, và năng lượng nhiễu nén `Codec Artifact RMS`.
     - `TransportAndMasterHeader::reserved[0]` cho cờ trạng thái thu âm (`ARM` / `REC` / `XRUN`) và mức đỉnh tín hiệu đầu vào phần cứng.
4. **Độc lập hoàn toàn về thư viện (Zero External Network Dependency Risk):**
   - Chỉ sử dụng các crate đã khóa sẵn trong `Cargo.lock` của workspace (`windows-sys = "0.61"`, `rtrb`, `triple_buffer`, `parking_lot`, `wide`, `bytemuck`, `rustfft`, `realfft`, `rubato`, `symphonia`, `rfd`, `serde`, `serde_json`, `rayon`).
   - Mở rộng thêm các feature sẵn có của `windows-sys = "0.61"` trong `Cargo.toml` gốc: `"Win32_Foundation"`, `"Win32_UI_WindowsAndMessaging"`, `"Win32_System_Registry"`, `"Win32_Graphics_Gdi"`. Tự định nghĩa chuẩn xác các struct C ABI của `CLAP 1.2` và COM vtable của `Steinberg VST3` / `Steinberg ASIO` bằng Rust thuần.
5. **Ranh giới phạm vi cứng:**
   - **Giữ nguyên 100% không chạm tới** thư mục di sản `ui-svelte-webgpu`; mọi nâng cấp giao diện thực hiện trên `ui-svelte5-wgpu`.
   - **Hoãn 100% tính năng MV / Video** (0 dòng code video/canvas MV).
   - **Tắt tiếng phần cứng khi chạy test tự động:** Mọi bài test chạm tới `WasapiHardwareStream` hoặc `AsioHardwareStream` bắt buộc gọi `set_volume_and_mute(0.0, true)` ngay khi khởi tạo để không phát âm thanh ra loa ngoài của người dùng.

---

## 2. Kiến Trúc Tổng Thể `ToneForge Studio v3`

```mermaid
flowchart TB
    subgraph HW_IN["Đầu Vào Phần Cứng Độ Trễ Thấp (Chặng C)"]
        ASIO_IN["Steinberg ASIO Duplex (`IASIO` COM)<br/>`Realtek ASIO` / Soundcard Rời"]
        WASAPI_IN["WASAPI Capture (`eCapture` Event-Driven)<br/>+ Adaptive DLL Drift Resampler"]
        MIDI_IN["Win32 MIDI In (`midiInOpen`)<br/>Bàn Phím MIDI / Pad Controller"]
    end

    subgraph HOST_UI["Svelte 5 + WebGPU 120 FPS UI & Host (`daw-tauri-host`)"]
        UNDO["Global Undo/Redo (`64` bước)<br/>`Ctrl+Z` / `Ctrl+Y`"]
        ARR["Pro Playlist & Piano Roll<br/>Slice • Fade • Stretch • Ghost Notes • Strum/Arp"]
        NEWTONE["Vocal Pitch & Time (`NewTone`)<br/>pYIN + Cepstral Formant Blob Editor"]
        PLUG_WIN["Win32 `HWND` Floating Plugin GUI<br/>(`IPlugView` / `clap_plugin_gui`)"]
    end

    subgraph RT_CORE["Luồng Âm Thanh Thời Gian Thực (`Box<MixerEngine>` Độc Quyền — 0 Mutex / 0 Alloc / 0 Free)"]
        SEQ["Hybrid PPQ 96–960 Sequencer<br/>+ `TransientLockedTimeStretcher` Clips"]
        DAG["502-Track Topological Mixer DAG<br/>+ 5-Point APDC (Kể Cả VST3/CLAP & Input)"]
        PITCHER["`AutoPitchCorrectorNode` (`Pitcher` RT)<br/>Formant-Preserving TD-PSOLA/WSOLA"]
        EXT_PLUG["`Vst3HostedPluginNode` & `ClapHostedPluginNode`<br/>Audio Stereo + Sample-Accurate MIDI + Automation"]
        PROQ4["Pro-Q 4 (`100%` Parity)<br/>EQ Sketch + Cross-Track Collision Bus"]
        SOOTHE3["Soothe3 (`100%` Parity)<br/>External Spectral Sidechain Ducking"]
        OZONE12["Ozone 12 (`100%` Parity)<br/>Low End Focus + Codec Preview (MP3/AAC/Opus)"]
    end

    subgraph BG_WORKERS["Luồng Nền Không Khóa (Off-RT Workers)"]
        REC_DISK["`DiskTrackRecorder` (`.wav` 32-bit Float)<br/>+ Loop Take Comping & Bù Trễ Khứ Hồi"]
        GC["`toneforge-rt-gc`<br/>Thu Gom Bộ Nhớ Ngoài Luồng RT"]
        SCAN["`toneforge-plugin-scanner`<br/>Quét `.vst3`/`.clap` Cách Ly Chống Sập"]
    end

    ASIO_IN & WASAPI_IN -->|"rtrb StereoFrame"| DAG
    MIDI_IN -->|"rtrb SequencerVoiceEvent"| SEQ
    UNDO & ARR & NEWTONE -->|"rtrb RtGraphMutation"| RT_CORE
    SEQ --> DAG
    DAG --> PITCHER & EXT_PLUG & PROQ4 & SOOTHE3 & OZONE12
    DAG -->|"rtrb Record Tap"| REC_DISK
    RT_CORE -->|"RetiredRtResource"| GC
    RT_CORE -->|"128 KB TripleBuffer (120 Hz)"| HOST_UI
    SCAN -.->|"Danh mục JSON an toàn"| HOST_UI
    EXT_PLUG -.->|"Gắn HWND"| PLUG_WIN
```

---

## 3. Lộ Trình Chi Tiết 4 Chặng Kỹ Thuật (Milestones A → B → C → D)

### Chặng A: Hoàn Thiện Quy Trình Sáng Tác Cốt Lõi & Đạt 100% Tính Năng 3 Plugin Nội Bộ
**Mục tiêu:** Biến thao tác biên tập dự án trở nên mượt, an toàn tuyệt đối và lấp kín 100% tính năng còn thiếu của `Pro-Q 4`, `Soothe3`, `Ozone 12`.

1. **Hệ thống Hoàn tác / Làm lại toàn cục (`crates/daw-tauri-host/src/undo_history.rs`):**
   - Thiết kế ngăn xếp lịch sử hai chiều `ProjectUndoHistory` (giới hạn mặc định `64` bước, dùng cơ chế chụp vi sai nhẹ `ToneForgeProjectState` + phát lại chuỗi `RtGraphMutation` tương ứng để đồng bộ tức thì với luồng RT mà không khựng tiếng).
   - Bao phủ toàn bộ thao tác: thêm/xóa/kéo nốt Piano Roll, cắt/dời/fade Audio Clip, chỉnh dải tần EQ, đổi thông số Soothe3/Ozone 12, định tuyến Mixer, nạp/gỡ Plugin, chỉnh nốt Vocal.
   - Tổ hợp phím toàn cục `Ctrl+Z` (Undo) và `Ctrl+Y` / `Ctrl+Shift+Z` (Redo) trên `App.svelte` kèm nhãn hiển thị tên hành động vừa hoàn tác.

2. **Nâng cấp Pro Playlist Arrangement & Piano Roll (`crates/daw-core-graph/src/sequencer_pianoroll.rs` & `PlaylistPianoRollEditor.svelte`):**
   - **Trên Audio / Pattern Playlist:**
     - Mở rộng `PlaylistClip` & `StemClipRegion` với: `source_offset_samples` (*Slip Trim* đầu clip không đổi vị trí phát), `fade_in_samples`, `fade_out_samples` (hàm crossfade công suất đều $\sin(\frac{\pi}{2}t)$ / $\cos(\frac{\pi}{2}t)$ chống sụt áp `-3 dB`), `clip_gain_linear` (`-24 dB .. +12 dB`), `stretch_ratio` (`0.5x .. 2.0x`), `pitch_semitones` (`-12.0 .. +12.0`), và `take_lane_id`.
     - Công cụ **Razor Slice** (cắt đôi clip tại vạch con trỏ với de-click micro-fade `2 ms` tự động), nhân bản nhanh (`Ctrl+B` / kéo giữ `Shift`), chọn vùng nhiều clip (**Box Lasso Selection**).
   - **Trên Piano Roll:**
     - **Multi-Channel Ghost Notes:** Hiển thị bóng mờ nốt của các rãnh/nhạc cụ khác trong cùng Pattern để viết hòa âm/bassline đối chiếu trực quan.
     - Bộ công cụ biến đổi nốt toán học chuẩn trong `sequencer_pianoroll.rs`:
       - `quantize_notes(grid_ticks, strength_0_to_1, quantize_duration)`
       - `strum_notes(time_spread_ticks, velocity_decay, alternate_direction)` (rải hợp âm guitar/piano tự nhiên)
       - `arpeggiate_notes(step_ticks, pattern_mode, octave_range, gate_ratio)`
     - **Làn biểu cảm dưới đáy (Bottom Expression Lane):** Chỉnh trực quan cột `Velocity`, `Pan`, `Fine Pitch (cents)` cho từng nốt hoặc quét chuột hàng loạt.

3. **Đạt 100% Tính năng `FabFilter Pro-Q 4` (`crates/dsp-proq4-eq` & `ProQ4InteractiveAnalyzer.svelte`):**
   - **Cross-Track Instance List & Spectral Collision Detection (`collision_bus.rs`):**
     - Tạo bảng đăng ký phổ liên rãnh không khóa `ProQ4CollisionBus` (`502` khe rãnh × `1,024` bin tần số lưu dạng nguyên tử hoặc `TripleBuffer` nhẹ mỗi khe).
     - Mỗi thực thể `ProQ4Processor` xuất phổ hậu xử lý vào khe `track_id` của mình và đọc phổ của bất kỳ rãnh tham chiếu nào được chọn (ví dụ đang mở EQ rãnh `Bass` chọn so sánh với rãnh `Kick` hoặc quét toàn bộ rãnh đang hoạt động), tính toán vạch đỏ cảnh báo chồng lấn mặt nạ thính giác (**Spectral Masking Collision**) ghi thẳng vào `ProQ4SpectrumFrame::collision_mask`.
   - **EQ Sketch (`eq_sketch_fitter.rs`):**
     - Thuật toán khớp đường cong thông minh `fit_eq_sketch_curve(points_hz_db: &[(f32, f32)], max_bands: usize) -> Vec<SketchFittedBand>`: Phân tích độ dốc biên trái/phải để tự nhận diện `LowCut` / `LowShelf` và `HighCut` / `HighShelf`, sau đó giải lặp cực trị phần dư (*Iterative Peak-Residual Gauss-Newton / Levenberg-Marquardt nhẹ*) để sinh tối đa `8–12` dải `Bell` khớp sát đường chuột người dùng vẽ một nét trên màn hình (`sai số RMS < 0.75 dB`).
   - **Frequency Piano Roll Bar:** Thanh phím đàn Piano chuẩn MIDI (`C0–C8`) nằm dưới đáy phổ Pro-Q 4 giúp ghim tần số dải EQ chính xác vào nốt nhạc gốc của bài hát.

4. **Đạt 100% Tính năng `oeksound Soothe3` (`crates/dsp-soothe3-spectral` & `Soothe3ResonanceView.svelte`):**
   - **External Spectral Sidechain Ducking:**
     - Bổ sung tham số `external_sidechain_enabled` vào `Soothe3Params`.
     - Trong `Soothe3Processor::process_slice`, khi `external_sidechain_enabled == true` và `sidechain_in` là `Some(sc_frames)` (nhận từ hệ thống định tuyến `is_sidechain_only` sẵn có của `MixerEngine`), nhánh phân tích STFT & nhận diện cộng hưởng sẽ đọc `sc_frames` (ví dụ: giọng hát chính) trong khi bộ lọc tích chập phân vùng pha cực tiểu `4,096`-tap áp dụng trực tiếp lên `main_io` (ví dụ: rãnh Nhạc nền / Guitar). Kết quả: chỉ đúng những tần số giọng hát đang phát ra mới bị nén xuống trên nhạc nền!
     - Xuất đồng thời phổ đầu vào chính và phổ Sidechain để giao diện hiển thị 2 lớp phổ chồng nhau.

5. **Đạt 100% Tính năng `iZotope Ozone 12` (`crates/dsp-ozone12-mastering` & `Ozone12MasterChain.svelte`):**
   - **Mô-đun `Low End Focus` (`low_end_focus.rs`):**
     - Bộ phân tần Linkwitz-Riley bậc 4 (`24 dB/oct` bù pha phẳng) tách vùng trầm tùy chỉnh `20 Hz .. 300 Hz`.
     - Tách đường bao nhanh (*Fast Envelope*) và đường bao chậm (*Slow Envelope*) để đo độ tương phản động của dải trầm: chế độ **`Punchy`** làm nổi bật cú đập đầu của Kick và gọn đuôi rền sub; chế độ **`Smooth`** làm đều lực nén giữa Kick và Bass tránh nhấp nhô âm lượng, đi kèm mạch khuếch đại bù giữ nguyên đỉnh (*Peak-Preserving Makeup*) và nút nghe riêng (`Solo Low Band`).
   - **Mô-đun `Codec Preview` (`codec_preview.rs`):**
     - Bộ mô phỏng lượng tử hóa miền tần số **MDCT + Mặt nạ tâm lý âm học Bark-band (Psychoacoustic Masking & Bitpool Quantization Simulator)** thời gian thực không cấp phát bộ nhớ cho 5 cấu hình phát hành: `MP3 128 kbps`, `MP3 320 kbps`, `AAC 128 kbps`, `AAC 256 kbps`, `Opus 128 kbps`.
     - Nút **`Artifact Solo (Delta)`** phát riêng phần sai lệch $e[n] = \hat{x}_{\text{codec}}[n] - x_{\text{pcm}}[n]$ để kỹ sư nghe rõ tiếng xèo pre-echo trên hi-hat/vocal, đồng thời đo **Đỉnh thực `4x` ITU-R BS.1770-4 sau giải mã Codec** để cảnh báo hiện tượng vượt trần `+0.5 .. +1.2 dBTP` khi đăng nhạc lên Spotify/YouTube/TikTok.

---

### Chặng B: Trình Chủ Nạp Plugin Hãng Thứ 3 `VST3` & `CLAP` Thực Thụ (`crates/daw-plugin-host`)
**Mục tiêu:** Mở toang hệ sinh thái hàng chục ngàn nhạc cụ ảo (Serum, Kontakt, Diva, Vital...) và hiệu ứng hãng thứ 3 trên Windows x64, tích hợp liền mạch vào 502 rãnh Mixer & Channel Rack.

1. **Trình chủ `CLAP 1.2` C ABI đầy đủ (`crates/daw-plugin-host/src/clap_host.rs`):**
   - Định nghĩa chuẩn xác `#[repr(C)]` theo đặc tả `CLAP 1.2.x`: `clap_plugin_entry`, `clap_plugin_factory_t`, `clap_host_t`, `clap_plugin_t`, `clap_process_t`, `clap_audio_buffer_t`, `clap_input_events_t`, `clap_output_events_t`, `clap_event_note_t`, `clap_event_midi_t`, `clap_event_param_value_t` (hỗ trợ cả Per-Note Modulation / Note ID), cùng các extension `clap.params`, `clap.audio-ports`, `clap.note-ports`, `clap.latency`, `clap.state`, `clap.gui` (`CLAP_WINDOW_API_WIN32`).
   - Bao gói thành `ClapHostedPluginNode` triển khai trực tiếp `UnifiedDspNode` (cấp phát sẵn toàn bộ mảng con trỏ kênh planar `[[f32; MAX_BLOCK_SIZE]; 2]` và danh sách sự kiện cố định `MAX_EVENTS = 512` trong `prepare()` → **0 cấp phát bộ nhớ trong `process_slice`**).

2. **Trình chủ `Steinberg VST3` COM ABI đầy đủ (`crates/daw-plugin-host/src/vst3_host.rs`):**
   - Định nghĩa chuẩn xác GUID (16-byte COM IID của VST3 `3.7.x`) và vtable `extern "system"` cho: `FUnknown`, `IPluginBase`, `IPluginFactory`, `IPluginFactory2`, `IPluginFactory3`, `IComponent`, `IAudioProcessor`, `IEditController`, `IComponentHandler` (nhận thao tác vặn núm từ giao diện plugin trả ngược về hệ thống Automation/Undo của ToneForge!), `IConnectionPoint`, `IPlugView`, `IBStream` (đọc/ghi state nhị phân vào bộ nhớ RAM), `ProcessData`, `AudioBusBuffers`, `IParameterChanges`, `IParamValueQueue`, `IEventList`.
   - Hỗ trợ cả hai kiến trúc VST3 ngoài đời thực: plugin gộp chung (`SingleComponentEffect` qua `queryInterface`) và plugin tách rời `IComponent` + `IEditController` nối qua `IConnectionPoint`.
   - Bao gói thành `Vst3HostedPluginNode` triển khai `UnifiedDspNode` + `ChannelGenerator` (cho VSTi nhạc cụ ảo), trả `IAudioProcessor::getLatencySamples()` thẳng về bộ bù trễ **5-Point APDC** của `MixerEngine`.

3. **Cửa sổ giao diện gốc Win32 `HWND` (`crates/daw-plugin-host/src/hwnd_window.rs`):**
   - Đăng ký lớp cửa sổ Win32 riêng (`RegisterClassExW`, `CreateWindowExW` với cờ `WS_OVERLAPPEDWINDOW` có đặt parent owner về cửa sổ Tauri chính để luôn nổi phía trên DAW mà không làm đơ WebView2).
   - Luồng giao diện chuyên trách bơm thông điệp `PeekMessageW` / `TranslateMessage` / `DispatchMessageW`, gắn `IPlugView::attached(hwnd as *mut c_void, "HWND\0")` hoặc `clap_plugin_gui::set_parent`, hỗ trợ co giãn kích thước cửa sổ (`IPlugView::onSize` / `IPlugFrame::resizeView`).

4. **Trình quét Plugin cách ly chống sập (`scanner.rs` + binary `toneforge-plugin-scanner`) & Plugin tự kiểm định (`crates/test-vst3-clap-plugin`):**
   - Thực thi quét thư mục chuẩn (`C:\Program Files\Common Files\VST3`, `C:\Program Files\Common Files\CLAP`, thư mục do người dùng chọn, và `f:\FL studio\extracted_targets`) trong **tiến trình con cách ly (`Command::new`)** với timeout mỗi DLL: nếu một plugin lỗi hoặc thiếu giấy phép gây `STATUS_ACCESS_VIOLATION`, tiến trình con ghi nhận đen danh sách lỗi và DAW chính không bao giờ bị văng!
   - Xây dựng crate `crates/test-vst3-clap-plugin` (`crate-type = ["cdylib"]`) xuất ra **cả hai chuẩn entry point thật** (`GetPluginFactory` của VST3 COM và `clap_entry` của CLAP 1.2) cho một bộ tổng hợp âm thanh đa âm + bộ lọc cộng hưởng có độ trễ khai báo `64 samples`, tự động lưu/khôi phục state và nhận MIDI `NoteOn`. Nhờ đó, bộ kiểm định tự động kiểm chứng **100% đường nạp `LoadLibraryW` thật** mà không lệ thuộc vào khóa bản quyền thương mại trên máy test!

---

### Chặng C: Thu Âm Phần Cứng Độ Trễ Thấp (`ASIO` + `WASAPI Capture` Duplex) & `MIDI In` + Multi-Take Comping
**Mục tiêu:** Đưa độ trễ khứ hồi thu âm hát/nhạc cụ xuống ngưỡng chuyên nghiệp (`2–8 ms`) và hỗ trợ thu đè nhiều lượt (*Loop Take Comping*) mượt như phòng thu lớn.

1. **Trình điều khiển phần cứng Steinberg `ASIO` (`crates/daw-core-graph/src/audio_io/asio_hardware_driver.rs`):**
   - Quét trực tiếp Registry `HKLM\SOFTWARE\ASIO` (đã xác nhận máy hiện tại có sẵn **`Realtek ASIO`**, `CLSID = {A80362FF-CE76-4DD9-874A-704C57BF0D6A}`), khởi tạo COM `CoCreateInstance` và giao tiếp vtable `IASIO` (`init`, `getChannels`, `getBufferSize`, `canSampleRate`, `setSampleRate`, `createBuffers`, `start`, `stop`, `controlPanel`).
   - Xử lý chuyển đổi mẫu planar `ASIOSTInt32LSB` / `ASIOSTInt24LSB` / `ASIOSTFloat32LSB` hai chiều (**Full-Duplex**: đọc kênh Mic/Line-In đưa thẳng vào `MixerEngine` đồng thời ghi đầu ra Master của `MixerEngine` ra DAC trong cùng một chu kỳ ngắt `bufferSwitch`!).

2. **Thu âm `WASAPI Capture` + Khử trôi xung nhịp `Adaptive DLL Resampler` (`wasapi_capture_duplex.rs`):**
   - Khi người dùng không dùng ASIO mà cắm **Mic USB rời** đồng thời nghe bằng **Tai nghe Bluetooth / DAC cổng khác**, hai thạch anh phần cứng chạy lệch nhau vài chục ppm.
   - Triển khai luồng `IMMDeviceEnumerator::EnumAudioEndpoints(eCapture, ...)` + `IAudioCaptureClient` Event-Driven kết hợp **Vòng khóa trễ thích nghi (Adaptive Delay-Locked Loop — DLL) điều khiển bộ lấy mẫu lại đa pha Kaiser Sinc**: đo mức điền đầy trung bình của hàng đợi `rtrb` để tinh chỉnh tỷ số lấy mẫu $\pm 500\text{ ppm}$ mượt ở cấp độ $10^{-7}$, bảo đảm thu liên tục nhiều giờ **0 tràn đệm (overrun), 0 hụt mẫu (underrun), 0 méo cao độ**.

3. **Đầu vào phần cứng `MIDI In` (`midi_hardware_input.rs`):**
   - Kết nối trực tiếp API Win32 `midiInGetNumDevs`, `midiInGetDevCapsW`, `midiInOpen`, `midiInStart`.
   - Hàm callback `MIM_DATA` giải mã tức thì các thông điệp `0x80` (`NoteOff`), `0x90` (`NoteOn`), `0xB0` (`ControlChange` / Sustain Pedal `CC64` / ModWheel `CC1`), `0xE0` (`PitchBend` 14-bit) thành `SequencerVoiceEvent` gắn nhãn thời gian chính xác từng mẫu (`sample_pos`), đẩy qua `rtrb` không cấp phát bộ nhớ tới nhạc cụ đang chọn/Arm (hỗ trợ cả `PolyphonicSynthGeneratorNode` nội bộ lẫn nhạc cụ `VST3`/`CLAP`!).

4. **Bộ ghi đĩa không khóa & Quản lý nhiều lượt thu (`disk_track_recorder.rs`):**
   - Mỗi `MixerTrack` có thuộc tính `input_source_bus` và cờ `record_armed: bool` (tích hợp trọn vẹn với Điểm bù trễ số 2 của hệ thống **5-Point APDC**).
   - Khi bật `● REC`, luồng RT sao chép khung mẫu của các rãnh đang Arm vào hàng đợi SPSC `rtrb` lớn (`~2 giây` bộ đệm RAM cấp phát sẵn); luồng nền `DiskTrackRecorder` ghi tuần tự ra tệp `.wav` IEEE Float 32-bit trong thư mục dự án, tự động trừ lùi **Round-Trip Hardware + Plugin Latency** để bản thu nằm **đúng phách tuyệt đối (`sai lệch < 1 mẫu`)** trên Playlist, và khi thu lặp vòng (*Loop Recording*) tự động xếp từng vòng thành các làn **`Take 1`, `Take 2`, `Take 3`...** để người dùng chọn câu hát hay nhất!

---

### Chặng D: Chỉnh Phô Giọng Hát Bảo Toàn Âm Sắc (`Pitcher` + `NewTone`) & Co Giãn Thời Gian Đa Âm (`crates/dsp-pitch-time`)
**Mục tiêu:** Giải bài toán DSP khó nhất của DAW hiện đại — chỉnh cao độ giọng hát tự nhiên không méo tiếng và co giãn tốc độ bài nhạc giữ nguyên lực đập của trống.

1. **Bộ dò cao độ xác suất `pYIN` không cấp phát bộ nhớ (`pitch_detector.rs`):**
   - Triển khai hàm sai phân trung bình chuẩn hóa tích lũy **CMNDF** (YIN) kết hợp phân bố ngưỡng Beta nhiều mức và giải mã Viterbi ngắn hạn (**pYIN**) trên dải `40 Hz .. 1,200 Hz` kèm nội suy đỉnh Parabol bậc 2.
   - Đạt độ chính xác cao độ cơ bản $f_0$ **`< 2 cents`**, phân biệt chính xác phụ âm vô thanh (`s`, `sh`, `t`, `k`, tiếng lấy hơi) với nguyên âm hữu thanh để **tuyệt đối không kéo pitch nhầm vào phụ âm gió** (nguyên nhân số 1 gây tiếng rè nhân tạo ở các plugin chỉnh phô rẻ tiền).

2. **Thuật toán Dịch cao độ bảo toàn âm sắc vòm họng (`formant_pitch_shifter.rs`):**
   - Kết hợp **Ước lượng đường bao phổ thực (True-Envelope Iterative Cepstral Liftering)** bậc $M = \lfloor f_s / (2 f_0) \rfloor$ để tách riêng đáp tuyến cộng hưởng vòm họng $E(\omega)$ (Formants $F_1, F_2, F_3$) khỏi cấu trúc hài thanh quản $H(\omega)$, đi cùng bộ tái tổng hợp khóa pha chu kỳ **Phase-Locked TD-PSOLA / WSOLA**.
   - Khi dịch cao độ từ $f_0 \to \alpha f_0$ (`±12` bán cung), các hài dịch chuyển tới $\alpha k f_0$ nhưng biên độ của chúng được áp lại theo đúng đường bao vòm họng gốc $E(\alpha k f_0)$ (hoặc dịch độc lập theo thanh trượt **Formant Shift `±12 st`**). Nhờ đó kéo giọng lên quãng 4 / quãng 5 (+5 đến +7 bán cung) vẫn giữ nguyên độ dày lồng ngực tự nhiên, triệt tiêu hoàn toàn hiệu ứng "sóc chuột".

3. **Hai chế độ vận hành thực chiến (Thay thế song kiếm `Pitcher` + `NewTone` của FL Studio):**
   - **Thời gian thực (`AutoPitchCorrectorNode` — thay thế `Pitcher`):** Cắm trực tiếp vào khe Insert của Mixer để ca sĩ hát nghe luôn trong tai nghe với độ trễ thấp (`< 12 ms`), chọn `Root Key (C..B)`, `Scale (Chromatic / Major / Minor / Pentatonic / Harmonic Minor / Tùy chỉnh 12 phím)`, tốc độ bắt nốt `Retune Speed (0 ms Hard-Tune T-Pain .. 400 ms Tự nhiên)`, `Humanize` (giữ rung ngân tự nhiên ở nốt dài), `Formant Preserve ON/OFF`, và `Gender / Formant Shift`.
   - **Biên tập nốt đồ họa (`VocalNoteGraphEngine` + `VocalPitchEditor.svelte` — thay thế `NewTone` / `Melodyne` cơ bản):** Phân tách file thu vocal thành các khối nốt **`VocalNoteBlob`** hiển thị trực tiếp cùng đường dao động cao độ thực tế trên nền phím đàn Piano. Cho phép người dùng gắp kéo từng từ về đúng nốt, chỉnh tinh từng `cent`, bóp phẳng độ chênh (`Pitch Drift`), tăng/giảm độ rung (`Vibrato`), cắt đôi nốt, kéo giãn nhịp từng chữ (`Time Warp`), và bấm **Commit / Render to Playlist** xuất ra rãnh âm thanh sạch tuyệt đối.

4. **Bộ co giãn thời gian đa âm khóa xung gõ (`polyphonic_time_stretch.rs` — `TransientLockedTimeStretcher`):**
   - Giải quyết điểm yếu chí mạng của Phase Vocoder thông thường (làm nhòe tiếng trống Kick/Snare) bằng kiến trúc **Tách Xung Gõ – Hài Âm + Khóa Pha Đỉnh Đa Độ Phân Giải (Transient-Separated Identity Phase-Locked Phase Vocoder)**:
     1. Phát hiện tức thời các khung mang năng lượng xung gõ (*Broadband Flux / High-Frequency Content*) và giữ nguyên hình dạng sóng miền thời gian tại vị trí đã quy đổi nhịp (không bôi trơn pha qua cửa sổ FFT dài!).
     2. Trên thành phần hài/ngân, tìm các đỉnh phổ cục bộ và khóa pha toàn bộ các bin lân cận theo pha tích lũy của đỉnh chủ đạo (*Laroche & Dolson Identity Phase Locking*), giữ nguyên độ nét stereo Mid/Side.
   - Hỗ trợ co giãn `0.50x .. 2.00x` và dịch tông độc lập `±12` bán cung ngay trên Audio Clip của Playlist cũng như chế độ render chất lượng cao.

---

## 4. Tích Hợp Giao Diện `120 FPS` (`ui-svelte5-wgpu`) & Trải Nghiệm Người Dùng

Chỉ nâng cấp thư mục chính thức `f:\FL studio\toneforge\ui-svelte5-wgpu` (giữ nguyên 100% `ui-svelte-webgpu`):
1. **`App.svelte` (Thanh điều khiển trung tâm):**
   - Nút **`● REC`** phát sáng đỏ (hỗ trợ thu một chạm hoặc đếm nhịp Pre-count), nút **`↶ Undo` / `↷ Redo`** kèm phím tắt `Ctrl+Z` / `Ctrl+Y`, hộp chọn **Audio Driver (`WASAPI Shared` / `WASAPI Exclusive` / `ASIO`) + Audio Input + MIDI Input**, và hai tab không gian làm việc mới: **`🎤 Vocal Pitch & Time (NewTone)`** và **`🔌 VST3 / CLAP Rack`**.
2. **`PlaylistPianoRollEditor.svelte`:**
   - Thanh công cụ `Draw` / `Slice (Razor)` / `Lasso Select`, tay nắm `Fade In` / `Fade Out` / `Clip Gain` / `Time-Stretch` ngay trên góc Audio Clip, bật/tắt `Ghost Notes`, thanh công cụ `Quantize` / `Strum` / `Arp`, và ngăn kéo `Velocity / Pan / Fine Pitch` dưới đáy Piano Roll.
3. **`ProQ4InteractiveAnalyzer.svelte`:** Chế độ vẽ tay **`✏️ EQ Sketch`**, bảng **`Instance Collision`** chọn rãnh đối chiếu đụng độ tần số, và thanh phím đàn Piano tần số dưới đáy.
4. **`Soothe3ResonanceView.svelte`:** Hộp chọn **`External Sidechain Source Track`** và lớp phủ phổ kép (Main vs. Sidechain).
5. **`Ozone12MasterChain.svelte`:** Mô-đun **`Low End Focus`** (`Punchy` / `Smooth`, dải cắt `20–300 Hz`, `Contrast`, `Gain`) và thanh kiểm âm **`Codec Preview`** (`PCM Bypass` / `MP3 128` / `MP3 320` / `AAC 128` / `AAC 256` / `Opus 128` + nút **`Artifact Solo`** + đồng hồ cảnh báo `Post-Codec True Peak`).
6. **`VocalPitchEditor.svelte` *(Mới)*:** Không gian chỉnh nốt Vocal đồ họa kiểu `NewTone` + bảng điều khiển `Pitcher` thời gian thực.
7. **`Vst3ClapHostBrowser.svelte` *(Mới)*:** Trình duyệt quét plugin `.vst3` / `.clap`, nạp vào Channel Rack hoặc 10 khe FX của bất kỳ rãnh nào trong 502 rãnh Mixer, nút **`🗔 Open Native GUI`** mở cửa sổ Win32 `HWND`, và thanh điều khiển nhanh các tham số tự động hóa.
8. **Cập nhật Cẩm nang tiếng Việt (`huong-dan-toneforge.html` & `ui-svelte5-wgpu/public/cam-nang-toneforge.html` lên `v2.0`):** Hướng dẫn chi tiết cách thu âm ASIO/Mic, nạp VST3/CLAP, chỉnh phô Vocal và dùng bộ công cụ mới.

---

## 5. BỘ CHECKLIST NGHIỆM THU ĐỊNH LƯỢNG & THỰC CHIẾN ("CƠ SỞ ĐÁNH GIÁ APP CÓ ỔN HAY KHÔNG")

Đây là **hợp đồng nghiệm thu hai tầng** bắt buộc phải vượt qua **100%** trước khi kết thúc lượt chạy `/goal`, đồng thời là bảng kiểm để người dùng trực tiếp đối chiếu sau khi nhận bàn giao.

### Tầng 1 — 10 Cổng Kiểm Định Tự Động Định Lượng Bằng Máy (`cargo test --workspace --release`)

Ngoài việc giữ nguyên **100% `PASS` trên toàn bộ `112` bài test hiện có và `Gates 1–6` của v2** (không được phép tụt bất kỳ chỉ số nào), bộ kiểm định mới `crates/daw-tauri-host/tests/phase6_full_daw_replacement_gates.rs` phải đo đạc và xuất kết quả JSON chứng minh vượt qua **4 Cổng Mới (`Gates 7–10`)**:

| Cổng Kiểm Định | Hạng Mục Đo Lường Toán Học & Hệ Thống | Ngưỡng ĐẠT Bắt Buộc (Pass Criteria) | Dấu Hiệu Bác Bỏ Nếu Làm Ẩu (Fail) |
|---|---|---|---|
| **Gate 7**<br/>*(Workflow & 100% Plugin Parity)* | **7.1.** Chu trình `Undo -> Redo` (`20` bước liên tiếp qua Note, Clip, EQ, Routing) | Khôi phục chính xác `100%` bit-exact trạng thái `ToneForgeProjectState` & âm thanh render | Lệch tham số hoặc sót rãnh định tuyến sau khi Undo |
| **Gate 7** | **7.2.** `Pro-Q 4 EQ Sketch` khớp đường mục tiêu phức hợp (`LowCut 60 Hz` + `Bell +6 dB @ 1 kHz` + `HighShelf -4 dB @ 8 kHz`) & `Collision Bus` giữa rãnh Kick (`65 Hz`) và Bass (`65 Hz`) | Sai số khớp đường cong RMS **`< 0.85 dB`** (`80 Hz – 15 kHz`); mặt nạ `collision_mask` đạt cực đại đúng tại bin `65 Hz ± 5 Hz` | Khớp lệch quá `1.5 dB` hoặc mặt nạ va chạm toàn số `0` |
| **Gate 7** | **7.3.** `Soothe3 External Sidechain`: Rãnh 1 (Pad dày toàn dải) nhận Sidechain từ Rãnh 2 (Tín hiệu cộng hưởng `2.5 kHz` bật/tắt) | Khi Sidechain bật, năng lượng vùng `2.5 kHz` của Pad giảm **`>= 8.0 dB`** trong khi vùng `200 Hz` và `8 kHz` thay đổi **`< 1.0 dB`**, **0 tiếng click** | Nén toàn dải như compressor thường hoặc không phản ứng theo `sidechain_in` |
| **Gate 7** | **7.4.** `Ozone 12 Low End Focus` (`Punchy` tăng tỉ số đỉnh/ngân dưới `150 Hz`) & `Codec Preview` (`MP3 128` vs `MP3 320` + `Artifact Solo`) | `Punchy` tăng Crest Factor dải `< 150 Hz` thêm **`>= 1.5 dB`** mà dải `> 500 Hz` bất biến (`< 0.1 dB`); nhiễu lượng tử `MP3 128` lớn hơn `MP3 320` ít nhất **`>= 6.0 dB`** và chế độ `Artifact Solo` triệt tiêu hoàn toàn sóng gốc khi ở `PCM Bypass` (`< -110 dBFS`) | Chỉ lọc thông thấp giả hiệu hoặc lệch pha dải cao khi bật Low End Focus |
| **Gate 8**<br/>*(VST3 & CLAP Real Binary Hosting)* | **8.1.** Nạp động thư viện thực tế (`.clap` & `.vst3` biên dịch từ `test-vst3-clap-plugin` qua `LoadLibraryW`) bằng cả `ClapHostedPluginNode` và `Vst3HostedPluginNode` | Khởi tạo thành công Factory → Plugin/Component → AudioProcessor, nhận MIDI `NoteOn (A4 = 440 Hz)` sinh sóng âm đúng tần số `440 Hz ± 1 Hz`, tự động hóa tham số mượt, lưu & khôi phục State nhị phân khớp `100%` | Chỉ mock bằng trait nội bộ mà không gọi `LoadLibraryW` / COM vtable / C ABI thật |
| **Gate 8** | **8.2.** Bù trễ tự động **5-Point APDC** với Plugin ngoài có `latency = 64 samples` song song với rãnh khô `latency = 0` + Kiểm tra `#[global_allocator]` | Phép thử đảo pha (*Null Test*) giữa rãnh qua Plugin bù trễ và rãnh tham chiếu triệt tiêu sâu **`<= -90 dBFS`** (`0` sample lệch pha); **`0` `alloc`, `0` `free`, `0` `Mutex`** trong `process_slice` | Lệch pha `64` mẫu gây comb-filtering hoặc cấp phát `Vec` mỗi khung gọi VST3/CLAP |
| **Gate 9**<br/>*(ASIO / WASAPI Duplex & MIDI Recording)* | **9.1.** Liệt kê phần cứng thực tế (`HKLM\SOFTWARE\ASIO` phát hiện `Realtek ASIO` nếu có + `WASAPI eCapture` + `Win32 MIDI In`) & Kiểm định bộ khử trôi xung nhịp `Adaptive DLL Resampler` (`+150 ppm` clock drift giả lập suốt `60` giây âm thanh) | Không tràn/hụt bộ đệm (`0` xrun), SNR sóng thu qua bộ bù trôi **`>= 95 dB`**, sai số tần số đầu ra khớp xung nhịp đích | Trượt mẫu gây tiếng nổ lụp bụp định kỳ khi lệch xung nhịp |
| **Gate 9** | **9.2.** Ghi âm không khóa `DiskTrackRecorder` + Bù trễ khứ hồi (`Round-Trip Latency`) + `Loop Take Comping` (`3` vòng lặp liên tiếp) | Cả `3` tệp `.wav` của 3 lượt thu được ghi ra đĩa đầy đủ, xung nhịp đầu tiên căn khớp vị trí `sample 0` trên Playlist với sai số **`0` mẫu**, và luồng RT ghi nhận **`0` `alloc` / `0` `free`** | Tệp thu bị trễ nhịp về phía sau hoặc gọi `File::write` ngay trên luồng âm thanh RT |
| **Gate 10**<br/>*(Studio Vocal Pitch & Polyphonic Time-Stretch)* | **10.1.** Độ chính xác bộ dò `pYIN` trên giọng hát tổng hợp có rung ngân (*vibrato* `5.5 Hz`, `±35 cents`) xen kẽ phụ âm gió vô thanh | Sai số RMS cao độ vùng hữu thanh **`< 2.0 cents`**; nhận diện đúng `100%` đoạn phụ âm gió là vô thanh (`voiced_prob < 0.2`) để không kéo méo | Báo sai quãng tám (octave error) hoặc kéo méo tiếng xì vô thanh |
| **Gate 10** | **10.2.** Dịch cao độ bảo toàn âm sắc (`+5.0` bán cung, tương đương tần số cơ bản tăng `+33.48%`) trên nguyên âm có đỉnh cộng hưởng vòm họng Formant $F_1 = 730\text{ Hz}, F_2 = 1090\text{ Hz}$ | Tần số cơ bản $f_0$ đạt chính xác **`+500 ± 5 cents`**, đồng thời đỉnh cộng hưởng vòm họng $F_1, F_2$ giữ nguyên trong phạm vi **`< 8.0%`** (so với dịch tốc độ thường bị trôi vọt `+33.5%` gây giọng sóc chuột!) | Đỉnh Formant trôi tuyến tính theo $f_0$ (`> 20%`) |
| **Gate 10** | **10.3.** Co giãn thời gian đa âm `TransientLockedTimeStretcher` ở tỷ lệ `1.35x` (chậm lại) và `0.75x` (nhanh lên) trên tổ hợp Kick sắc cạnh + Hợp âm ngân stereo | Độ dài đầu ra chính xác tới `±4` mẫu; hệ số đỉnh xung gõ (*Transient Crest Factor*) của tiếng Kick giữ đạt **`>= 85%`** bản gốc (không bị bệt nhòe) và tương quan pha Stereo hợp âm **`>= 0.95`** | Tiếng Kick bị tù đầu, mất lực đập hoặc xuất hiện tiếng vọng đôi (*double-attack*) |

---

### Tầng 2 — Checklist 8 Bài Kiểm Tra Thực Chiến Dành Cho Người Dùng (Mở `ToneForge_Studio.exe` Đánh Giá Trực Tiếp)

Sau khi cửa sổ `/goal` báo hoàn thành và xuất file `ToneForge_Studio.exe` mới, anh dùng trực tiếp **8 bài kiểm tra thực tế dưới đây** để nghiệm thu bằng tai và thao tác chuột xem ứng dụng đã thực sự "ổn và thay thế được FL Studio" hay chưa:

- [ ] **Bài 1 — Độ đầm của thao tác & Hoàn tác tức thì (`Undo / Redo`):**
  - *Cách thử:* Mở bài nhạc đang phát qua loa WASAPI, liên tục vẽ thêm nốt Piano Roll, cắt đôi Audio Clip trên Playlist, kéo mạnh 1 dải EQ trên Pro-Q 4 rồi bấm `Ctrl+Z` 3 lần liên tiếp và `Ctrl+Y` 3 lần liên tiếp.
  - *Tiêu chuẩn ĐẠT:* Giao diện và âm thanh quay lui / tiến tới tức thì đúng từng bước, **loa phát mượt tuyệt đối không có bất kỳ tiếng khựng hay nổ "tách" nào**.
- [ ] **Bài 2 — Sáng tác & Cắt ghép trên Playlist + Piano Roll:**
  - *Cách thử:* Dùng dao `Razor Slice` cắt một đoạn Audio trên Playlist, kéo góc trên trái/phải tạo đường `Fade In / Fade Out`, kéo giãn clip theo BPM mới; mở Piano Roll bật `Ghost Notes`, quét chọn một hợp âm rồi bấm `Strum` và kéo thanh `Velocity` dưới đáy.
  - *Tiêu chuẩn ĐẠT:* Vết cắt không bị click ở điểm nối; khi co giãn tốc độ tiếng trống vẫn đanh gọn không bị nhòe như dưới nước; hợp âm rải nghe tự nhiên như tay người đánh.
- [ ] **Bài 3 — Kiểm chứng 100% `Pro-Q 4` (`EQ Sketch` + `Đụng độ tần số Kick/Bass`):**
  - *Cách thử:* Bật chế độ `✏️ EQ Sketch`, cầm chuột vẽ một nét lượn sóng từ trầm lên cao rồi nhả chuột; sau đó mở danh sách `Collision` chọn đối chiếu giữa rãnh `Bass` và rãnh `Kick`.
  - *Tiêu chuẩn ĐẠT:* Nhả chuột ra là đường cong tự biến thành các điểm EQ mượt khớp đúng nét vẽ; trên dải trầm `50–90 Hz` hiện rõ vùng cảnh báo đỏ nơi Kick và Bass đang đè nhau.
- [ ] **Bài 4 — Kiểm chứng 100% `Soothe3` (`External Sidechain` dọn chỗ cho Vocal):**
  - *Cách thử:* Đặt `Soothe3` lên rãnh Nhạc nền (`Other` / `Instruments`), chọn `External Sidechain Source = Vocals` và tăng `Depth`.
  - *Tiêu chuẩn ĐẠT:* Chỉ khi ca sĩ cất giọng hát thì đúng dải tần trung của nhạc nền mới hõm xuống nhường chỗ cho giọng hát nổi bật lên trước mặt; lúc ca sĩ nghỉ lấy hơi, nhạc nền đầy đặn trở lại 100%.
- [ ] **Bài 5 — Kiểm chứng 100% `Ozone 12` (`Low End Focus` + `Codec Preview`):**
  - *Cách thử:* Trên tab Mastering, bật `Low End Focus` chuyển qua lại giữa `Punchy` và `Smooth`; sau đó bật `Codec Preview -> MP3 128 kbps` và ấn nút `Artifact Solo`.
  - *Tiêu chuẩn ĐẠT:* `Punchy` làm tiếng trống trầm bật rõ lực đập mà không làm đục giọng hát; khi ấn `Artifact Solo` ở `MP3 128`, tai nghe thấy rõ tiếng xèo nhiễu nén số và đồng hồ báo chính xác mức đỉnh thực sau khi nén MP3.
- [ ] **Bài 6 — Nạp Plugin Hãng Thứ 3 (`VST3` / `CLAP`) & Mở Giao Diện Gốc:**
  - *Cách thử:* Mở tab `🔌 VST3 / CLAP Rack`, bấm quét thư mục Plugin (hoặc nạp plugin `.vst3`/`.clap` có sẵn), chèn vào rãnh, bấm `🗔 Open Native GUI` và gõ phím MIDI / chạy bài nhạc.
  - *Tiêu chuẩn ĐẠT:* Cửa sổ đồ họa riêng của plugin bật lên mượt mà, không làm đơ giao diện chính của ToneForge; vặn núm trên cửa sổ plugin âm thanh đổi ngay, lưu Project tắt đi mở lại vẫn giữ nguyên thông số plugin.
- [ ] **Bài 7 — Thu âm thật qua Mic / Soundcard (`ASIO` / `WASAPI`) & Lặp nhiều Take:**
  - *Cách thử:* Chọn đầu vào Mic (hoặc `Realtek ASIO` / soundcard rời) và bàn phím MIDI (nếu có), bấm `Arm` trên một rãnh trống, bật vòng lặp 4 ô nhịp và ấn `● REC` gõ nhịp / hát thử 2–3 lượt liên tiếp.
  - *Tiêu chuẩn ĐẠT:* Nhịp thu phát lại khớp khít với tiếng nhạc nền (không bị trễ đuôi sau phách), các lượt thu xếp gọn thành `Take 1`, `Take 2` để chọn lượt ưng ý nhất, không bị tiếng rẹt rẹt lệch xung nhịp.
- [ ] **Bài 8 — Chỉnh phô giọng hát (`Pitcher` thời gian thực & `NewTone` kéo nốt đồ họa):**
  - *Cách thử:* Mở một rãnh Vocal trong tab `🎤 Vocal Pitch & Time`, bấm phân tích nốt rồi cầm một nốt hát hơi chênh kéo về đúng ô phím đàn (và thử kéo cao lên hẳn `+4` đến `+5` bán cung với `Formant Preserve = ON`), sau đó bấm `Commit`.
  - *Tiêu chuẩn ĐẠT:* Nốt chênh vào đúng tông mượt mà; ngay cả khi kéo lên `+5` bán cung giọng ca sĩ vẫn là giọng người thật dày dặn (không biến thành giọng hoạt hình méo mó), các chữ gió `s`, `sh`, `th` giữ nguyên độ sạch tự nhiên.

---

## 6. Yêu Cầu Báo Cáo Phản Biện Trung Thực Sau Khi Triển Khai (`10_toneforge_v3_acceptance_report.md`)

Sau khi hoàn tất toàn bộ code, vượt qua `100%` bài test `--release` (`Gates 1–10`), `npm run build` (`0` lỗi), và xuất bản `f:\FL studio\toneforge\ToneForge_Studio.exe` mới:
1. Ghi báo cáo nghiệm thu chi tiết bằng tiếng Việt tại `f:\FL studio\research_notes\10_toneforge_v3_acceptance_report.md` kèm bảng số liệu đo thực tế của từng cổng `Gate 1 -> Gate 10` và mã băm SHA-256 của bản `.exe` mới.
2. **Mục bắt buộc ở cuối báo cáo — "Danh Sách Tự Phản Biện Các Điểm Biên Cần Cùng Người Dùng Tinh Chỉnh Tiếp":**
   - Không được tô hồng. Phải chỉ thẳng những góc biên thực tế nào sau khi chạy thử nghiệm định lượng thấy còn có thể tối ưu sâu hơn ở vòng tinh chỉnh tiếp theo (ví dụ: hành vi với các VST3 thương mại dùng cơ chế bảo vệ iLok/licenser đặc thù, chất lượng chuyển tiếp âm gió khi kéo cao độ cực đại `> +9` bán cung, cảm giác kéo thả con trỏ ở mật độ `> 2,000` nốt trên một màn hình, hay độ tương thích buffer size siêu nhỏ `32–64 samples` trên từng dòng ASIO driver khác nhau) để hai anh em có cơ sở kỹ thuật chuẩn xác nhất bước vào giai đoạn rà soát và hoàn thiện hậu triển khai.
