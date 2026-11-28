## ToneForge Studio v0.1.2 — Vá đợt kiểm toán liên thông (REM-16)

### Sửa lỗi
- **Hoàn tác / Mở dự án:** plugin ngoài khôi phục lại luôn ở trạng thái cửa sổ đóng, khớp với lõi (lõi đóng mọi cửa sổ plugin khi Hoàn tác/Mở). Hết cảnh nút "đang mở" mà không thấy cửa sổ.
- Thêm kiểm tra bố cục ABI cho `IMidiMappingVtbl` để bắt sớm lệch giao tiếp VST3.

---

## ToneForge Studio v0.1.1 — Vá lỗi Toàn diện DSP/FFI/WebGPU & Tối ưu RAM/VRAM

### 1. Ổn định lõi FFI/ABI, MIDI & Luồng Âm thanh Thời gian thực (Real-Time Safety)
- **VST3 & CLAP Host:** Căn chỉnh chuẩn ABI `Vst3Event` (`48` byte, `payload` tại offset `24`), ghim con trỏ `NativePluginWindow` trên heap kèm đếm tham chiếu `InitDll`/`ExitDll`, đóng cửa sổ GUI an toàn ngoài khóa `Mutex`, mở rộng không giới hạn số lượng tham số cho Synth lớn và cắt chia khối `process` theo `max_block_size` mà không rơi sự kiện MIDI.
- **APDC, ASIO & WASAPI:** Loại bỏ `assert!` trên luồng RT trong `ApdcDelayLine`, sửa UB truy cập đồng thời callback Win32 `MIDI` (`UnsafeCell` + truyền đủ Pitch-Bend), tự khởi động lại luồng `ASIO` khi nhận `kAsioResetRequest`, đọc RIFF WAV đa chunk chuẩn xác và xử lý trọn gói thu `WASAPI > 4096` frame (kèm PCM 8-bit & downmix đa kênh).
- **Thu âm trực tiếp (`DiskTrackRecorder`):** Bảo toàn và đóng gói đầy đủ header RIFF WAV của các bản thu đang chạy trước khi thoát ứng dụng, đồng thời hỗ trợ thu âm tự do không bị tua lặp `4` bar khi chưa kẹp vòng lặp.

### 2. Độ chính xác DSP Mastering, Chuẩn `-14 LUFS` & Giải phóng VRAM
- **Pro-Q 4, Soothe3 & Ozone 12:** Khử sạch `NaN`/`Inf` đầu vào, đồng bộ trục tần số Logarit `10..30,000 Hz` giữa Rust FFT và WebGPU, tách 2 nhân FIR `LinearPhase` độc lập theo kênh Stereo/Mid-Side, sửa chuẩn hóa `IRC V` / `Low End Focus`, tắt mặc định nhuộm hài `BassControl`, cộng bù chính xác độ trễ `soft_clip_oversampler` và giữ nền ảnh nhiệt `-120 dB` không rách khung `TripleBuffer`.
- **AI Stem & GPU Runtime (`CUDA 12` / `DirectML`):** Khởi tạo lười (lazy-init) các worker BSRNN/DeLimit, tách độc lập 3 bộ hẹn giờ nhả VRAM rảnh (`15s`), tái sử dụng bộ đệm scratch STFT không cấp phát lặp và gom gọn tài nguyên trọng số về một nguồn chuẩn duy nhất.
- **Sidecar `akunarch-audio-cli` & Offline Render:** Đồng bộ đo lường Integrated LUFS theo chuẩn ITU-R BS.1770-4 hai cổng, sửa công thức `bgm-gain-db`, xuất `StemMastering` bám đúng chuỗi Master hiện hành và gỡ giới hạn cắt `30s` trên trình biên tập cao độ NewTone.

### 3. Giao diện WebGPU 120 FPS & Đồng bộ Dự án DAW
- Dùng chung một `GPUDevice` singleton và giải phóng tài nguyên GPU khi chuyển phân khu, gộp lệnh kéo chuột IPC theo khung hình `requestAnimationFrame` kèm `setPointerCapture`, hiển thị đủ `8` dải Sidechain của Soothe3, bù trễ `APDC` cho Playhead và khôi phục trọn vẹn trạng thái giao diện lẫn bộ đệm âm thanh khi mở tệp dự án `.tfproj`.
