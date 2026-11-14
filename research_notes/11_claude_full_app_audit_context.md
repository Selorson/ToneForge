# Phạm Vi & Bất Biến Cốt Lõi Dự Án ToneForge (`f:\FL studio\toneforge`)

Tài liệu này chỉ ghi nhận các ràng buộc phạm vi cố định của chủ dự án (không chứa hướng dẫn kiểm toán để đảm bảo tính độc lập tuyệt đối ở Pha 1):

1. **Phạm vi kiểm toán & hoàn thiện:** Toàn bộ ứng dụng `f:\FL studio\toneforge` kể từ khi bắt đầu xây dựng đến hiện tại (toàn bộ workspace Rust `Cargo.toml`, `shell/src-tauri`, `assets/`, và frontend chính thức `ui-svelte5-wgpu`).
2. **Các bất biến kỹ thuật không được phép phá vỡ:**
   - Luồng âm thanh thời gian thực (`Real-Time Audio Thread`): `0` `Mutex`, `0` cấp phát động (`alloc`), `0` giải phóng bộ nhớ (`dealloc`).
   - Khung dữ liệu hiển thị 120 FPS: `std::mem::size_of::<VisualTelemetryFrame>() == 131_072` bytes (`128 KB`).
   - Giữ nguyên không sửa đổi thư mục giao diện thế hệ cũ `f:\FL studio\toneforge\ui-svelte-webgpu`.
   - Hoãn 100% tính năng sản xuất MV / Video.
   - Chất lượng âm thanh & độ trung thực toán học chuẩn phòng thu thương mại là ưu tiên số 1 tuyệt đối (cấm placeholder, trọng số giả, dữ liệu ngụy trang, hoặc dùng WebAudio giả làm DSP Rust trong Tauri).
   - Mọi bài kiểm thử tự động chạm tới phần cứng âm thanh thật phải gọi `set_volume_and_mute(0.0, true)`.
3. **Tài liệu niêm phong chỉ mở sau khi hoàn tất kiểm toán độc lập (`12_claude_independent_findings.md`):**
   - `f:\FL studio\research_notes\11b_sealed_prior_audit_ledger.md`
   - `f:\FL studio\research_notes\10_toneforge_v3_acceptance_report.md`
