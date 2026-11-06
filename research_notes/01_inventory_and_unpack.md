# Bước 1: Danh Mục Giải Nén Tĩnh & Kiến Trúc PE (Inventory & Static Unpack)

> **Ngày thực hiện:** 2026-11-05  
> **Công cụ phân tích:**  
> - [`craft-anything-cli.exe`](file:///f:/FL%20studio/craft-anything/target/release/craft-anything-cli.exe) (`v0.4.0` — Rust native binary/app analyzer)  
> - [`rea.mjs`](file:///f:/FL%20studio/rea/scripts/rea.mjs) (`v6.8.0` — Reverse Engineering Assistant)  
> - Bộ script Python giải nén tĩnh chuyên biệt cho 4 định dạng installer (`FS/ISB/3`, `Inno Setup 5.5.7`, `BitRock InstallBuilder / CookFS`, `NSIS 64-bit`).

---

## 1. Trạng Thái Môi Trường Công Cụ (`craft-anything` & `rea`)

1. **`craft-anything` (`v0.4.0`)**:
   - Đã biên dịch thành công bản `release` tại [`f:\FL studio\craft-anything\target\release\craft-anything-cli.exe`](file:///f:/FL%20studio/craft-anything/target/release/craft-anything-cli.exe).
   - Các lệnh kiểm thử hoạt động tốt: `doctor`, `app-folder-explain`, `core-target-triage`, `native-offered-symbols-list`, `native-layout-show`, `native-pe-resources-list`.
2. **`rea` (`v6.8.0`)**:
   - Đã cài đặt dependencies và biên dịch TypeScript sang `dist/` tại [`f:\FL studio\rea\scripts\rea.mjs`](file:///f:/FL%20studio/rea/scripts/rea.mjs).
   - Các lệnh kiểm thử hoạt động tốt: `inspect-pe-imports`, `inspect-native-exports`, `triage`.

---

## 2. Cơ Chế Giải Nén Tĩnh 4 Bộ Cài Đặt (Không Cài Vào Windows)

Do cả 4 bộ cài `.exe` đều sử dụng các định dạng đóng gói (installer container) tùy biến khiến `7z.exe` mặc định không thể giải nén trực tiếp toàn bộ payload, quá trình giải nén tĩnh đã phân tích cấu trúc header nhị phân của từng định dạng và bung toàn bộ dữ liệu vào [`f:\FL studio\extracted_targets`](file:///f:/FL%20studio/extracted_targets):

| Bộ cài gốc | Định dạng Container / Compression | Phương pháp giải nén tĩnh | Thư mục đích |
| :--- | :--- | :--- | :--- |
| [`ffproq413x64.exe`](file:///f:/FL%20studio/ffproq413x64.exe) | **Delphi Custom Stub + `FS/ISB/3` Container** (tại offset `0x20000`, nén từng file bằng luồng `zlib` `0x78 0xDA`) | Quét bảng mục lục `FS/ISB/3` và giải nén toàn bộ các luồng `zlib` theo tên file gốc | [`extracted_targets/ffproq4`](file:///f:/FL%20studio/extracted_targets/ffproq4) |
| [`soothe3_v1.0.5_win.exe`](file:///f:/FL%20studio/soothe3_v1.0.5_win.exe) | **Inno Setup `5.5.7 (u)`** (Header tại `0x78C00`, dữ liệu tại `0x16800`, nén **Solid LZMA1** stream) | Giải mã Inno Setup Metadata Header để lấy danh sách tên file + kích thước, sau đó giải nén luồng Solid LZMA1 (`5D 00 00 80 00`) và cắt theo đúng kích thước từng file | [`extracted_targets/soothe3`](file:///f:/FL%20studio/extracted_targets/soothe3) |
| [`iZotope_Ozone_12_Advanced_v12_2_0_1951-1.exe`](file:///f:/FL%20studio/iZotope_Ozone_12_Advanced_v12_2_0_1951-1.exe) | **BitRock InstallBuilder / Tcl CookFS** (`CFS0002` catalog chính tại cuối file + các sub-archive `CFS2.200` cho từng module VST3/AAX/Core) | Giải nén cấu trúc CookFS `pages_index` + `fs_index` (nén `zlib`), sau đó tiếp tục bung các gói con `CFS2.200` của `mothership` và `maximizer` | [`extracted_targets/ozone12`](file:///f:/FL%20studio/extracted_targets/ozone12) |
| [`flstudio_win64_26.1.7.5654.exe`](file:///f:/FL%20studio/flstudio_win64_26.1.7.5654.exe) | **NSIS 64-bit Custom Header** (`firstheader` 36 bytes tại `0x59000` thay vì 28 bytes chuẩn, nén **Non-Solid LZMA1**, tổng cộng **27,574 entries** `EW_EXTRACTFILE`) | Giải nén khối NSIS Header (`11.27 MB`), trích xuất toàn bộ danh mục 27,574 file vào [`all_files_inventory.txt`](file:///f:/FL%20studio/extracted_targets/flstudio26/all_files_inventory.txt) và bung toàn bộ các file thực thi, DLL core, DSP engine, UI engine và bộ plugin gốc của FL Studio | [`extracted_targets/flstudio26`](file:///f:/FL%20studio/extracted_targets/flstudio26) |

---

## 3. Danh Mục File & Kiến Trúc PE Từng Mục Tiêu

### 3.1. Mục tiêu 1: FabFilter Pro-Q 4 (`v4.13` — Build 25/06/2026)
- **Thư mục:** [`f:\FL studio\extracted_targets\ffproq4`](file:///f:/FL%20studio/extracted_targets/ffproq4)

#### A. Danh mục file đã bung
| File | Kích thước | Vai trò kỹ thuật |
| :--- | ---: | :--- |
| [`FabFilter Pro-Q 4.dll`](file:///f:/FL%20studio/extracted_targets/ffproq4/FabFilter%20Pro-Q%204.dll) | `4,784,128 B` (4.56 MB) | Binary plugin chính tích hợp cả **VST3** và **VST2** (64-bit PE DLL, SHA256: `1c38134df707786b7f75e27a69cad9828c1f99f0414c688006f1198c8fd5eb3e`) |
| [`FabFilter Pro-Q 4.clap`](file:///f:/FL%20studio/extracted_targets/ffproq4/FabFilter%20Pro-Q%204.clap) | `4,640,768 B` (4.43 MB) | Bản **CLAP** (`clap_entry` tại `0x1802841d8`) — cấu trúc gọn hơn VST3, rất thuận lợi để đối chiếu hàm DSP |
| [`FabFilter Pro-Q 4.aaxplugin`](file:///f:/FL%20studio/extracted_targets/ffproq4/FabFilter%20Pro-Q%204.aaxplugin) | `4,773,984 B` (4.55 MB) | Bản **AAX** (`ACFRegisterPlugin` tại `0x18007e6c0`) |
| [`DigiPageTable.xml`](file:///f:/FL%20studio/extracted_targets/ffproq4/DigiPageTable.xml) | `572,321 B` (559 KB) | **Tài liệu vàng:** Bảng ánh xạ toàn bộ hệ thống tham số (Parameters) cho 24 EQ Bands (Frequency, Gain, Q, Filter Shape, Slope, Stereo Placement, Dynamic Range, Threshold, Attack, Release, Spectral Density, Tilt, Character mode...) |
| [`FabFilter Pro-Q 4.chm`](file:///f:/FL%20studio/extracted_targets/ffproq4/FabFilter%20Pro-Q%204.chm) | `2,319,182 B` (2.21 MB) | Tài liệu hướng dẫn chi tiết giải thích nguyên lý hoạt động của từng chế độ (Zero Latency, Natural Phase, Linear Phase, Spectral Dynamics, EQ Sketch, Instance List) |
| [`Setup.ini`](file:///f:/FL%20studio/extracted_targets/ffproq4/Setup.ini) / [`Readme.txt`](file:///f:/FL%20studio/extracted_targets/ffproq4/Readme.txt) | `1.2 KB` / `9.5 KB` | Thông tin cấu hình cài đặt và release notes |

#### B. Kiến trúc PE của [`FabFilter Pro-Q 4.dll`](file:///f:/FL%20studio/extracted_targets/ffproq4/FabFilter%20Pro-Q%204.dll)
- **Trình biên dịch:** Microsoft Visual C++ (MSVC x64), liên kết tĩnh C++ Runtime (`static MSVC runtime`).
- **Bố cục Section (`native-layout-show`):**
  - `.text`: VA `0x1000`, Size `2,469,888 B` (`2.36 MB`), Entropy `6.37` (**Không bị pack/mã hóa** — có thể dịch ngược trực tiếp toàn bộ thuật toán Biquad, Natural Phase, Linear Phase FFT và Spectral Dynamics!).
  - `.rdata`: VA `0x25C000`, Size `585,728 B`, Entropy `5.57` (chứa RTTI, vtables, hằng số toán học DSP và chuỗi shader OpenGL).
  - `.data`: VA `0x2EB000`, Size `53,248 B`, Entropy `3.82`.
  - `.rsrc`: VA `0x30C000`, Size `1,646,592 B` (`1.57 MB`), Entropy `7.89` (chứa **361 PE Resources** gồm PNG sprites độ phân giải cao `@2x`, font chữ, icon và layout UI: `ANALYZERBACKGROUND_2X`, `BOTTOMBAR_2X`, ` PIANO`, v.v.).
- **Exports (`inspect-native-exports`):**
  - `GetPluginFactory` (`0x180106f90`) — Entry point chuẩn VST3.
  - `VSTPluginMain` (`0x180106fa0`) — Entry point chuẩn VST2.
- **Imports (`inspect-pe-imports`):**
  - **Đồ họa UI:** `OPENGL32.dll` (`glBindTexture`, `glTexSubImage2D`, `glGetIntegerv`, `wglMakeCurrent`, `wglShareLists`...), `GDI32.dll`, `dwmapi.dll` (`DwmFlush` để đồng bộ khung hình mượt mà).
  - **Hệ thống / Timing:** `WINMM.dll` (`timeGetTime`), `KERNEL32.dll`, `USER32.dll`, `SHLWAPI.dll`, `SHELL32.dll`, `ADVAPI32.dll`.

---

### 3.2. Mục tiêu 2: oeksound Soothe3 (`v1.0.5` — Build 29/06/2026)
- **Thư mục:** [`f:\FL studio\extracted_targets\soothe3`](file:///f:/FL%20studio/extracted_targets/soothe3)

#### A. Danh mục file đã bung & Phát hiện đặc biệt (`soothe3_symbols.map`)
| File | Kích thước | Vai trò kỹ thuật |
| :--- | ---: | :--- |
| [`soothe3.vst3`](file:///f:/FL%20studio/extracted_targets/soothe3/soothe3.vst3) | `47,501,064 B` (45.3 MB) | Binary VST3 64-bit chính |
| [`soothe3.aaxplugin`](file:///f:/FL%20studio/extracted_targets/soothe3/soothe3.aaxplugin) | `47,445,256 B` (45.2 MB) | Binary AAX 64-bit |
| [`soothe3_symbols.map`](file:///f:/FL%20studio/extracted_targets/soothe3/soothe3_symbols.map) | `26,182,145 B` (24.97 MB) | **PHÁT HIỆN QUAN TRỌNG NHẤT:** File **MSVC Linker Map** hoàn chỉnh do nhà phát triển đóng gói kèm trong bộ cài! Chứa tên đầy đủ (mangled + RVA + object file) của **toàn bộ mọi class, hàm C++, bộ lọc STFT, thuật toán nhận diện cộng hưởng (Resonance Suppression), JUCE framework và Intel IPP** bên trong `soothe3`! |
| [`DigiPageTable.xml`](file:///f:/FL%20studio/extracted_targets/soothe3/DigiPageTable.xml) | `160,367 B` (156 KB) | Bảng ánh xạ toàn bộ tham số điều khiển của Soothe3 (Depth, Sharpness, Selectivity, Attack, Release, Soft/Hard mode, Band nodes, Mid/Side/Left/Right balance, Trim, Mix, Delta...) |
| [`moduleinfo.json`](file:///f:/FL%20studio/extracted_targets/soothe3/moduleinfo.json) | `1,398 B` | Metadata cấu hình module VST3 (`CID: 4F656B73537468336F656B7353746833`, Vendor: `oeksound`, Version: `1.0.5`) |
| [`Presets/`](file:///f:/FL%20studio/extracted_targets/soothe3/Presets) (68 files) | `~150 KB` | Toàn bộ 68 Factory Presets dưới dạng **JSON thuần túy** (ví dụ: [`Lead Vocal - De-ess and tame.preset`](file:///f:/FL%20studio/extracted_targets/soothe3/Presets/Vocals/Lead%20Vocal%20-%20De-ess%20and%20tame.preset)), cho thấy chính xác giá trị chuẩn hóa `0.0 -> 1.0` của từng tham số trong thực tế! |
| [`soothe3_manual.pdf`](file:///f:/FL%20studio/extracted_targets/soothe3/soothe3_manual.pdf) | `5,731,278 B` (5.46 MB) | Tài liệu kỹ thuật chính thức của Soothe3 |

#### B. Kiến trúc PE & Phân tích [`soothe3_symbols.map`](file:///f:/FL%20studio/extracted_targets/soothe3/soothe3_symbols.map)
- Mặc dù section `.text` của `soothe3.vst3` được bọc bởi PACE (`wraps` / `.pace`, entropy `8.00`), việc có sẵn file [`soothe3_symbols.map`](file:///f:/FL%20studio/extracted_targets/soothe3/soothe3_symbols.map) (`26.18 MB`) giúp chúng ta **không cần phải giải mã lớp bọc PACE** mà vẫn đọc được toàn bộ kiến trúc hàm nội bộ, cấu trúc class C++ và thư viện toán học của Soothe3:
  - **Layout gốc trong `.map`:**
    - `.text` (`CODE`): `0x0197C230` bytes (`26.7 MB` code thực thi).
    - `IPPCODE` (`CODE`): `0x000B4B7D` bytes (`740 KB` — xác nhận Soothe3 sử dụng **Intel IPP (Integrated Performance Primitives)** cho các phép biến đổi FFT/STFT và vector math).
    - `.rdata` (`DATA`): `0x010DA028` bytes (`17.6 MB`).
- **Exports (`soothe3.vst3`):** `GetPluginFactory`, `IPluginCompatibility`.
- **Imports (`soothe3.vst3`):** `OPENGL32.dll` (render phổ tần số 3D/2D bằng OpenGL), `GDI32.dll`, `WININET.dll`, `IMM32.dll`, `COMDLG32.dll`, `OLEAUT32.dll`, `WS2_32.dll`.

---

### 3.3. Mục tiêu 3: iZotope Ozone 12 Advanced (`v12.2.0.1951`)
- **Thư mục:** [`f:\FL studio\extracted_targets\ozone12`](file:///f:/FL%20studio/extracted_targets/ozone12)

#### A. Danh mục file trọng tâm đã bung
| File | Kích thước | Vai trò kỹ thuật |
| :--- | ---: | :--- |
| [`iZOzone12Core.dll`](file:///f:/FL%20studio/extracted_targets/ozone12/megacore_component/MEGACORE_WIN/iZOzone12Core.dll) | `579,805,696 B` (**552.9 MB**) | **Trái tim DSP & AI của toàn bộ Ozone 12:** Chứa toàn bộ thuật toán xử lý tín hiệu cho mọi module (Maximizer/IRC, Equalizer, Dynamic EQ, Stabilizer, Impact, Imager, Clarity, Master Rebalance, Low End Focus, Spectral Shaper, Exciter, Vintage Limiter/EQ/Compressor/Tape, Unlimiter, Stem EQ) |
| [`Ozone 12.vst3`](file:///f:/FL%20studio/extracted_targets/ozone12/mothership_unpacked/VST3/VST3_Win/Ozone%2012.vst3) | `2,115,072 B` (2.02 MB) | VST3 Frontend Wrapper cho Mothership (nạp động `iZOzone12Core.dll`) |
| [`Ozone 12 Maximizer.vst3`](file:///f:/FL%20studio/extracted_targets/ozone12/maximizer_unpacked/VST3/VST3_Win/Ozone%2012%20Maximizer.vst3) | `2,115,584 B` (2.02 MB) | VST3 Frontend Wrapper cho module độc lập (tương tự cho 20 module thành phần khác) |
| [`onnxruntime.dll`](file:///f:/FL%20studio/extracted_targets/ozone12/libonnx_component/libonnx_component_x64/onnxruntime.dll) | `10,971,648 B` (10.46 MB) | Engine suy luận mạng nơ-ron **ONNX Runtime** cho Master Assistant & AI |
| [`vocals-small_sanitized.npz`](file:///f:/FL%20studio/extracted_targets/ozone12/requiredFiles_component/ssepnpz/vocals-small_sanitized.npz) | `26,261,516 B` (25.04 MB) | Trọng số mạng nơ-ron tách/cân bằng giọng hát (Master Rebalance / Stem EQ) định dạng NumPy `.npz` |
| [`drums-small_sanitized.npz`](file:///f:/FL%20studio/extracted_targets/ozone12/requiredFiles_component/ssepnpz/drums-small_sanitized.npz) | `26,261,516 B` (25.04 MB) | Trọng số mạng nơ-ron tách/cân bằng trống định dạng NumPy `.npz` |
| [`bass-small_sanitized.npz`](file:///f:/FL%20studio/extracted_targets/ozone12/requiredFiles_component/ssepnpz/bass-small_sanitized.npz) | `26,261,516 B` (25.04 MB) | Trọng số mạng nơ-ron tách/cân bằng bass định dạng NumPy `.npz` |
| [`Exciter Cache/*.dat`](file:///f:/FL%20studio/extracted_targets/ozone12/exciterCache_component/ExciterCache/Exciter%20Cache) | `~70 MB` (24 files) | Bảng tra cứu (Lookup Tables) mô phỏng phi tuyến bóng đèn điện tử (`exciter-triode-*.dat`, `exciter-dualtriode-*.dat`) được tính sẵn cho từng tần số lấy mẫu (`44.1kHz`, `48kHz`, `88.2kHz`, `96kHz`, `176.4kHz`, `192kHz`) |
| [`Meter Tap 3.dll`](file:///f:/FL%20studio/extracted_targets/ozone12/metertap3_component/MeterTap64/Meter%20Tap%203.dll) | `449,864 B` (439 KB) | Giao thức truyền dữ liệu phổ âm thanh liên tiến trình (Inter-plugin communication) cho Tonal Balance Control / Metering |
| [`libmp3lame.dll`](file:///f:/FL%20studio/extracted_targets/ozone12/libmp3lame_component/libmp3lame_x64/libmp3lame.dll) | `372,224 B` (363 KB) | Codec kiểm thử Audition Codec Preview (MP3/AAC artifact preview) |
| [`Presets/`](file:///f:/FL%20studio/extracted_targets/ozone12/presets_component/Presets/Global%20Presets) (700+ files) | `~5 MB` | Hơn 700 file `.xml` preset cho Mothership và toàn bộ 20 module thành phần, lưu rõ tên tham số và giá trị cấu hình |

#### B. Kiến trúc PE của [`iZOzone12Core.dll`](file:///f:/FL%20studio/extracted_targets/ozone12/megacore_component/MEGACORE_WIN/iZOzone12Core.dll) & Khám phá AI `.npz`
- **Trình biên dịch:** MSVC x64 (`Dynamic MSVC Runtime`: `MSVCP140.dll`, `VCRUNTIME140.dll`, `ucrtbase.dll`).
- **Bố cục Section (`native-layout-show`):**
  - `.text`: VA `0x1000`, Size `43,925,504 B` (**41.89 MB** mã máy thực thi!), Entropy `6.37` (**Hoàn toàn không bị mã hóa/pack!**).
  - `.rdata`: VA `0x29E5000`, Size `534,536,192 B` (**509.77 MB**), Entropy `7.42` (chứa các mô hình AI nhúng, bảng IR, hằng số DSP, RTTI, và tài nguyên giao diện).
  - `.data`: VA `0x227AA000`, Size `743,424 B`, Entropy `4.69`.
- **Exports (`native-offered-symbols-list`):**
  - `PluginCoreCallback` (`0x1805d9020`) — Cổng giao tiếp chính giữa vỏ VST3 (`Ozone 12.vst3`) và nhân `iZOzone12Core.dll`.
  - `PluginCoreInterfaceVersion` (`0x1805dc720`).
  - `iZResampler::ResamplerEngine` (8 hàm C++ được export trực tiếp gồm constructor, `operator=`, `GetVersion`, vtable).
- **Imports (`inspect-pe-imports`):**
  - **Đồ họa:** `OPENGL32.dll`, `d3d9.dll` (`Direct3DCreate9Ex`), `gdiplus.dll`, `GDI32.dll`, `MSIMG32.dll`.
  - **Audio / Media:** `MFPlat.DLL`, `MFReadWrite.dll` (Windows Media Foundation để giải mã/mã hóa codec preview).
- **Cấu trúc bên trong các file AI `.npz` (`vocals-small_sanitized.npz`, `drums-small_sanitized.npz`, `bass-small_sanitized.npz`):**
  - Mỗi file `.npz` chứa đúng **27 mảng NumPy (`.npy`)**:
    - Siêu tham số STFT: `FFT_SIZE.npy`, `STFT_MASK_EPSILON.npy`, `SIR_REGULARIZER.npy`, `NUM_INPUT_BATCHES.npy`, `RETURN_COMPLEMENT.npy`.
    - Trọng số mạng nơ-ron: `arr_0.npy` đến `arr_21.npy` (tổng cộng ~26.2 MB trọng số `float32`), cho phép đọc trực tiếp cấu trúc các lớp (layers) bằng Python/Rust!

---

### 3.4. Mục tiêu 4: FL Studio 26 (`v26.1.7.5654`)
- **Thư mục:** [`f:\FL studio\extracted_targets\flstudio26`](file:///f:/FL%20studio/extracted_targets/flstudio26)
- **Danh mục đầy đủ 27,574 files:** Xem tại [`all_files_inventory.txt`](file:///f:/FL%20studio/extracted_targets/flstudio26/all_files_inventory.txt).

#### A. Danh mục các Binary Cốt Lõi (Core Engine, Graphics, Audio I/O & Native Plugins)
| Nhóm | File | Kích thước | Vai trò kỹ thuật & Kiến trúc |
| :--- | :--- | ---: | :--- |
| **DAW Host Core** | [`FL64.exe`](file:///f:/FL%20studio/extracted_targets/flstudio26/FL64.exe) | `597,064 B` | Executable khởi chạy chính (nạp động `FLEngine_x64.dll`) |
| **DAW Host Core** | [`FLEngine_x64.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/FLEngine_x64.dll) | `22,227,016 B` (21.2 MB) | **Trái tim của FL Studio:** Sequencer, Playlist, Piano Roll, Mixer Routing Graph, Automation, Project `.flp` serializer. Biên dịch bằng **Delphi / Embarcadero 64-bit** (`CODE` section `14.55 MB`, entropy `6.37` không bị pack) |
| **SIMD / Math Engine** | [`dsp_ippv2_x64.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/dsp_ippv2_x64.dll) | `39,942,656 B` (38.1 MB) | Thư viện **Intel IPP** tùy biến của Image-Line, chịu trách nhiệm thực thi FFT, Resampling (Sinc interpolation), Convolution, Biquad filter và SIMD buffer mixing |
| **2D Vector UI Engine** | [`Blend2D.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/Blend2D.dll) | `2,065,920 B` (1.97 MB) | Engine đồ họa vector 2D JIT-accelerated (kế thừa Anti-Grain Geometry, export `AggCreateInstance` tại `0x180005160`) |
| **Font Rendering** | [`QuickFontCache_x64.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/QuickFontCache_x64.dll) & [`freetype_x64.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/freetype_x64.dll) | `2.07 MB` & `607 KB` | Hệ thống glyph caching và rasterization cho toàn bộ giao diện FL Studio |
| **Audio Driver Bridge** | [`ILWASAPI2ASIO_x64.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/System/Plugin/ASIO/ILWASAPI2ASIO_x64.dll) | `531,456 B` (519 KB) | Cầu nối chuyển đổi Windows WASAPI (Shared/Exclusive) sang chuẩn ASIO độ trễ thấp (FL Studio ASIO) |
| **Plugin Hosting & Graph** | [`Fruity Wrapper_x64.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/Plugins/Fruity/Effects/Fruity%20Wrapper/Fruity%20Wrapper_x64.dll) | `5,997,056 B` (5.72 MB) | Bộ nạp (Host Wrapper) cho các chuẩn plugin VST2, VST3 và CLAP bên ngoài |
| **Plugin Hosting & Graph** | [`Patcher_x64.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/Plugins/Fruity/Effects/Patcher/Patcher_x64.dll) | `7,056,384 B` (6.73 MB) | Node-based modular audio/parameter routing graph |
| **Native EQ & Dynamics** | [`Fruity Parametric EQ 2_x64.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/Plugins/Fruity/Effects/Fruity%20Parametric%20EQ%202/Fruity%20Parametric%20EQ%202_x64.dll) | `4,036,608 B` (3.85 MB) | EQ 7-band kèm heatmap spectrum analyzer |
| **Native EQ & Dynamics** | [`Maximus_x64.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/Plugins/Fruity/Effects/Maximus/Maximus_x64.dll) | `5,904,384 B` (5.63 MB) | Multiband Maximizer / Compressor / Waveshaper |
| **Native EQ & Dynamics** | [`Emphasis_x64.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/Plugins/Fruity/Effects/Emphasis/Emphasis_x64.dll) | `4,142,080 B` (3.95 MB) | Mastering Limiter thế hệ mới của FL Studio 26 |
| **Native EQ & Dynamics** | [`Frequency Splitter_x64.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/Plugins/Fruity/Effects/Frequency%20Splitter/Frequency%20Splitter_x64.dll) | `4,386,816 B` (4.18 MB) | Bộ phân tần Linear-Phase / Linkwitz-Riley Crossover |
| **Native EQ & Dynamics** | [`Fruity Limiter_x64.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/Plugins/Fruity/Effects/Fruity%20Limiter/Fruity%20Limiter_x64.dll) | `3,868,160 B` (3.69 MB) | Single-band Limiter / Compressor / Noise Gate |
| **AI / Spectral Tools** | [`AudioRestore.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/AudioRestore.dll) & [`iZAudioRestore.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/iZAudioRestore.dll) | `954 KB` & `2.48 MB` | Thư viện khử nhiễu và xử lý phổ âm thanh |

#### B. Kiến trúc PE của [`FLEngine_x64.dll`](file:///f:/FL%20studio/extracted_targets/flstudio26/FLEngine_x64.dll)
- **Trình biên dịch:** Embarcadero Delphi 64-bit (nhận diện qua các section `CODE`, `DATA`, `.pdata`, `.rsrc` và symbol `__dbk_fcall_wrapper`).
- **Exports:** `CreateFruityInstance` (`0x180df7eb0` — khởi tạo toàn bộ DAW Engine).
- **Giao thức Fruity Plugin SDK nội bộ:** Toàn bộ các plugin gốc của FL Studio (`Fruity Parametric EQ 2_x64.dll`, `Maximus_x64.dll`, `Emphasis_x64.dll`, `Patcher_x64.dll`...) đều export hàm `CreatePlugInstance`, giao tiếp trực tiếp với `FLEngine_x64.dll` và gọi hàm toán học từ `dsp_ippv2_x64.dll`.

---

### 3.5. Mục tiêu 5: Ultimate Vocal Remover GUI v5.6 (`UVR5`)
- **Thư mục:** [`f:\FL studio\ultimatevocalremovergui`](file:///f:/FL%20studio/ultimatevocalremovergui)
- **Kiến trúc mã nguồn mở (Python / PyTorch / ONNX):**
  - [`separate.py`](file:///f:/FL%20studio/ultimatevocalremovergui/separate.py) (`61,158 B`): Chứa 3 pipeline tách nhạc cốt lõi:
    1. `SeperateMDX` / `SeperateMDXC` (MDX-Net & MDX23C dựa trên STFT + TFC-TDF-v3 / ONNX Runtime).
    2. `SeperateDemucs` (Hybrid Transformer Demucs v4 — kết hợp miền thời gian + miền tần số).
    3. `SeperateVR` (Vocal Remover kiến trúc Cascaded U-Net trên phổ Magnitude STFT).
  - [`lib_v5/tfc_tdf_v3.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/tfc_tdf_v3.py) & [`lib_v5/mdxnet.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/mdxnet.py): Định nghĩa kiến trúc mạng Time-Frequency Convolution + Time-Distributed Fully-connected.
  - [`lib_v5/spec_utils.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/spec_utils.py) (`35,442 B`): Chứa thuật toán STFT/ISTFT, overlap-add windowing, cùng thuật toán kết hợp Ensemble (`ensembler` — Max Spec, Min Spec, Average, Median) giữa nhiều mô hình.
  - [`models/`](file:///f:/FL%20studio/ultimatevocalremovergui/models): Chứa các file JSON cấu hình hyperparameter (`model_data.json`, `model_name_mapper.json`) cho hàng trăm model `.onnx`, `.ckpt`, `.th`, `.pth`.

---

## 4. Tổng Hợp Những Điểm Lợi Thế Lớn Cho Các Bước Tiếp Theo

1. **Soothe3 có sẵn `soothe3_symbols.map` (`26.18 MB`) + 68 JSON Presets:** Chúng ta có bản đồ tên hàm và địa chỉ RVA đầy đủ của toàn bộ thuật toán Soothe3 mà không bị cản trở bởi lớp nén thực thi.
2. **FabFilter Pro-Q 4 (`FabFilter Pro-Q 4.dll` & `.clap`) không hề bị pack (`.text` entropy `6.37`) và đi kèm `DigiPageTable.xml` (`572 KB`):** Có thể trích xuất toàn bộ công thức hệ số bộ lọc (Filter Coefficients), cấu trúc 24-band Dynamic/Spectral EQ và shader giao diện OpenGL.
3. **Ozone 12 (`iZOzone12Core.dll` `579.8 MB`) không bị pack (`.text` entropy `6.37`) + đi kèm 3 model `.npz` NumPy thuần + bảng `.dat` Exciter + 700 XML Presets:** Có thể phân tích rõ ràng từ kiến trúc IRC Limiter, Multiband Crossover cho tới cấu trúc mạng nơ-ron Stem Separation.
4. **FL Studio 26 (`FLEngine_x64.dll` + `dsp_ippv2_x64.dll` + `Blend2D.dll`):** Cho thấy kiến trúc phân tầng cực kỳ rõ ràng giữa Host Engine, SIMD DSP Library (`dsp_ippv2_x64.dll`) và 2D Vector Graphics (`Blend2D.dll`).
