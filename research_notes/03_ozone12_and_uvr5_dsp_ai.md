# Bước 3: Deep DSP, AI Stem Separation Models & Exciter Lookup Tables Reverse Engineering — iZotope Ozone 12 Advanced & Ultimate Vocal Remover GUI v5.6 (UVR5)

> **Tài liệu nghiên cứu Clean-Room (Bước 3 / 5)**
> - **Mục tiêu phân tích 1**: **iZotope Ozone 12 Advanced (`v12.2.0.1951`)** — Giải mã nhân trung tâm [`iZOzone12Core.dll`](file:///f:/FL%20studio/extracted_targets/ozone12/megacore_component/MEGACORE_WIN/iZOzone12Core.dll) (`579.8 MB`), các VST3 shell wrappers ([`Ozone 12.vst3`](file:///f:/FL%20studio/extracted_targets/ozone12/mothership_unpacked/VST3/VST3_Win/Ozone%2012.vst3), [`Ozone 12 Maximizer.vst3`](file:///f:/FL%20studio/extracted_targets/ozone12/maximizer_unpacked/VST3/VST3_Win/Ozone%2012%20Maximizer.vst3)), hệ thống tệp ảo nhúng `IZRESEMBED` (`469.25 MB`), 23 bảng tham số `DSP/AutomatableParamList_*.txt`, 679 tệp XML Presets trong [`presetFolders_component/`](file:///f:/FL%20studio/extracted_targets/ozone12/presetFolders_component), 3 mô hình `.npz` tách/cân bằng Stem trong [`requiredFiles_component/ssepnpz/`](file:///f:/FL%20studio/extracted_targets/ozone12/requiredFiles_component/ssepnpz), 12 mô hình `.onnx` Band-Split RNN nhúng trong `IZRESEMBED`, và 38 bảng tra cứu phi tuyến bóng đèn điện tử `.dat` (`KState<3, 2, 2, KorenFunction>`) trong [`exciterCache_component/`](file:///f:/FL%20studio/extracted_targets/ozone12/exciterCache_component/ExciterCache/Exciter%20Cache/exciter-triode-48000.000000.dat).
> - **Mục tiêu phân tích 2**: **Ultimate Vocal Remover GUI v5.6 (`UVR5`)** — Giải mã toàn bộ mã nguồn Python / PyTorch / ONNX trong [`separate.py`](file:///f:/FL%20studio/ultimatevocalremovergui/separate.py), [`UVR.py`](file:///f:/FL%20studio/ultimatevocalremovergui/UVR.py), [`lib_v5/`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5), [`demucs/`](file:///f:/FL%20studio/ultimatevocalremovergui/demucs), và [`models/`](file:///f:/FL%20studio/ultimatevocalremovergui/models) cho 3 kiến trúc tách nguồn âm (**MDX-Net / MDX23C**, **Demucs v4 HTDemucs**, **VR Architecture CascadedASPPNet**) cùng thuật toán hợp nhất **Ensemble (`MAX_SPEC`, `MIN_SPEC`, `AVERAGE`)** và khử pha **`invert_audio`**.

---

## 1. Kiến Trúc Nhị Phân iZotope Ozone 12 Advanced (`v12.2.0.1951`)

### 1.1. Kiến trúc 21 VST3 Plugin Shells vs. Nhân Trung Tâm `iZOzone12Core.dll`
Trong [`f:\FL studio\extracted_targets\ozone12`](file:///f:/FL%20studio/extracted_targets/ozone12), hệ thống iZotope Ozone 12 Advanced gồm **21 gói cài đặt plugin** (1 Mothership `Install Ozone 12 Plug-in.exe` + 20 module độc lập, xem [`file_manifest.txt`](file:///f:/FL%20studio/extracted_targets/ozone12/file_manifest.txt)):

1. **Tầng VST3 Shell Wrapper**:
   - [`Ozone 12.vst3`](file:///f:/FL%20studio/extracted_targets/ozone12/mothership_unpacked/VST3/VST3_Win/Ozone%2012.vst3) (`2,115,072` B = `2.02 MB`, SHA-256: `92cf2b434d9c9b7d375d62b764e0048c496a1af8baafc401a20df8b4fbb28c58`)
   - [`Ozone 12 Maximizer.vst3`](file:///f:/FL%20studio/extracted_targets/ozone12/maximizer_unpacked/VST3/VST3_Win/Ozone%2012%20Maximizer.vst3) (`2,115,584` B = `2.02 MB`, SHA-256: `a8f73697466b6d77f1705cefbc9a58558c5ae57a9aec5745e2d55d15554d8966`)
   - Mỗi shell gồm 8 PE sections (`.text`, `IPPCODE`, `.rdata`, `.data`, `.pdata`, `IPPDATA`, `.rsrc`, `.reloc`), xuất 3 hàm chuẩn VST3 (`GetPluginFactory`, `InitDll`, `ExitDll`) và nạp động nhân trung tâm `iZOzone12Core.dll`.

2. **Tầng Nhân DSP & AI Trung Tâm — [`iZOzone12Core.dll`](file:///f:/FL%20studio/extracted_targets/ozone12/megacore_component/MEGACORE_WIN/iZOzone12Core.dll)**:
   - **Kích thước**: `579,805,696` bytes (`552.95 MB`) | **SHA-256**: `44a104fee42335b3ec2b37c30467172f8abcca5b8f86e83a45a270b8a1f08254`
   - **ImageBase**: `0x180000000` (PE32+ x86_64, MSVC C++17/20, **10 PE Sections**):

| # | Section | VirtualAddress (RVA) | VirtualSize | RawOffset | RawSize | Đặc tả nội dung |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `.text` | `0x00001000` | `0x04EFB46E` (`78.98 MB`) | `0x00000600` | `0x04EFB600` | Mã máy DSP cốt lõi, Eigen/MKL, ONNX Runtime (`Ort::*`), `NeuralNet::*`, UI |
| 2 | `IPPCODE` | `0x04EFD000` | `0x001CE450` (`1.81 MB`) | `0x04EFBC00` | `0x001CE600` | Nhân Intel IPP tối ưu SIMD AVX2 / AVX-512 (FFT, FIR, IIR, Vector math) |
| 3 | `.rdata` | `0x050CC000` | `0x010223E4` (`16.13 MB`) | `0x050CA200` | `0x01022400` | C++ RTTI `CompleteObjectLocator`, `ClassHierarchyDescriptor`, VTables (`??_7`), hằng số DSP |
| 4 | `.data` | `0x060EF000` | `0x0041A8D8` (`4.10 MB`) | `0x060EC600` | `0x003CCE00` | `16,779` C++ RTTI `TypeDescriptor` (`??_R0`), biến toàn cục & trạng thái singleton |
| 5 | `.pdata` | `0x0650A000` | `0x00429B40` (`4.16 MB`) | `0x064B9400` | `0x00429C00` | Bảng `RUNTIME_FUNCTION` (x64 SEH exception unwinding) |
| 6 | `IPPDATA` | `0x06934000` | `0x00002270` (`8.61 KB`) | `0x068E3000` | `0x00002400` | Bảng twiddle factors & hằng số cửa sổ của Intel IPP |
| 7 | `minATL` | `0x06937000` | `0x00000018` (`24 B`) | `0x068E5400` | `0x00000200` | Metadata ATL tối giản |
| 8 | `_RDATA` | `0x06938000` | `0x00000130` (`304 B`) | `0x068E5600` | `0x00000200` | Dữ liệu chỉ đọc bổ sung |
| 9 | `.rsrc` | `0x06939000` | `0x1BFA2230` (`447.63 MB`) | `0x068E5800` | `0x1BFA2400` | Chứa Virtual Filesystem **`IZRESEMBED`** (`469,253,664` B tại file offset `0x068E5B84`) |
| 10 | `.reloc` | `0x228DC000` | `0x0006A554` (`425.33 KB`) | `0x22887C00` | `0x0006A600` | Bảng relocation 64-bit (`IMAGE_REL_BASED_DIR64`) |

### 1.2. Giải mã hệ thống tệp ảo `IZRESEMBED` trong `.rsrc`
Tại file offset `0x068E5B84` (ngay sau chuỗi định danh `IZRESEMBED`), archive nội bộ rộng `469,253,664` bytes (`0x068E5B84..0x228699A4`) lưu **`2,970` tệp** theo cấu trúc 2 vùng tách biệt:
1. **Vùng Table of Contents (TOC)** (`rel_off = 0x00000000..0x00025A9C`, dài `154,268` B, pad 0 tới `0x00025AA0`): Gồm `2,970` bản ghi liên tiếp, mỗi bản ghi có cấu trúc:
   $$\underbrace{\text{path\_utf8}\ \| \ \texttt{0x00}}_{\text{null-terminated C-string}} \ \| \ \underbrace{\text{rel\_data\_offset}}_{\texttt{u32 LE (16B-aligned)}} \ \| \ \underbrace{\text{data\_size}}_{\texttt{u32 LE}}$$
2. **Vùng Data Payloads** (`rel_off = 0x00025AA0..0x1BF83E04`, tức `file_off = 0x0690B624..0x22869988`): Chứa nội dung nguyên bản của `2,970` tệp (căn lề 16 bytes), bao gồm:
   - **23 tệp `DSP/AutomatableParamList_*.txt`**: Bảng TSV 3 cột `[Host Automation Name]\t[ElementID]\t[ParamID]` ánh xạ tham số tự động hóa VST3/AAX/AU cho toàn bộ các module.
   - **12 mô hình mạng nơ-ron `.onnx`** (`219,552,803` B = `209.38 MB` tổng cộng, chi tiết ở Mục 4.2).
   - **2 mô hình `.npz` của Master Assistant / Referencing** (`26,376,589` B tổng cộng, chi tiết ở Mục 4.3).
   - **26 tệp XML preset mục tiêu** (`Audiolens Targets/*.xml`, `MasterAssistant/*.xml`) và **2,761 tài nguyên đồ họa/shaders/fonts**.

---

## 2. Bản Đồ RTTI, VTable & Phân Cấp Lớp DSP Của Ozone 12

Quét toàn bộ `16,779` `TypeDescriptor` trong `.data` (`0x1860EF000..`) và đối chiếu `CompleteObjectLocator` trong `.rdata` (`0x1850CC000..`) cho thấy kiến trúc hướng đối tượng 2 tầng rõ rệt:
- **Lớp cơ sở phần tử DSP**: `DSP::ElementImpl` (`VTable = 0x1856DF158`, `vfn[0] = 0x182660C10`, `vfn[1] = 0x182660EB0`, `vfn[58] = 0x182663B40`, `vfn[59] = 0x182663B50`, `vfn[61] = 0x182663B70`, `vfn[62] = 0x182663B80`).
- **Lớp cơ sở chuỗi Mastering**: `DSP::OzoneChainElement` (`VTable = 0x18572E258`, kế thừa `DSP::ElementImpl`, ghi đè `vfn[61] = 0x18343CDB0` và `vfn[62] = 0x18343CE00`).
- Mọi module trong Ozone 12 gồm một lớp vỏ **`DSP::Ozone*`** (kế thừa `DSP::OzoneChainElement`, điều phối tham số/metering/bypass tại `vfn[58]` và `vfn[59]`) bọc bên ngoài một hoặc nhiều nhân xử lý **`DSP::Shared::*`** (thực thi DSP mẫu/khối tại `vfn[58]`, `vfn[59]`, `vfn[61]`, `vfn[62]`):

| Module / Class (C++ RTTI) | VTable VA | `vfn[0]` (`dtor`) | `vfn[58]` (`reset/init`) | `vfn[59]` (`process_ element`) | `vfn[61]` (`clear_ state`) | `vfn[62]` (`process_ dsp_block`) | Vai trò trong kiến trúc DSP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `DSP::ElementImpl` | `0x1856DF158` | `0x182660C10` | `0x182663B40` | `0x182663B50` | `0x182663B70` | `0x182663B80` | Lớp cơ sở gốc của mọi phần tử DSP |
| `DSP::OzoneChainElement` | `0x18572E258` | `0x18343BB50` | `0x182663B40` | `0x182663B50` | `0x18343CDB0` | `0x18343CE00` | Lớp cơ sở của các module trên thanh chuỗi Ozone 12 |
| `DSP::OzoneMaximizer` | `0x185736808` | `0x183654030` | `0x183656140` | `0x183656210` | `0x18343CDB0` | `0x18343CE00` | Vỏ module Maximizer (quản lý IRC, SoftClip, UpwardComp, Dither) |
| `DSP::Shared::IRC::Maximizer` | `0x185767808` | `0x183F64F20` | `0x183F6A320` | `0x183F6A730` | `0x183F68F60` | `0x183F696A0` | Nhân Maximizer cốt lõi (IRC I–V, SoftClip, LowLevelBoost) |
| `DSP::Shared::IRC::IRCLimiter` | `0x1856BE678` | `0x182245310` | `0x18224D3A0` | `0x18224D3F0` | `0x182663B70` | `0x182663B80` | Bộ giới hạn IRC toàn dải / đa băng |
| `DSP::Shared::IRC::LimEnvelopePsycho` | `0x185734588` | `0x183598610` | `0x182663B40` | `0x182663B50` | `0x182663B70` | `0x182663B80` | Bộ dò đường bao tâm sinh lý học (IRC III / IV) |
| `DSP::OzoneImager` | `0x185739B60` | `0x1836C85C0` | `0x1836C9340` | `0x1836C93A0` | `0x18343CDB0` | `0x18343CE00` | Vỏ module Stereo Imager 4 dải |
| `DSP::Imager` | `0x18573A070` | `0x1836D08B0` | `0x1836D2110` | `0x1836D2190` | `0x1836D0B50` | `0x1836D0E40` | Nhân xử lý M/S Width, Stereoizer I/II & Recover Sides |
| `DSP::OzoneExciter` | `0x1856A9B78` | `0x181D6C610` | `0x181D6E5A0` | `0x181D6E5C0` | `0x18343CDB0` | `0x18343CE00` | Vỏ module Exciter 4 dải + Oversampling |
| `DSP::Shared::Exciter` | `0x1856AA150` | `0x181D77E70` | `0x181D7AE20` | `0x181D7AE70` | `0x181D7ADF0` | `0x181D7A130` | Nhân bão hòa phi tuyến (Analog, Retro, Tape, Tube, Warm, Triode, Dual Triode) |
| `DSP::OzoneEqualizer` | `0x18572EE70` | `0x183465840` | `0x18346C7E0` | `0x18346CA60` | `0x18343CDB0` | `0x18343CE00` | Vỏ module Equalizer 8 băng (Analog IIR / Digital FIR) |
| `DSP::Shared::EqualizerIIR` | `0x185711960` | `0x182FCBF00` | `0x182663B40` | `0x182663B50` | `0x182FCFB80` | `0x182FCF700` | Nhân lọc Biquad IIR pha tối thiểu |
| `DSP::Shared::EqualizerFIR` | `0x18572F230` | `0x1834AA300` | `0x1834B3EE0` | `0x1834B3F80` | `0x1834AE240` | `0x1834AE110` | Nhân lọc FIR pha tuyến tính / pha hỗn hợp |
| `DSP::OzoneDynEq` | `0x185734AC0` | `0x1835AB460` | `0x1835AD5C0` | `0x1835AD620` | `0x18343CDB0` | `0x18343CE00` | Vỏ module Dynamic EQ 6 băng |
| `DSP::Shared::DynamicEQ` | `0x1857112E0` | `0x182FA6BD0` | `0x182FA90E0` | `0x182FA9130` | `0x182FA8E30` | `0x182FA83C0` | Nhân lọc Biquad động theo đường bao RMS/Peak |
| `DSP::OzoneStabilizer` | `0x185696008` | `0x181A04A20` | `0x181A08860` | `0x181A08870` | `0x18343CDB0` | `0x18343CE00` | Cân bằng phổ thích ứng thông minh (Shape / Cut + Tame Transients) |
| `DSP::OzoneImpact` | `0x185692C40` | `0x181934AC0` | `0x181938730` | `0x1819388B0` | `0x18343CDB0` | `0x18343CE00` | Điều khiển vi động lực (Microdynamics) 4 băng |
| `DSP::OzoneClarity` | `0x18569CA60` | `0x181B80820` | `0x181B82F70` | `0x181B82F80` | `0x18343CDB0` | `0x18343CE00` | Khử đục phổ tâm sinh lý học (Psychoacoustic Spectral Unmasking) |
| `DSP::OzoneLowEndFocus` | `0x185694018` | `0x181992200` | `0x181995D50` | `0x181995D60` | `0x18343CDB0` | `0x18343CE00` | Vỏ module Low End Focus (`Processor` tại `VT=0x185693D40`) |
| `DSP::Shared::SpectralContrast` | `0x1856944D0` | `0x1819C5F50` | `0x182663B40` | `0x182663B50` | `0x1819C60C0` | `0x1819C60D0` | Nhân tách tương phản phổ dải trầm (Punchy / Smooth) |
| `DSP::OzoneSpectralShaper` | `0x185738C88` | `0x183689630` | `0x18368A430` | `0x18368A440` | `0x18343CDB0` | `0x18343CE00` | Vỏ Spectral Shaper (`DeEsser` tại `VT=0x185738FA0`, `SpectralDeEsser` tại `VT=0x185739368`) |
| `DSP::OzoneStemEQ` | `0x1857397B0` | `0x18369D430` | `0x18369FF80` | `0x18369FF90` | `0x18343CDB0` | `0x18343CE00` | EQ độc lập trên 4 Stems (Vocals, Drums, Bass, Other) |
| `DSP::OzoneMasterRebalance` | `0x18569B9E8` | `0x181B1EEB0` | `0x181B1FA00` | `0x181B21110` | `0x18343CDB0` | `0x18343CE00` | Cân bằng Gain thời gian thực cho Vocals / Bass / Drums |
| `DSP::OzoneUnlimiter` | `0x18568D6E0` | `0x181843C80` | `0x181845C40` | `0x181845C90` | `0x18343CDB0` | `0x18343CE00` | Khôi phục đỉnh động lực bị nén/clip bằng mạng `delimit` ONNX |
| `DSP::OzoneBassControl` | `0x18569B330` | `0x181AE9110` | `0x181AEC1D0` | `0x181AEC380` | `0x18343CDB0` | `0x18343CE00` | Module mới của Ozone 12: cân bằng Sustain Power & Peak Punch dải trầm |
| `DSP::OzoneDynamics` | `0x185735500` | `0x1835B41A0` | `0x1835BD910` | `0x1835BDA50` | `0x18343CDB0` | `0x18343CE00` | Compressor / Limiter / Expander 4 dải |
| `SourceSeparationTwo::OnnxInference` | `0x18572F9D8` | `0x1834E9D10` | — (`vfn[1]=0x1834E9F10`) | — (`vfn[2]=0x1834E99D0`) | — | — | Engine suy luận ONNX Runtime 3-slot (`dtor`, `process`, `reset`) |
| `NeuralNet::GRU` | `0x18572E0A8` | `0x1834213A0` | — (`vfn[1]=0x183421790`) | — (`vfn[2]=0x183421F10`) | — | — | Lớp GRU 3-slot (`dtor`, `forward`, `reset_state`) cho `.npz` |

*(Ngoài ra còn có 4 cặp module Vintage: `OzoneTubeLimiter`/`Shared::TubeLimiter` tại `0x185736120`/`0x185737458`, `OzoneVintageTape`/`Shared::Tape` tại `0x1856AE538`/`0x1856AE8F0`, `OzoneVintageCompressor`/`Shared::VintageDynamics` tại `0x1856AF328`/`0x1856AF730`, và `OzoneVintageEQ`/`Shared::PultecEQ` tại `0x1856AD0E8`/`0x1856AD538`.)*

---

## 3. Hệ Thống Tham Số Thực Tế & Giải Mã Thuật Toán DSP Cốt Lõi

### 3.1. Maximizer (`ElementID = "Maximizer"`) — IRC I/II/III/IV/V, Upward Compression & Soft Clip
- **Tham số thực tế trong `AutomatableParamList_Maximizer.txt` & 679 XML Presets**:
  `Mode` (`0`=IRC I, `1`=IRC II, `2`=IRC III, `3`=IRC IV, `4`=IRC V, `5`=IRC LL), `Gain` ($[-20, +20]\text{ dB}$) / `Threshold` ($[-20, 0]\text{ dB}$), `Margin` (Ceiling $[-20, 0]\text{ dB}$), `Character` ($[0, 10]$), `Max Character` (`0`=Pumping, `1`=Balanced, `2`=Crisp, `3`=Classic, `4`=Modern, `5`=Transient), `EnableLowLevelBoost` (`bool`), `LowLevelBoostWetAmount` ($[0, 100]\%$), `EnableSoftClipping` (`bool`), `SoftClipMix` ($[0, 100]\%$), `SoftClipMode` (`0`=Light, `1`=Medium, `2`=Heavy), `Enable Transient Shaping` (`bool`), `Transient Shaping` ($[-100, +100]\%$), `Stereo Link` ($[0, 100]\%$ sustain stereo independence), `Stereo Transient Link Amount` ($[0, 100]\%$), `Prevent Intersample Clipping` (`bool`).
- **Chuỗi xử lý DSP trong `DSP::Shared::IRC::Maximizer::process` (`0x183F6A730`), `callback` (`0x183F696A0`) & `0x183F81D90..0x183F82500`**:
  1. **Upward Compression (`EnableLowLevelBoost` / `LowLevelBoostWetAmount`)**: Nâng mức tín hiệu nhỏ dưới ngưỡng $T_{\text{low}} = -24\text{ dB}$ (trên ngưỡng noise floor $-60\text{ dBFS}$) trước khi vào bộ giới hạn mà không làm thay đổi đỉnh trên $T_{\text{ceil}}$:
     $$G_{\text{up}}(E_{\text{rms}}[n]) = 1 + \alpha_{\text{wet}} \cdot \frac{\max(0,\; T_{\text{low}} - E_{\text{rms}}[n])}{R_{\text{up}} \cdot (1 + e^{(E_{\text{rms}}[n] - T_{\text{knee}})/\sigma})}$$
  2. **Pre-Limiter Soft Clipper (`EnableSoftClipping`)**: Áp dụng hàm bão hòa đa thức/tanh có ngưỡng knee $k_m \in \{0.85, 0.70, 0.50\}$ ứng với `SoftClipMode` $\in \{0, 1, 2\}$ (Light / Medium / Heavy), pha trộn theo tỷ lệ $w_{\text{sc}} = \text{SoftClipMix}/100$:
     $$y_{\text{sc}}[n] = (1 - w_{\text{sc}})\,x[n] + w_{\text{sc}}\,\text{sgn}(x[n])\begin{cases} |x[n]|, & |x[n]| \le k_m \\ k_m + (1 - k_m)\tanh\!\left(\frac{|x[n]| - k_m}{1 - k_m}\right), & |x[n]| > k_m \end{cases}$$
  3. **Tách Transient / Sustain bằng Max-Heap (`DSP::MaxHeap` / `DSP::CircBuf` tại `0x183F81EE4..0x183F81F25`)**: Duy trì bộ đệm lookahead $L = \lfloor 0.005 f_s \rfloor$ mẫu cùng cây phân đoạn cực đại trượt (`shr r8, 1; cmovg ecx, [r10+r8*4]`) để tìm cực đại cục bộ. Tín hiệu được phân rã thành $x[n] = x_{\text{trans}}[n] + x_{\text{sust}}[n]$ qua tỷ số đường bao nhanh ($\tau_a = 0.2\text{ ms}, \tau_r = 15\text{ ms}$) và chậm ($\tau_a = 15\text{ ms}, \tau_r = 200\text{ ms}$), cho phép liên kết Stereo độc lập giữa Transient (`Stereo Transient Link Amount`) và Sustain (`Stereo Link`).
  4. **5 Thuật toán Intelligent Release Control (IRC I – IRC V tại `0x183F69CB2`)**:
     - **IRC I (`eax == 0` $\to$ `0x183F6AF90`)**: Tính biên độ đỉnh đa kênh $x_{\text{peak}}[n] = \max_c |x_c[n]|$, điều khiển thời gian nhả $\tau_{\text{rel}}[n]$ phụ thuộc năng lượng crest factor toàn dải (`[rbx+14CCh..14D4h]`) và lọc làm mượt gain $g[n]$ bằng bộ lọc 1 cực tấn công/nhả bất đối xứng trước khi nhân với tín hiệu trễ trong ring buffer lookahead `[rbx+14F0h]`.
     - **IRC II (`eax == 1` $\to$ `0x183F6B1D0`)**: Tập trung bảo toàn transient sắc nét (trống/percussion) thông qua gain mục tiêu tức thời $g_{\text{target}}[n] = \min(1.0, T_{\text{thresh}} / x_{\text{env}}[n])$ kết hợp bộ lọc FIR/minimum-window lookahead (`0x1802B6640`) và làm mượt 2 giai đoạn.
     - **IRC III (`eax == 2`, `LimEnvelopePsycho` tại `0x185734588`)**: Phân tích ngưỡng che lấp tâm sinh lý học trên miền STFT/Bark để phân bổ lượng giảm độ lợi (Gain Reduction) vào những dải tần ít gây méo nghe thấy nhất.
     - **IRC IV (`eax == 3`, `Max Character`: Classic / Modern / Transient)**: Chia tín hiệu thành các băng tần Psychoacoustic (dùng ngân hàng bộ lọc Linkwitz-Riley / STFT), giới hạn độc lập từng băng kèm tách đường Transient/Sustain để khi kick/sub-bass chạm ngưỡng không kéo tụt vocal/hi-hat, sau đó tổng hợp lại qua limiter toàn dải cuối cùng.
     - **IRC V (`eax == 4` $\to$ `0x183F69CE4`, gọi `0x1835825F0` — Mới trong Ozone 12)**: Kết hợp phân tích đa băng mật độ cao với điều chế đường bao hình thái học (Transient/Sustain envelope morphing) và Upward Compression nội tại trước/sau khâu giới hạn.
     - **IRC Low Latency (`Mode = 5`, `0x183F6AAA0`)**: Thay thế bộ lọc lookahead dài bằng bộ lọc minimum-phase IIR + lookahead ngắn.
  5. **True-Peak Oversampling (`Prevent Intersample Clipping` tại `0x183F69E6E..0x183F69FCD`) & MBIT+ Dither (`0x183F63430`)**: Nội suy 4× bằng bộ lọc FIR đa pha 48-tap (`IPPCODE`) để bắt các đỉnh liên mẫu (Intersample Peaks), kết thúc bằng bộ lượng tử hóa **MBIT+ Dither** (`Dither Amount`, `Noise Shaping` curve psychoacoustic, `Bit Depth` $\in \{8, 16, 20, 24\}$, `Filter DC Offset`, `Harmonic Suppression`, `Auto-Blanking`).

### 3.2. Exciter (`ElementID = "Multi-Band Exciter"`) & Giải Mã Bảng Tra Cứu Bóng Đèn `.dat`
- **Tham số thực tế (`AutomatableParamList_Exciter.txt` & XML)**:
  4 băng tần (`Band 1..4 Enable`, `Crossover Cutoff 1..3`), mỗi băng có `Band b Post FilterPower` (Amount $[0, 100]\%$), `Band b WetMix` ($[0, 100]\%$), `Band b Circuit Type` (`0`=Analog, `1`=Retro, `2`=Tape, `3`=Warm, `4`=Triode, `5`=Dual Triode, `6`=Tube), cùng `PostFilterFreq` và `PostFilterGain`.
- **Giải mã cấu trúc nhị phân chính xác của 38 tệp `.dat` (`exciter-triode-*.dat` & `exciter-dualtriode-*.dat`)**:
  Trong [`file_manifest.txt`](file:///f:/FL%20studio/extracted_targets/ozone12/file_manifest.txt) có **38 tệp `.dat`** (19 tần số lấy mẫu từ `8000 Hz` đến `768000 Hz` × 2 mạch `triode` `2,074,532 B` / `dualtriode` `4,149,044 B`). RTTI trong `iZOzone12Core.dll` xác nhận lớp thực thi là **`DSP::Shared::KState<3, 2, 2, DSP::Shared::KorenFunction>`** (`dtor = 0x181DB7650`, `reset = 0x181DB8700`, `process = 0x181DB8BA0`, `KorenFunction = 0x181DC03D0`) — phương pháp **DK-method (Discretized Kirchhoff Nodal Analysis)** cho mạch khuếch đại đèn 3 cực 12AX7 với $N_x = 3$ tụ điện trạng thái, $N_u = 2$ nguồn áp đầu vào $[V_{\text{in}}, V_{B+}]^T$, và $N_{nl} = 2$ nhánh dòng phi tuyến $[I_g(V_{gk}), I_p(V_{gk}, V_{pk})]^T$.
  - Hằng số vật lý bóng đèn 12AX7 của phương trình Norman Koren trong `DSP::Shared::KorenFunction` (`0x181DC03D0`):
    $$\mu = 100.0,\quad k_g = 1060.0,\quad k_p = 600.0,\quad k_{vb} = 300.0,\quad x = 1.4,\quad R_{gi} = 2000\,\Omega$$
    $$E_1 = \frac{V_{pk}}{k_p} \ln\!\left(1 + \exp\!\left(k_p \left(\frac{1}{\mu} + \frac{V_{gk}}{\sqrt{k_{vb} + V_{pk}^2}}\right)\right)\right),\quad I_p = \frac{E_1^x}{k_g}\,(1 + \text{sgn}(E_1))$$
  - Cấu trúc byte tuyệt đối của [`exciter-triode-48000.000000.dat`](file:///f:/FL%20studio/extracted_targets/ozone12/exciterCache_component/ExciterCache/Exciter%20Cache/exciter-triode-48000.000000.dat) (`2,074,532` bytes = `0x1FA7A4` B):

| Offset (Hex) | Kích thước | Kiểu dữ liệu | Giá trị thực tế trong `exciter-triode-48000.000000.dat` | Ý nghĩa vật lý & Toán học (DK-Method `KState<3,2,2>`) |
| :--- | :--- | :--- | :--- | :--- |
| `0x0000..0x0003` | `4 B` | `u32` | `0x8BAD43FC` | Magic signature của định dạng bảng K-method iZotope |
| `0x0004..0x0007` | `4 B` | `u32` | `4` | Phiên bản cấu trúc serialization (`version = 4`) |
| `0x0008..0x000F` | `8 B` | `f64` | `-0.04588765735184097` | Điện áp lệch tĩnh đầu ra ($V_{\text{dc,out}}$) |
| `0x0010..0x0013` | `4 B` | `u32` | `0` | Cờ chế độ mạch (`0 = Triode` 1 tầng, `1 = DualTriode` 2 tầng nối tiếp) |
| `0x0014..0x0033` | `32 B` | `u32[2]` + `f64[3,1]` | `[6.68e-14, 196.282586, 1.555761]^T` | Điện áp tĩnh trên 3 tụ điện $\mathbf{x}_{\text{dc}} = [V_{C1}, V_{C2(\text{plate})}, V_{C3(\text{cathode})}]^T\text{ (V)}$ |
| `0x0034..0x004B` | `24 B` | `u32[2]` + `f64[2,1]` | `[0.0, 300.0]^T` | Nguồn áp tĩnh $\mathbf{u}_{\text{dc}} = [V_{\text{in,dc}} = 0\text{ V},\; V_{B+} = 300\text{ V}]^T$ |
| `0x004C..0x0063` | `24 B` | `u32[2]` + `f64[2,1]` | `[9.06e-25, 0.001037174]^T` | Dòng đèn tĩnh $\mathbf{i}_{nl,\text{dc}} = [I_{g,\text{dc}}, I_{p,\text{dc}} = 1.037\text{ mA}]^T$ ($\Rightarrow R_k = 1.5\text{ k}\Omega, R_p = 100\text{ k}\Omega$) |
| `0x0064..0x006B` | `8 B` | `f64` | `48000.0` | Tần số lấy mẫu $f_s$ (Hz) |
| `0x006C..0x00BB` | `80 B` | `u32[2]` + `f64[3,3]` | $\mathbf{A}_c$: $A_{00}=-146.433, A_{11}=-2.353\times 10^6, A_{22}=-26.667$ | Ma trận trạng thái miền liên tục $\dot{\mathbf{x}} = \mathbf{A}_c\mathbf{x} + \mathbf{B}_c\mathbf{u} + \mathbf{C}_c\mathbf{i}_{nl}$ ($C_3 = 25\,\mu\text{F}$) |
| `0x00BC..0x00F3` | `56 B` | `u32[2]` + `f64[3,2]` | $\mathbf{B}_c \in \mathbb{R}^{3\times 2}$ | Ma trận đầu vào miền liên tục |
| `0x00F4..0x012B` | `56 B` | `u32[2]` + `f64[3,2]` | $\mathbf{C}_c \in \mathbb{R}^{3\times 2}$ ($C_{20}=C_{21}=40000 = 1/C_3$) | Ma trận ghép dòng phi tuyến miền liên tục |
| `0x012C..0x0163` | `56 B` | `u32[2]` + `f64[2,3]` | $\mathbf{D}_x \in \mathbb{R}^{2\times 3}$ | Ma trận chiếu trạng thái sang áp nhánh $\mathbf{v}_{nl} = \mathbf{D}_x\mathbf{x} + \mathbf{E}_u\mathbf{u} + \mathbf{F}_i\mathbf{i}_{nl}$ |
| `0x0164..0x018B` | `40 B` | `u32[2]` + `f64[2,2]` | $\mathbf{E}_u \in \mathbb{R}^{2\times 2}$ (`0.588235, 0.411765`) | Ma trận chiếu nguồn áp sang áp nhánh |
| `0x018C..0x01B3` | `40 B` | `u32[2]` + `f64[2,2]` | $\mathbf{F}_i \in \mathbb{R}^{2\times 2}$ (`-41176.4706` mọi phần tử) | Trở kháng tương đương Thevenin nhìn từ cổng phi tuyến ($-41.18\text{ k}\Omega$) |
| `0x01B4..0x0203` | `80 B` | `u32[2]` + `f64[3,3]` | $10^{-5}\mathbf{I}_3$ | Ma trận điều hòa số học (Regularization $\epsilon \mathbf{I}_3, \epsilon = 10^{-5}$) |
| `0x0204..0x023B` | `56 B` | `u32[2]` + `f64[2,3]` | $\mathbf{D}_d \in \mathbb{R}^{2\times 3}$ | Ma trận rời rạc hóa Trapezoidal tính tọa độ tra bảng $\mathbf{p}[n]$ từ $\mathbf{x}[n-1]$ |
| `0x023C..0x0263` | `40 B` | `u32[2]` + `f64[2,2]` | $\mathbf{E}_d \in \mathbb{R}^{2\times 2}$ | Ma trận đóng góp $\mathbf{u}[n]$ vào $\mathbf{p}[n]$ |
| `0x0264..0x028B` | `40 B` | `u32[2]` + `f64[2,2]` | $\mathbf{F}_d \in \mathbb{R}^{2\times 2}$ | Ma trận đóng góp $\mathbf{u}[n-1]$ vào $\mathbf{p}[n]$ |
| `0x028C..0x02B3` | `40 B` | `u32[2]` + `f64[2,2]` | $\mathbf{G}_d \in \mathbb{R}^{2\times 2}$ | Ma trận đóng góp $\mathbf{i}_{nl}[n-1]$ vào $\mathbf{p}[n]$ |
| `0x02B4..0x0303` | `80 B` | `u32[2]` + `f64[3,3]` | $\mathbf{A}_d \in \mathbb{R}^{3\times 3}$ ($\lambda = 0.999455, -0.921501, 0.999445$) | Ma trận chuyển trạng thái rời rạc $\mathbf{x}[n]$ |
| `0x0304..0x033B` | `56 B` | `u32[2]` + `f64[3,2]` | $\mathbf{B}_d \in \mathbb{R}^{3\times 2}$ | Ma trận đầu vào rời rạc cho $\mathbf{x}[n]$ |
| `0x033C..0x0373` | `56 B` | `u32[2]` + `f64[3,2]` | $\mathbf{C}_d \in \mathbb{R}^{3\times 2}$ | Ma trận dòng phi tuyến rời rạc cho $\mathbf{x}[n]$ |
| `0x0374..0x0393` | `32 B` | `f64[4]` | `[-600.0, 300.0, 240.0, 303.0]` | Cận biên lưới 2D: $p_{1,\min}=-600, p_{2,\min}=300, p_{1,\max}=240, p_{2,\max}=303\text{ (V)}$ |
| `0x0394..0x03A3` | `16 B` | `u32[2]` + `u64` | `360, 360, 129600` | Kích thước lưới tra cứu $N_1 = 360, N_2 = 360$, tổng số nút $= 129,600$ |
| `0x03A4..0x1FA7A3` | `2,073,600 B` | `f64[360, 360, 2]` | Mảng 2D LUT $[I_g(p_1,p_2),\; I_p(p_1,p_2)]$ | Nghiệm tiền giải phương trình ẩn Koren $\mathbf{i}_{nl} = \mathcal{K}(\mathbf{p} + \mathbf{F}_i\mathbf{i}_{nl})$ |

- **Chu trình tính toán từng mẫu (Per-Sample DK-Method Loop tại `0x181DB8BA0..0x181DB9515`)**:
  1. Tính tọa độ biến đổi tuyến tính $\mathbf{p}[n] = [p_1, p_2]^T \in \mathbb{R}^2$:
     $$\mathbf{p}[n] = \mathbf{D}_d\,\mathbf{x}[n-1] + \mathbf{E}_d\,\mathbf{u}[n] + \mathbf{F}_d\,\mathbf{u}[n-1] + \mathbf{G}_d\,\mathbf{i}_{nl}[n-1]$$
  2. Chuẩn hóa $(p_1, p_2)$ vào miền chỉ số thực $(u, v) \in [0, 359]^2$:
     $$u = 359 \cdot \text{clamp}\!\left(\frac{p_1 - (-600.0)}{240.0 - (-600.0)},\, 0,\, 1\right), \quad v = 359 \cdot \text{clamp}\!\left(\frac{p_2 - 300.0}{303.0 - 300.0},\, 0,\, 1\right)$$
     Nội suy song tuyến tính (bilinear interpolation) trên bảng `LUT[360][360][2]` để thu ngay vector dòng đèn $\mathbf{i}_{nl}[n] = [I_g[n], I_p[n]]^T$ mà không cần lặp Newton-Raphson lúc chạy.
  3. Cập nhật trạng thái tụ điện (kèm chống denormal $|x_r| < 10^{-15} \to 0$) và xuất điện áp plate đã khử DC:
     $$\mathbf{x}[n] = \mathbf{A}_d\,\mathbf{x}[n-1] + \mathbf{B}_d(\mathbf{u}[n] + \mathbf{u}[n-1]) + \mathbf{C}_d(\mathbf{i}_{nl}[n] + \mathbf{i}_{nl}[n-1]), \quad y[n] = (x_1[n] - x_{1,\text{dc}}) - V_{\text{dc,out}}$$

### 3.3. Các Module Mastering Cốt Lõi Khác Của Ozone 12
1. **Stereo Imager (`ElementID = "Stereo Imager"`, `DSP::Imager` tại `0x18573A070`)**:
   - **Tham số**: `Band 1..4 Width Percent` ($[-100, +100]\%$), `Enable Stereoizer`, `Stereoizer Mode` (`0`=Stereoize I velvet-noise/Haas phase-compensated decorrelation, `1`=Stereoize II linear-phase comb/all-pass network), `Stereoizer Delay` ($[0, 30]\text{ ms}$), `Recover Sides Enabled`, `Recover Sides Gain Offset (dB)`.
   - **DSP**: Phân tách 4 dải Linkwitz-Riley bậc 4. Trong mỗi dải, biến đổi $M = (L+R)/\sqrt{2}, S = (L-R)/\sqrt{2}$. Khi bật `Recover Sides` và $w_b < 0$ (thu hẹp ảnh nổi dải trầm), năng lượng thành phần $S$ bị cắt giảm $\Delta E_S = (1 - (1 + w_b)^2)|S|^2$ được bù pha và cộng vào kênh $M$ (điều chỉnh bởi `Recover Sides Gain Offset (dB)`) để bảo toàn năng lượng nhạc cụ lệch bên khi gộp mono.
2. **Stabilizer (`ElementID = "Stabilizer"`, `DSP::OzoneStabilizer` tại `0x185696008`)**:
   - **Tham số**: `Mode` (`0`=Shape, `1`=Cut), `Target` (`1..10` ứng với các đường cong đích All-Around, EDM, Hip-Hop, Pop, Rock...), `Amount` ($[0, 100]$), `Speed` ($[0, 100]$), `FreqSmoothing` ($[0, 100]$), `LFStrength`, `MFStrength`, `HFStrength`, `TameTransients`.
   - **DSP**: Tính mật độ phổ công suất trên thang Bark/Mel, làm mượt theo trục tần số bằng `FreqSmoothing` và so sánh với dải phân vị mục tiêu $\mu_{\text{target}}(f) \pm \sigma_{\text{target}}(f)$. Ở chế độ `Shape`, áp dụng cả boost và cut nhẹ quanh đường cong đích; ở chế độ `Cut`, chỉ cắt các cộng hưởng vượt ngưỡng trên, kết hợp bộ nén nhanh `TameTransients` cho phần gai xung.
3. **Impact (`ElementID = "Impact"`, `DSP::OzoneImpact` tại `0x185692C40`)**:
   - **Tham số**: `Band 1..4 Amount` ($[-100, +100]$), `Envelope` ($[10, 500]\text{ ms}$), `HostSyncEnvelope`, `HostSyncEnvelopeEnabled`, `Crossover Cutoff 1..3`.
   - **DSP**: Trong mỗi băng $b$, tính tỷ số phong bì vi động lực $D_b[n] = \text{env}_{\text{fast}}[n] / (\text{env}_{\text{slow}}[n] + \epsilon)$ với hằng số nhả $\tau_{\text{rel}} = \text{Envelope (ms)}$ (hoặc đồng bộ nhịp BPM `HostSyncEnvelope`), rồi nhân hệ số khuếch đại động $G_b[n] = (D_b[n])^{\text{Amount}_b / 100}$ (dương: mở rộng punch; âm: nén dày microdynamics).
4. **Clarity (`ElementID = "Clarity"`, `DSP::OzoneClarity` tại `0x18569CA60`)**:
   - **Tham số**: `Amount` ($[0, 100]$), `Tilt` ($[-6, +6]\text{ dB/oct}$), `Attack` ($[1, 200]\text{ ms}$), `Release` ($[10, 1000]\text{ ms}$), `ActionRegionLowCutoff`, `ActionRegionHighCutoff`.
   - **DSP**: Dựa trên mô hình độ to chuẩn ISO 226 / Moore-Glasberg trên lưới STFT trong miền $[f_{\text{low}}, f_{\text{high}}]$, điều chỉnh biên độ từng bin hướng tới đường cong phổ nghiêng `Tilt` sao cho tổng phương sai che lấp giữa các băng tới hạn (critical bands) giảm thiểu mà không gây hiệu ứng bơm (pumping).
5. **Low End Focus (`ElementID = "Low End Focus"`, `DSP::Shared::SpectralContrast` tại `0x1856944D0`)**:
   - **Tham số**: `CharacterMode` (`0`=Punchy, `1`=Smooth), `Strength` ($[-100, +100]$), `Gain` ($[-12, +12]\text{ dB}$), `ActionRegionLowCutoff` ($20\text{ Hz}$), `ActionRegionHighCutoff` ($[80, 300]\text{ Hz}$).
   - **DSP**: Tách phổ vùng trầm $[f_{\text{low}}, f_{\text{high}}]$ thành phần xung lực (transient/high-contrast spectral bins) và phần nền kéo dài (sustain/resonant bins), tăng/giảm tỷ lệ tương phản phổ theo `Strength` với hằng số thời gian nhanh (`Punchy`) hoặc chậm (`Smooth`).
6. **Spectral Shaper (`ElementID = "Spectral Shaper"`, `DSP::OzoneSpectralShaper` tại `0x185738C88`)**:
   - **Tham số**: `Amount`, `THRESHOLD`, `COMPRESSION_MODE` (`0`=Light 2:1, `1`=Medium 4:1, `2`=Heavy 8:1), `ATTACK_TIME`, `RELEASE_TIME`, `SPECTRAL_TILT`, `LOW_CUTOFF`, `HIGH_CUTOFF`.
   - **DSP**: Bộ nén động lực phân giải cao từng bin STFT nằm trong cửa sổ $[f_{\text{low}}, f_{\text{high}}]$ (kế thừa từ `DSP::SpectralDeEsser`), cắt gọt các đỉnh chói gắt vượt `THRESHOLD` đã bù nghiêng `SPECTRAL_TILT`.
7. **Equalizer & Dynamic EQ (`DSP::Shared::EqualizerIIR` / `EqualizerFIR` / `DynamicEQ`)**:
   - `Equalizer` (`255` tham số) hỗ trợ 8 băng độc lập M/S hoặc L/R với 2 nhân: `EqualizerIIR` (Biquad bậc 2/4/8 kiểu Analog/Baxandall/Proportional-Q) và `EqualizerFIR` (bộ lọc FIR FFT overlap-save pha tuyến tính / pha hỗn hợp với tham số `Phase` và `Ripple`).
   - `DynamicEQ` (`181` tham số) hỗ trợ 6 băng Biquad có hệ số khuếch đại $G_b[n]$ biến thiên từng mẫu theo bộ dò đường bao đầu vào đã lọc qua bộ lọc sidechain tần số $f_{sc,b}, Q_{sc,b}$.
8. **Stem EQ (`DSP::OzoneStemEQ` `0x1857397B0`) & Master Rebalance (`DSP::OzoneMasterRebalance` `0x18569B9E8`)**:
   - **Master Rebalance**: Tham số `SourceType` (`0`=Vocals, `1`=Bass, `2`=Drums), `SourceGain` ($[-8, +8]\text{ dB}$), `Auto Gain Enable`. Áp dụng mặt nạ mềm $\hat{M}_s(f,t)$ từ mạng nơ-ron lên phổ STFT hỗn hợp $X(f,t)$:
     $$Y(f,t) = \left[1 + \left(10^{\text{SourceGain}/20} - 1\right)\hat{M}_s(f,t)\right] X(f,t)$$
     Công thức cộng tính này đảm bảo khi `SourceGain = 0 dB`, $Y(f,t) \equiv X(f,t)$ (bit-transparent, không có sai số tái tạo pha).
   - **Stem EQ**: Tách tín hiệu thành 4 nhánh (Vocals, Drums, Bass, Other = $X - \sum_{s=1}^3 \hat{S}_s$), áp dụng chuỗi EQ 8 băng (`EqualizerIIR`/`FIR`) riêng biệt lên từng nhánh $\Delta H_s(f)$, rồi cộng bù vào tín hiệu gốc:
     $$Y(f,t) = X(f,t) + \sum_{s \in \{\text{voc, drm, bas, oth}\}} \left(\Delta H_s(f) - 1\right) \hat{S}_s(f,t)$$
9. **Unlimiter (`DSP::OzoneUnlimiter` `0x18568D6E0`) & Bass Control (`DSP::OzoneBassControl` `0x18569B330`)**:
   - **Unlimiter**: Tham số `Amount` ($[0, 100]$), `Threshold` ($[-30, 0]\text{ dB}$), `InputGain`, `OutputGain`. Đưa bản thu đã bị nén/clip qua mô hình `models/delimit_exp009_ep080_fixed2.onnx` để dự đoán đường bao đỉnh nguyên bản $\hat{E}_{\text{unlim}}[n]$, sau đó điều chế mở rộng động lực các vùng vượt `Threshold` theo tỷ lệ `Amount`.
   - **Bass Control**: Tham số `Balance` ($[-100, +100]$ tỷ lệ Sustain/Punch), `CutoffFreq` ($[40, 250]\text{ Hz}$), `Punch`, `SustainPowerThreshold`, `SustainPowerAmount`, `SustainPowerBlend`, `PeakControlThreshold`, `PeakControlCharacter`. Phân tách phần sub-bass dưới `CutoffFreq` thành 2 luồng Sustain Power (nén lên + bão hòa hài bậc thấp) và Peak Control (điều khiển đỉnh transient trầm).

---

## 4. Giải Mã Toàn Bộ Mô Hình AI Trong iZotope Ozone 12 (`.npz` & `.onnx`)

### 4.1. Giải mã 27 mảng `.npy` trong `vocals-small_sanitized.npz`, `drums-small_sanitized.npz`, `bass-small_sanitized.npz`
Ba tệp trong [`requiredFiles_component/ssepnpz/`](file:///f:/FL%20studio/extracted_targets/ozone12/requiredFiles_component/ssepnpz) có kích thước tệp nén ZIP giống hệt nhau là **`26,261,516` bytes** (`25.04 MB`, tổng payload không nén = `26,259,464` B = `6,564,866` tham số `float32` + 5 hằng số vô hướng):
- [`vocals-small_sanitized.npz`](file:///f:/FL%20studio/extracted_targets/ozone12/requiredFiles_component/ssepnpz/vocals-small_sanitized.npz) — SHA-256: `c0fb92cc043866d98fb30479da78eb667db767879184805004fb230b8d7df17f`
- [`drums-small_sanitized.npz`](file:///f:/FL%20studio/extracted_targets/ozone12/requiredFiles_component/ssepnpz/drums-small_sanitized.npz) — SHA-256: `0b71522b44455d1c074a34f653789c78f24d4785a773cf798862796a8fa9407f`
- [`bass-small_sanitized.npz`](file:///f:/FL%20studio/extracted_targets/ozone12/requiredFiles_component/ssepnpz/bass-small_sanitized.npz) — SHA-256: `36b6412362a64c245997d6b910427f8e0ef5eb76d978a6f82094d58cdcc6671a`

Bằng cách dịch ngược hàm nạp `.npz` tại `0x181B55500..0x181B55B00` và lớp **`NeuralNet::GRU`** tại `0x1834213A0..0x18342202D` (trong đó `vfn` tại `0x183421780` trả về `mov eax, 10` — tức mỗi lớp GRU tiêu thụ đúng **10 mảng `.npy`**), cấu trúc chính xác của toàn bộ **27 tệp `.npy`** được xác lập như sau:

| Thứ tự | Tên tệp trong `.npz` | Kiểu dữ liệu | Kích thước (Shape) | Số phần tử | Dung lượng Payload | Ánh xạ trong `NeuralNet::GRU` (`0x183421550`) & Công thức toán học |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `FFT_SIZE.npy` | `int64` | `()` (vô hướng) | `1` | `8 B` | $N_{\text{fft}} = 2048 \implies F = N_{\text{fft}}/2 + 1 = 1025$ bins tần số ($\text{stereo} \implies 2050$) |
| 2 | `FFT_STRIDE.npy` | `int64` | `()` (vô hướng) | `1` | `8 B` | Bước nhảy STFT $H = 512$ mẫu ($75\%$ overlap, $11.61\text{ ms}$ tại $44.1\text{ kHz}$) |
| 3 | `SAMPLING_RATE.npy` | `int64` | `()` (vô hướng) | `1` | `8 B` | Tần số lấy mẫu chuẩn của mô hình $f_s = 44100\text{ Hz}$ |
| 4 | `STFT_MASK_EPSILON.npy` | `float32` | `()` (vô hướng) | `1` | `4 B` | Hằng số chống chia cho 0 khi tính mặt nạ tỷ số: $\epsilon_{\text{mask}} = 10^{-10}$ |
| 5 | `SIR_REGULARIZER.npy` | `float32` | `()` (vô hướng) | `1` | `4 B` | Hằng số điều hòa Signal-to-Interference Ratio: $\lambda_{\text{sir}} = 0.001$ |
| 6 | `arr_0.npy` | `float32` | `(2050, 512)` | `1,049,600` | `4,198,400 B` | **GRU 1 — Reset Gate (`[r14+0x20]`)**: Ma trận trọng số đầu vào $W_r^{(1)} \in \mathbb{R}^{2050\times 512}$ |
| 7 | `arr_1.npy` | `float32` | `(512, 512)` | `262,144` | `1,048,576 B` | **GRU 1 — Reset Gate**: Ma trận trọng số hồi quy $U_r^{(1)} \in \mathbb{R}^{512\times 512}$ |
| 8 | `arr_2.npy` | `float32` | `(512,)` | `512` | `2,048 B` | **GRU 1 — Reset Gate**: Vector độ lệch $b_r^{(1)} \in \mathbb{R}^{512}$ |
| 9 | `arr_3.npy` | `float32` | `(2050, 512)` | `1,049,600` | `4,198,400 B` | **GRU 1 — Update Gate (`[r14+0xA8]`)**: Ma trận trọng số đầu vào $W_z^{(1)} \in \mathbb{R}^{2050\times 512}$ |
| 10 | `arr_4.npy` | `float32` | `(512, 512)` | `262,144` | `1,048,576 B` | **GRU 1 — Update Gate**: Ma trận trọng số hồi quy $U_z^{(1)} \in \mathbb{R}^{512\times 512}$ |
| 11 | `arr_5.npy` | `float32` | `(512,)` | `512` | `2,048 B` | **GRU 1 — Update Gate**: Vector độ lệch $b_z^{(1)} \in \mathbb{R}^{512}$ (khởi tạo quanh $+1.0$) |
| 12 | `arr_6.npy` | `float32` | `(2050, 512)` | `1,049,600` | `4,198,400 B` | **GRU 1 — Candidate Gate (`[r14+0x130]`)**: Trọng số đầu vào $W_c^{(1)} \in \mathbb{R}^{2050\times 512}$ |
| 13 | `arr_7.npy` | `float32` | `(512, 512)` | `262,144` | `1,048,576 B` | **GRU 1 — Candidate Gate**: Ma trận trọng số hồi quy $U_c^{(1)} \in \mathbb{R}^{512\times 512}$ |
| 14 | `arr_8.npy` | `float32` | `(512,)` | `512` | `2,048 B` | **GRU 1 — Candidate Gate**: Vector độ lệch $b_c^{(1)} \in \mathbb{R}^{512}$ |
| 15 | `arr_9.npy` | `float32` | `(1, 512)` | `512` | `2,048 B` | **GRU 1 — Initial Hidden State (`[r14+0x1C0]`)**: Trạng thái ẩn ban đầu $h_0^{(1)} \in \mathbb{R}^{1\times 512}$ |
| 16 | `arr_10.npy` | `float32` | `(512, 512)` | `262,144` | `1,048,576 B` | **GRU 2 — Reset Gate**: Ma trận trọng số đầu vào $W_r^{(2)} \in \mathbb{R}^{512\times 512}$ |
| 17 | `arr_11.npy` | `float32` | `(512, 512)` | `262,144` | `1,048,576 B` | **GRU 2 — Reset Gate**: Ma trận trọng số hồi quy $U_r^{(2)} \in \mathbb{R}^{512\times 512}$ |
| 18 | `arr_12.npy` | `float32` | `(512,)` | `512` | `2,048 B` | **GRU 2 — Reset Gate**: Vector độ lệch $b_r^{(2)} \in \mathbb{R}^{512}$ |
| 19 | `arr_13.npy` | `float32` | `(512, 512)` | `262,144` | `1,048,576 B` | **GRU 2 — Update Gate**: Ma trận trọng số đầu vào $W_z^{(2)} \in \mathbb{R}^{512\times 512}$ |
| 20 | `arr_14.npy` | `float32` | `(512, 512)` | `262,144` | `1,048,576 B` | **GRU 2 — Update Gate**: Ma trận trọng số hồi quy $U_z^{(2)} \in \mathbb{R}^{512\times 512}$ |
| 21 | `arr_15.npy` | `float32` | `(512,)` | `512` | `2,048 B` | **GRU 2 — Update Gate**: Vector độ lệch $b_z^{(2)} \in \mathbb{R}^{512}$ (quanh $+1.0$) |
| 22 | `arr_16.npy` | `float32` | `(512, 512)` | `262,144` | `1,048,576 B` | **GRU 2 — Candidate Gate**: Ma trận trọng số đầu vào $W_c^{(2)} \in \mathbb{R}^{512\times 512}$ |
| 23 | `arr_17.npy` | `float32` | `(512, 512)` | `262,144` | `1,048,576 B` | **GRU 2 — Candidate Gate**: Ma trận trọng số hồi quy $U_c^{(2)} \in \mathbb{R}^{512\times 512}$ |
| 24 | `arr_18.npy` | `float32` | `(512,)` | `512` | `2,048 B` | **GRU 2 — Candidate Gate**: Vector độ lệch $b_c^{(2)} \in \mathbb{R}^{512}$ |
| 25 | `arr_19.npy` | `float32` | `(1, 512)` | `512` | `2,048 B` | **GRU 2 — Initial Hidden State**: Trạng thái ẩn ban đầu $h_0^{(2)} \in \mathbb{R}^{1\times 512}$ |
| 26 | `arr_20.npy` | `float32` | `(512, 2050)` | `1,049,600` | `4,198,400 B` | **Output `NeuralNet::Dense` (`0x18572E088`)**: Trọng số chiếu ra $W_{\text{out}} \in \mathbb{R}^{512\times 2050}$ |
| 27 | `arr_21.npy` | `float32` | `(2050,)` | `2,050` | `8,200 B` | **Output `NeuralNet::Dense`**: Độ lệch đầu ra $b_{\text{out}} \in \mathbb{R}^{2050}$ |

- **Phương trình suy luận chính xác từ mã máy `NeuralNet::GRU::forward` (`0x183421790..0x183421EF0`)**:
  Tại mỗi khung STFT $t$, đầu vào là vector biên độ phổ ghép kênh đôi $x_t = \big[|X_L(0..1024, t)|,\; |X_R(0..1024, t)|\big] \in \mathbb{R}^{1\times 2050}$. Với mỗi tầng GRU $l \in \{1, 2\}$ (trong đó $h_t^{(0)} = x_t$ và trạng thái khởi đầu khi `reset` tại `0x183421F10` được chép từ $h_0^{(l)}$):
  $$\begin{aligned}
  r_t^{(l)} &= \sigma\!\left(h_t^{(l-1)} W_r^{(l)} + h_{t-1}^{(l)} U_r^{(l)} + b_r^{(l)}\right) \\
  z_t^{(l)} &= \sigma\!\left(h_t^{(l-1)} W_z^{(l)} + h_{t-1}^{(l)} U_z^{(l)} + b_z^{(l)}\right) \\
  c_t^{(l)} &= \tanh\!\left(h_t^{(l-1)} W_c^{(l)} + \left(r_t^{(l)} \odot h_{t-1}^{(l)}\right) U_c^{(l)} + b_c^{(l)}\right) \quad \text{(quy ước Theano/Lasagne: } r_t \text{ nhân trước } U_c\text{)} \\
  h_t^{(l)} &= \left(1 - z_t^{(l)}\right) \odot c_t^{(l)} + z_t^{(l)} \odot h_{t-1}^{(l)}
  \end{aligned}$$
  Mặt nạ phổ mềm đầu ra $\hat{M}_t \in (0, 1)^{2050}$ được tính qua lớp Dense cuối cùng kết hợp điều hòa SIR:
  $$\hat{M}_t = \frac{\sigma\!\left(h_t^{(2)} W_{\text{out}} + b_{\text{out}}\right) + \lambda_{\text{sir}}}{1 + \lambda_{\text{sir}} + \epsilon_{\text{mask}}}$$

### 4.2. Giải mã 12 mô hình `.onnx` nhúng trong `IZRESEMBED` (Band-Split RNN & De-Limiter)
Bên cạnh 3 tệp `.npz` GRU thế hệ cũ, Ozone 12 nhúng **12 mô hình `.onnx`** (`opset = 15/17`, chạy qua `SourceSeparationTwo::OnnxInference` tại `0x18572F9D8`) sử dụng kiến trúc **BSRNN (Band-Split RNN — Luo & Yu, 2023)** cho tách Stem thời gian thực (`120 ms` lookahead) và Offline:

| # | Đường dẫn trong `IZRESEMBED` | File Offset (`iZOzone12Core.dll`) | Kích thước (Bytes) | Kiến trúc & Nhiệm vụ trong Ozone 12 |
| :- | :--- | :--- | :--- | :--- |
| 1 | `models/ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx` | `0x22139A44` | `7,230,143` (`6.90 MB`) | **Mặc định cho Stem EQ (Ozone 12)**: BSRNN 4-stem (Vocals, Drums, Bass, Other) độ trễ thấp `120 ms` |
| 2 | `models/mss_exp040_ep196.onnx` | `0x1ECF1CC4` | `7,230,143` (`6.90 MB`) | BSRNN 4-stem real-time (`120 ms`, bản thử nghiệm `exp040` cùng kiến trúc với mục 1) |
| 3 | `models/extra_small_vocals_separation_model_120ms_ep189.onnx` | `0x1DA77084` | `5,996,934` (`5.72 MB`) | BSRNN 1-stem Vocals siêu nhẹ (`120 ms`) cho Master Rebalance (Vocals) |
| 4 | `models/mss_exp042_voc_ep192.onnx` | `0x1F3D6F84` | `6,147,721` (`5.86 MB`) | BSRNN 1-stem Vocals (`ep192`) |
| 5 | `models/mss_exp043_drums_ep162.onnx` | `0x1F9B3E24` | `6,147,654` (`5.86 MB`) | BSRNN 1-stem Drums (`ep162`) cho Master Rebalance (Drums) |
| 6 | `models/mss_exp044_bass_ep191.onnx` | `0x1FF90C84` | `6,147,654` (`5.86 MB`) | BSRNN 1-stem Bass (`ep191`) cho Master Rebalance (Bass) |
| 7 | `models/mss_exp013_sweep6_ep016.onnx` | `0x1E02F224` | `6,408,756` (`6.11 MB`) | BSRNN đa stem thử nghiệm `exp013` |
| 8 | `models/mss_exp024_ep200.onnx` | `0x1E64BC64` | `6,971,483` (`6.65 MB`) | BSRNN đa stem thử nghiệm `exp024` |
| 9 | `models/mss_v02_3stem_ep034.onnx` | `0x2056DAE4` | `29,146,970` (`27.80 MB`) | BSRNN 3-stem độ chính xác trung bình |
| 10 | `izmlss/mss-offline_exp004_ep041.onnx` | `0x155C8E04` | `51,479,867` (`49.09 MB`) | BSRNN Offline chất lượng cao (`exp004`, 49 subbands, 6-layer BiLSTM) |
| 11 | `izmlss/mss-offline_exp011_ep044.onnx` | `0x186E1344` | `82,065,036` (`78.26 MB`) | BSRNN Offline chất lượng tối đa (`exp011`, 49 subbands, 6-layer BiLSTM) |
| 12 | `models/delimit_exp009_ep080_fixed2.onnx` | `0x1D5249E4` | `5,580,442` (`5.32 MB`) | Mạng BSRNN khôi phục đỉnh động lực bị nén/clip cho module **Ozone 12 Unlimiter** (`DSP::OzoneUnlimiter`) |

- **Đặc tả Tensor I/O của `models/ozone12_mss4stem_bsrnn_xs_120ms_ep147.onnx`**:
  - **Inputs**:
    - `input_spectrogram`: `float32[2, 1, 1025, 3, 2]` (2 kênh L/R theo chiều batch, $1025$ bin tần số ứng với $N_{\text{FFT}}=2048$, cửa sổ trượt $3$ khung thời gian $\approx 120\text{ ms}$, $2$ thành phần phức `[Real, Imag]`).
    - `input_states`: `float32[2, 2, 2, 1, 13, 128]` (trạng thái ẩn đệ quy $[h, c]$ của 2 tầng LSTM một chiều theo trục thời gian, chia phổ thành **$B = 13$ băng con (subbands)** với số chiều ẩn $H = 128$).
    - `input_states_normalize`: `float32[2, 1]` (trạng thái chuẩn hóa năng lượng trực tuyến).
  - **Outputs**:
    - `output_spectrogram`: `float32[2, 3, 1, 1025, 1, 2]` (phổ phức của 3 stem `[Vocals, Drums, Bass]`; stem thứ 4 `Other` được tính bằng `input - sum(output)`).
    - `output_states`: `float32[2, 2, 2, 1, 13, 128]`, `output_states_normalize`: `float32[2, 1]`.
- **Đặc tả Tensor I/O của `models/delimit_exp009_ep080_fixed2.onnx` (Unlimiter)**:
  - **Inputs**: `input_spectrogram` `float32[1, 1, 1025, 2, 2]`, `input_states` `float32[1, 2, 2, 1, 13, 128]`.
  - **Outputs**: `output_spectrogram` `float32[1, 1, 1, 1025, 1, 2]`, `output_states` `float32[1, 2, 2, 1, 13, 128]`.

### 4.3. Giải mã 2 mô hình `.npz` của Master Assistant / Referencing (`Referencing/DSP/*.npz`)
Bên trong `IZRESEMBED` còn nhúng 2 tệp `.npz` phục vụ phân tích tự động của **Master Assistant** và **Audiolens Referencing**:
1. **`Referencing/DSP/gparamsBN8.0.npz`** (`rel_off = 0x01BFE9E0`, `file_off = 0x084E4564`, **`6,522,880` bytes**, gồm **34 mảng `.npy`**):
   - Là mạng **CNN + BatchNorm phân loại thể loại nhạc (Genre Classifier)** hoạt động trên ảnh phổ Mel 80 băng chuẩn hóa theo `mu.npy (80,)` và `sigma.npy (80,)`.
   - Cấu trúc các tầng trọng số (`float32`):
     - **Khối Conv2D + BatchNorm 1..4**:
       - `conv1_w (32, 1, 7, 7)`, `conv1_b (32,)`, `bn1_{beta, gamma, mean, var} (32,)`
       - `conv2_w (64, 32, 5, 5)`, `conv2_b (64,)`, `bn2_{beta, gamma, mean, var} (64,)`
       - `conv3_w (128, 64, 3, 3)`, `conv3_b (128,)`, `bn3_{beta, gamma, mean, var} (128,)`
       - `conv4_w (256, 128, 3, 3)`, `conv4_b (256,)`, `bn4_{beta, gamma, mean, var} (256,)`
     - **Khối Fully-Connected 1..3**:
       - `dense1_w (2560, 512)`, `dense1_b (512,)`
       - `dense2_w (512, 512)`, `dense2_b (512,)`
       - `dense3_w (512, 10)`, `dense3_b (10,)` $\to$ Softmax ra xác suất 10 thể loại mục tiêu (`Target 1..10` trong `Stabilizer` & `Master Assistant`).
2. **`Referencing/DSP/instrument_classifier_params.npz`** (`rel_off = 0x022371E0`, `file_off = 0x08B1CD64`, **`19,853,709` bytes**, gồm **12 mảng `.npy`**):
   - Là mạng CNN nhận diện nhạc cụ / cấu trúc phổ trên 80 băng Mel (`mu.npy (80,)`, `sigma.npy (80,)`):
   - Gồm 5 tầng trọng số:
     - `layer0_W (16, 1, 1, 4)`, `layer0_b (16,)`
     - `layer1_W (32, 16, 3, 3)`, `layer1_b (32,)`
     - `layer2_W (64, 32, 3, 3)`, `layer2_b (64,)`
     - `layer3_W (4800, 1000)`, `layer3_b (1000,)`
     - `layer4_W (1000, 10)`, `layer4_b (10,)`.

---

## 5. Giải Mã Toàn Diện Ultimate Vocal Remover GUI v5.6 (`UVR5`)

Trong [`f:\FL studio\ultimatevocalremovergui`](file:///f:/FL%20studio/ultimatevocalremovergui), hệ thống tách nhạc AI điều phối từ [`UVR.py`](file:///f:/FL%20studio/ultimatevocalremovergui/UVR.py) (`7,265` dòng) và [`separate.py`](file:///f:/FL%20studio/ultimatevocalremovergui/separate.py) (`1,222` dòng) chia thành **3 kiến trúc mạng nơ-ron chính** cùng bộ hợp nhất phổ **Ensemble**:

### 5.1. Kiến trúc 1: MDX-Net (`ConvTDFNet`) & MDX23C (`TFC_TDF_net`)
Thực thi tại [`separate.py`](file:///f:/FL%20studio/ultimatevocalremovergui/separate.py#L221-L598) (`SeperateMDX`, `SeperateMDXC`), [`lib_v5/mdxnet.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/mdxnet.py#L1-L216) và [`lib_v5/tfc_tdf_v3.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/tfc_tdf_v3.py#L1-L157):
1. **Biến đổi STFT / ISTFT chuẩn hóa (`STFT` class tại [`lib_v5/tfc_tdf_v3.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/tfc_tdf_v3.py#L10-L47))**:
   - Với cấu hình $N_{\text{fft}} \in \{4096, 6144, 7680, 8192\}$, hop length $H = 1024$, cửa sổ Hann $w[n] = 0.5\big(1 - \cos(2\pi n / N_{\text{fft}})\big)$:
   - Phổ phức $X \in \mathbb{C}^{B \times 2 \times (N_{\text{fft}}/2+1) \times T}$ được **cắt bỏ bin Nyquist cuối cùng** (`x = x[:, :, :-1]`, giữ đúng $F = \text{dim\_f} \le N_{\text{fft}}/2$ bins chẵn luỹ thừa 2 để qua $L$ tầng Downsampling chia đôi không lẻ kích thước), rồi xếp chồng phần thực và phần ảo của 2 kênh Stereo thành tensor **4 kênh thực** (Complex-as-Channels — CaC):
     $$\mathbf{X}_{\text{in}} = \big[\text{Re}(X_L),\; \text{Im}(X_L),\; \text{Re}(X_R),\; \text{Im}(X_R)\big] \in \mathbb{R}^{B \times 4 \times F \times T}$$
   - Khi tái tạo (`__inverse__`), tensor đầu ra $\hat{\mathbf{Y}} \in \mathbb{R}^{B \times 4 \times F \times T}$ được đệm thêm 1 bin 0 ở tần số Nyquist (`F.pad(x, (0,0,0,1))`) rồi gọi `torch.istft`.
2. **Khối nền tảng TFC_TDF (`Time-Frequency Convolution + Time-Distributed Fully-Connected`)**:
   - Trong [`lib_v5/mdxnet.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/mdxnet.py#L22-L105) (`ConvTDFNet`, cấu hình từ [`model_name_mapper.json`](file:///f:/FL%20studio/ultimatevocalremovergui/models/MDX_Net_Models/model_data/model_name_mapper.json)) và [`lib_v5/tfc_tdf_v3.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/tfc_tdf_v3.py#L49-L115) (`TFC_TDF_net` cho MDX23C, cấu hình từ [`models/MDX_Net_Models/model_data/mdx_c_configs/*.yaml`](file:///f:/FL%20studio/ultimatevocalremovergui/models/MDX_Net_Models/model_data/mdx_c_configs/model_2_stem_full_band_8k.yaml) với `dim_f=4096, n_fft=8192, num_scales=5, num_subbands=4, num_tdf_layers=6`):
   - Mỗi khối ở tỷ lệ $s \in \{0..S-1\}$ với số bin tần số $F_s = F / 2^s$ thực hiện:
     $$\begin{aligned}
     \mathbf{H}_{\text{tfc}} &= \text{TFC}(\mathbf{Z}_{\text{in}}) \quad (\text{DenseNet gồm } k \text{ lớp }\texttt{Conv2d}_{3\times 3} + \texttt{BatchNorm/InstanceNorm} + \texttt{GELU/ReLU}) \\
     \mathbf{H}_{\text{tdf}} &= \mathbf{H}_{\text{tfc}} + W_2^{(s)}\,\sigma_{\text{act}}\!\left(\text{BN}\!\left(W_1^{(s)} \mathbf{H}_{\text{tfc}}\right)\right), \quad W_1^{(s)} \in \mathbb{R}^{(F_s / b_f) \times F_s},\; W_2^{(s)} \in \mathbb{R}^{F_s \times (F_s / b_f)} \\
     \mathbf{Z}_{\text{out}} &= \text{TFC}_{\text{post}}(\mathbf{H}_{\text{tdf}}) + \text{Conv}_{1\times 1}(\mathbf{Z}_{\text{in}})
     \end{aligned}$$
     Nhánh **TDF** là phép nhân ma trận dọc theo **toàn bộ trục tần số $F_s$** tại mỗi khung thời gian $t$, giúp mô hình nắm bắt tương quan hài âm toàn cục (fundamental + overtones) mà tích chập $3\times 3$ cục bộ không nhìn thấy được. Trong MDX23C còn có phép gấp băng con `cac2cws` / `cws2cac` chia trục tần số thành `num_subbands` dải xếp vào chiều kênh.
3. **Overlap-Add Windowing & Khử Nhiễu Đảo Dấu 2 Chiều (`demix` tại [`separate.py`](file:///f:/FL%20studio/ultimatevocalremovergui/separate.py#L406-L635))**:
   - Âm thanh được cắt thành các đoạn `chunk_size = hop * (mdx_segment_size - 1)` với `trim = n_fft // 2`, `gen_size = chunk_size - 2 * trim`, nhân cửa sổ Hanning `windowing_array` để triệt tiêu tiếng nổ biên (`boundary artifacts`) và nhân hệ số bù `compensate` (thường `1.035`).
   - Khi bật `is_denoise`, mô hình chạy suy luận 2 lần trên phổ dương và phổ đảo dấu:
     $$\hat{\mathbf{S}} = \frac{1}{2}\Big(\mathcal{M}(\mathbf{S}) - \mathcal{M}(-\mathbf{S})\Big)$$
     giúp triệt tiêu hoàn toàn thành phần bias chẵn và nhiễu nền tĩnh của mạng ONNX.

### 5.2. Kiến trúc 2: Demucs v4 — Hybrid Transformer Demucs (`HDemucs` & `HTDemucs`)
Thực thi tại [`separate.py`](file:///f:/FL%20studio/ultimatevocalremovergui/separate.py#L601-L820) (`SeperateDemucs`), [`demucs/hdemucs.py`](file:///f:/FL%20studio/ultimatevocalremovergui/demucs/hdemucs.py#L1-L725), [`demucs/htdemucs.py`](file:///f:/FL%20studio/ultimatevocalremovergui/demucs/htdemucs.py#L1-L692) và [`demucs/transformer.py`](file:///f:/FL%20studio/ultimatevocalremovergui/demucs/transformer.py#L1-L580):
1. **Kiến trúc U-Net Kép (Dual-Branch Time + Spectral U-Net)**:
   - **Nhánh Phổ (`z`, Spectral Encoder `encoders`)**: Tính STFT ($N_{\text{fft}} = 4096$, hop $H = 1024$) qua biểu diễn phức (`cac = True`, tức Complex-as-Channels: $2C$ kênh thực/ảo), đi qua 6 tầng `HEncLayer` (tích chập dọc trục tần số với stride `4` hoặc `2`, kèm nhánh `DConv` dilated residual + `GroupNorm` + `GLU` + `LayerScale`).
   - **Nhánh Thời Gian (`xt`, Time Encoder `tencoder`)**: Tín hiệu sóng âm thô $x[n] \in \mathbb{R}^{B \times 2 \times L}$ đi qua 6 tầng `HEncLayer` 1D (`kernel_size = 8, stride = 4`), sao cho tại tầng thứ 5, độ dài chuỗi thời gian của nhánh `xt` khớp chính xác với số khung STFT $T$ của nhánh phổ `z`.
2. **Cross-Domain Transformer Bottleneck (`CrossTransformerEncoder` tại [`demucs/transformer.py`](file:///f:/FL%20studio/ultimatevocalremovergui/demucs/transformer.py#L415-L550))**:
   - Tại cổ chai (`bottleneck`), tensor phổ 2D được trải phẳng cùng hướng với tensor thời gian 1D, cộng **2D Sinusoidal Positional Embedding** (`create_2d_sin_embedding`) và đưa qua $N_{\text{layers}} \in \{5, 6\}$ lớp xen kẽ giữa **Self-Attention** và **Cross-Attention** giữa miền thời gian $\mathbf{Q}_t, \mathbf{K}_f, \mathbf{V}_f$ và miền tần số $\mathbf{Q}_f, \mathbf{K}_t, \mathbf{V}_t$:
     $$\text{CrossAttn}(\mathbf{X}_f, \mathbf{X}_t) = \text{Softmax}\!\left(\frac{(\mathbf{X}_f W_Q)(\mathbf{X}_t W_K)^T}{\sqrt{d_k}}\right)(\mathbf{X}_t W_V)$$
3. **Tổng hợp Đa Nguồn (4-Stem / 6-Stem Output)**:
   - Đầu ra của nhánh giải mã phổ (qua `ispec` ISTFT) và nhánh giải mã thời gian (qua `tdecoder` ConvTranspose1d) được **cộng trực tiếp trên miền thời gian** cho từng stem $s \in \{\text{drums, bass, other, vocals, (guitar, piano)}\}$:
     $$\hat{y}_s[n] = \text{ISTFT}\!\left(\hat{Z}_s(f, t)\right)[n] + \hat{x}_{t, s}[n]$$

### 5.3. Kiến trúc 3: VR Architecture (`CascadedNet` & `CascadedASPPNet`)
Thực thi tại [`separate.py`](file:///f:/FL%20studio/ultimatevocalremovergui/separate.py#L823-L1140) (`SeperateVR`), [`lib_v5/vr_network/nets.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/vr_network/nets.py#L1-L215), [`lib_v5/vr_network/nets_new.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/vr_network/nets_new.py#L1-L185), [`lib_v5/vr_network/layers.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/vr_network/layers.py#L1-L260) và cấu hình JSON [`lib_v5/vr_network/modelparams/4band_v3.json`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/vr_network/modelparams/4band_v3.json):
1. **Ngân hàng STFT Đa Độ Phân Giải (`wave_to_spectrogram` & `combine_spectrograms` tại [`lib_v5/spec_utils.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/spec_utils.py#L96-L246))**:
   - Theo cấu hình `4band_v3.json`, tín hiệu đầu vào được lấy mẫu lại ở 4 tốc độ và kích thước FFT khác nhau (`band 1`: $f_s = 7350\text{ Hz}, N_{\text{fft}} = 640$; `band 2`: $f_s = 7350\text{ Hz}, N_{\text{fft}} = 320$; `band 3`: $f_s = 14700\text{ Hz}, N_{\text{fft}} = 512$; `band 4`: $f_s = 44100\text{ Hz}, N_{\text{fft}} = 960$). Các dải tần con từ 4 phổ STFT này được cắt ghép dọc trục tần số thành một ma trận biên độ hợp nhất $|X_{\text{comb}}| \in \mathbb{R}^{2 \times F_{\text{total}} \times T}$ (`672` bins).
2. **Mạng U-Net Phân Tầng Nối Tiếp (`CascadedASPPNet` tại [`lib_v5/vr_network/nets_new.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/vr_network/nets_new.py#L62-L168))**:
   - **Stage 1 (`stg1_low_band_net` + `stg1_high_band_net`)**: Chia đôi trục tần số thành nửa thấp $[0, F/2)$ và nửa cao $[F/2, F)$, chạy 2 mạng `BaseASPPNet` song song rồi ghép lại.
   - **Stage 2 (`stg2_bridge` + `stg2_full_band_net`)**: Nối (concatenate) đặc trưng đầu ra Stage 1 với phổ gốc $|X_{\text{comb}}|$, đưa qua U-Net toàn dải kèm khối **ASPP (`Atrous Spatial Pyramid Pooling`)** và `LSTMModule`.
   - **Stage 3 (`stg3_bridge` + `stg3_full_band_net`)**: Tinh chỉnh lần cuối và xuất mặt nạ $\hat{M}(c, f, t) = \sigma(\text{out}(c, f, t)) \in (0, 1)$.
3. **Điều chỉnh độ quyết liệt (`adjust_aggr` tại [`lib_v5/spec_utils.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/spec_utils.py#L383-L446)) & Post-Process (`merge_artifacts` tại [`L450-L475`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/spec_utils.py#L450-L475))**:
   - Cho phép đẩy mạnh hoặc giảm mặt nạ Vocal/Instrumental theo tham số `aggressiveness['value']` $\alpha \in [-0.3, +0.3]$ trên dải tần `aggr_threshold`.

### 5.4. Thuật Toán Hợp Nhất Đa Mô Hình (Ensemble) & Khử Pha Trong [`lib_v5/spec_utils.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/spec_utils.py#L502-L1161)
1. **Spectral Ensemble (`ensembling` tại [`lib_v5/spec_utils.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/spec_utils.py#L526-L549) & `ensemble_inputs` tại [`L564-L590`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/spec_utils.py#L564-L590))**:
   Với $K$ sóng âm đầu ra từ $K$ mô hình $\{x_1[n], \dots, x_K[n]\}$ và trọng số người dùng $w_i$:
   - **Chế độ `AVERAGE` (`average_audio` tại [`L751-L774`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/spec_utils.py#L751-L774))**: Trung bình có trọng số **trực tiếp trên miền thời gian**:
     $$y_{\text{avg}}[c, n] = \frac{1}{K}\sum_{i=1}^{K} w_i\,x_i[c, n]$$
   - **Chế độ `MAX_SPEC` (`max_fft`), `MIN_SPEC` (`min_fft`)**: Biến đổi từng sóng âm đã nhân trọng số sang miền STFT phức $X_i(c, f, t) \in \mathbb{C}$ ($N_{\text{fft}} = 2048$). Tại mỗi phần tử $(c, f, t)$, thuật toán (`np.where(np.abs(inputs[i]) <= np.abs(input), inputs[i], input)`) chọn **toàn bộ số phức STFT** (cả biên độ lẫn pha) của mô hình đạt cực đại/cực tiểu biên độ:
     $$\begin{aligned}
     Y_{\text{min\_spec}}(c, f, t) &= X_{i^*_{\min}}(c, f, t) \quad \text{với } i^*_{\min} = \arg\min_{i \in \{1..K\}} \big|X_i(c, f, t)\big| \\
     Y_{\text{max\_spec}}(c, f, t) &= X_{i^*_{\max}}(c, f, t) \quad \text{với } i^*_{\max} = \arg\max_{i \in \{1..K\}} \big|X_i(c, f, t)\big|
     \end{aligned}$$
     *(Quy tắc vàng trong UVR5: Khi gộp nhiều mô hình cho **Inst**, dùng `MAX_SPEC` để giữ trọn nhạc cụ; khi gộp cho **Vocals**, dùng `MIN_SPEC` để triệt sạch nhạc nền lọt sang; sau đó lấy phần bù bằng `invert_audio`).*
2. **Trừ Phổ Khử Pha (`invert_audio` tại [`lib_v5/spec_utils.py`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/spec_utils.py#L502-L517)) & Căn Chỉnh Trễ Liên Tương Quan (`time_correction` tại [`L1119-L1161`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/spec_utils.py#L1119-L1161))**:
   - Sau khi căn chỉnh độ lệch mẫu sub-sample bằng tương quan chéo FFT + biến đổi Hilbert (`phase_shift_hilbert` tại [`L1035-L1060`](file:///f:/FL%20studio/ultimatevocalremovergui/lib_v5/spec_utils.py#L1035-L1060)), hàm `invert_audio(specs, invert_p=True)` tách stem phụ từ bản gốc $X_0$ và stem chính $X_1$ bằng cách ép biên độ cực đại theo góc pha của $X_0$:
     $$Y_{\text{inv}}(c, f, t) = X_1(c, f, t) - \max\!\big(|X_0(c, f, t)|,\; |X_1(c, f, t)|\big)\cdot e^{j \angle X_0(c, f, t)}$$

---

## 6. Thiết Kế Kiến Trúc Rust Tương Đương (`Ozone 12 Mastering Suite` & `UVR5 AI Engine`)

```rust
//! Kiến trúc Pure-Rust + ONNX Runtime cho Ozone 12 Mastering Suite & UVR5 Stem Separation Engine

use std::sync::Arc;

/// Trait cơ sở tương đương `DSP::ElementImpl` (`VT = 0x1856DF158`) & `DSP::OzoneChainElement` (`VT = 0x18572E258`)
pub trait OzoneDspModule: Send + Sync {
    fn reset(&mut self, sample_rate: f64, max_block_size: usize); // vfn[58] / vfn[61]
    fn process_block(&mut self, left: &mut [f32], right: &mut [f32]); // vfn[59] / vfn[62]
    fn latency_samples(&self) -> usize { 0 }
}

/// 1. Bảng tra cứu & Bộ giải DK-Method cho Exciter Triode / DualTriode (`KState<3, 2, 2, KorenFunction>`)
/// Khớp chính xác byte-by-byte với `exciter-triode-48000.000000.dat` (2,074,532 bytes)
#[repr(C)]
#[derive(Clone)]
pub struct DkTriodeLut {
    pub magic: u32,               // 0x0000: 0x8BAD43FC
    pub version: u32,             // 0x0004: 4
    pub dc_offset: f64,           // 0x0008: -0.04588765735184097
    pub circuit_mode: u32,        // 0x0010: 0 = Triode, 1 = DualTriode
    pub x_dc: [f64; 3],           // 0x001C: [0.0, 196.282586, 1.555761]
    pub u_dc: [f64; 2],           // 0x003C: [0.0, 300.0]
    pub i_nl_dc: [f64; 2],        // 0x0054: [0.0, 0.001037174]
    pub sample_rate: f64,         // 0x0064: 48000.0
    pub d_d: [[f64; 3]; 2],       // 0x020C: Ma trận chiếu trạng thái rời rạc D_d (2x3)
    pub e_d: [[f64; 2]; 2],       // 0x0244: Ma trận đầu vào hiện tại E_d (2x2)
    pub f_d: [[f64; 2]; 2],       // 0x026C: Ma trận đầu vào trễ F_d (2x2)
    pub g_d: [[f64; 2]; 2],       // 0x0294: Ma trận dòng phi tuyến trễ G_d (2x2)
    pub a_d: [[f64; 3]; 3],       // 0x02BC: Ma trận chuyển trạng thái A_d (3x3)
    pub b_d: [[f64; 2]; 3],       // 0x030C: Ma trận cập nhật đầu vào B_d (3x2)
    pub c_d: [[f64; 2]; 3],       // 0x0344: Ma trận cập nhật dòng phi tuyến C_d (3x2)
    pub bounds: [f64; 4],         // 0x0374: [-600.0, 300.0, 240.0, 303.0] (p1_min, p2_min, p1_max, p2_max)
    pub grid_dims: [u32; 2],      // 0x0394: [360, 360]
    pub lut: Arc<[[[f64; 2]; 360]; 360]>, // 0x03A4..0x1FA7A4: 129,600 nút x [I_g, I_p]
}

pub struct DkTriodeChannelState {
    pub x: [f64; 3],
    pub u_prev: [f64; 2],
    pub i_nl_prev: [f64; 2],
    pub lut: Arc<DkTriodeLut>,
}

impl DkTriodeChannelState {
    #[inline(always)]
    pub fn tick(&mut self, v_in: f64) -> f64 {
        let l = &*self.lut;
        let u = [v_in, l.u_dc[1]]; // [V_in[n], 300.0V]
        // 1. Tính p[n] = D_d * x[n-1] + E_d * u[n] + F_d * u[n-1] + G_d * i_nl[n-1]
        let mut p = [0.0f64; 2];
        for r in 0..2 {
            p[r] = l.d_d[r][0] * self.x[0] + l.d_d[r][1] * self.x[1] + l.d_d[r][2] * self.x[2]
                 + l.e_d[r][0] * u[0]      + l.e_d[r][1] * u[1]
                 + l.f_d[r][0] * self.u_prev[0] + l.f_d[r][1] * self.u_prev[1]
                 + l.g_d[r][0] * self.i_nl_prev[0] + l.g_d[r][1] * self.i_nl_prev[1];
        }
        // 2. Nội suy song tuyến tính trên lưới 360x360
        let u_norm = ((p[0] - l.bounds[0]) / (l.bounds[2] - l.bounds[0])).clamp(0.0, 1.0) * 359.0;
        let v_norm = ((p[1] - l.bounds[1]) / (l.bounds[3] - l.bounds[1])).clamp(0.0, 1.0) * 359.0;
        let (i0, j0) = ((u_norm as usize).min(358), (v_norm as usize).min(358));
        let (fu, fv) = (u_norm - i0 as f64, v_norm - j0 as f64);
        let mut i_nl = [0.0f64; 2];
        for k in 0..2 {
            let c00 = l.lut[i0][j0][k];
            let c10 = l.lut[i0 + 1][j0][k];
            let c01 = l.lut[i0][j0 + 1][k];
            let c11 = l.lut[i0 + 1][j0 + 1][k];
            i_nl[k] = (1.0 - fu) * ((1.0 - fv) * c00 + fv * c01)
                    + fu * ((1.0 - fv) * c10 + fv * c11);
        }
        // 3. Cập nhật x[n] = A_d * x[n-1] + B_d * (u[n] + u[n-1]) + C_d * (i_nl[n] + i_nl[n-1])
        let u_sum = [u[0] + self.u_prev[0], u[1] + self.u_prev[1]];
        let i_sum = [i_nl[0] + self.i_nl_prev[0], i_nl[1] + self.i_nl_prev[1]];
        let mut x_next = [0.0f64; 3];
        for r in 0..3 {
            let val = l.a_d[r][0] * self.x[0] + l.a_d[r][1] * self.x[1] + l.a_d[r][2] * self.x[2]
                    + l.b_d[r][0] * u_sum[0]  + l.b_d[r][1] * u_sum[1]
                    + l.c_d[r][0] * i_sum[0]  + l.c_d[r][1] * i_sum[1];
            x_next[r] = if val.abs() < 1e-15 { 0.0 } else { val };
        }
        self.x = x_next;
        self.u_prev = u;
        self.i_nl_prev = i_nl;
        (self.x[1] - l.x_dc[1]) - l.dc_offset
    }
}

/// 2. Mô hình GRU Tách Stem `.npz` (`NeuralNet::GRU` tại `0x1834213A0` — 27 mảng `.npy`)
pub struct GruGateWeights {
    pub w_x: Vec<f32>, // [in_dim * 512]
    pub u_h: Vec<f32>, // [512 * 512]
    pub b: [f32; 512], // [512]
}

pub struct GruLayer512 {
    pub in_dim: usize,         // 2050 cho GRU 1, 512 cho GRU 2
    pub reset: GruGateWeights, // arr_0..2 (hoặc arr_10..12)
    pub update: GruGateWeights,// arr_3..5 (hoặc arr_13..15)
    pub cand: GruGateWeights,  // arr_6..8 (hoặc arr_16..18)
    pub h0: [f32; 512],        // arr_9 (hoặc arr_19)
    pub h_state: [f32; 512],
}

pub struct OzoneNpzStemModel {
    pub fft_size: usize,       // 2048
    pub fft_stride: usize,     // 512
    pub sample_rate: u32,      // 44100
    pub mask_epsilon: f32,     // 1e-10
    pub sir_regularizer: f32,  // 0.001
    pub gru1: GruLayer512,     // 2050 -> 512 (arr_0..9)
    pub gru2: GruLayer512,     // 512  -> 512 (arr_10..19)
    pub dense_w: Vec<f32>,     // [512 * 2050] (arr_20)
    pub dense_b: Vec<f32>,     // [2050]       (arr_21)
}

/// 3. UVR5 Multi-Architecture Stem Separation & Ensemble Engine
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum UvrEnsembleMode {
    AverageWaveform, // `average_audio` (miền thời gian)
    MaxSpec,         // `max_fft`: giữ nguyên số phức STFT có |X_i| lớn nhất
    MinSpec,         // `min_fft`: giữ nguyên số phức STFT có |X_i| nhỏ nhất
    MedianSpec,      // `median_fft`
}

#[derive(Clone, Copy)]
pub struct Complex32 { pub re: f32, pub im: f32 }
impl Complex32 {
    #[inline(always)]
    pub fn abs_sq(&self) -> f32 { self.re * self.re + self.im * self.im }
}

/// Khớp chính xác với `ensembling()` tại `lib_v5/spec_utils.py:L526-L549`
pub fn uvr5_spectral_ensemble(specs: &[Vec<Complex32>], mode: UvrEnsembleMode) -> Vec<Complex32> {
    let n_bins = specs[0].len();
    let mut out = specs[0].clone();
    for b in 0..n_bins {
        let mut best = out[b];
        for m in 1..specs.len() {
            let cand = specs[m][b];
            let take = match mode {
                UvrEnsembleMode::MaxSpec => cand.abs_sq() >= best.abs_sq(),
                UvrEnsembleMode::MinSpec => cand.abs_sq() <= best.abs_sq(),
                _ => false,
            };
            if take { best = cand; }
        }
        out[b] = best;
    }
    out
}
```
