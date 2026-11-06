# Bước 2: Deep DSP & Parameter Reverse Engineering — FabFilter Pro-Q 4 (`v4.13`) & oeksound Soothe3 (`v1.0.5`)

> **Clean-Room Engineering Specification**  
> **Ngày thực hiện:** 2026-11-05  
> Tài liệu này tổng hợp kết quả phân tích hệ thống tham số, cấu trúc C++ RTTI/vtable, hằng số toán học DSP, pipeline STFT/Spectral và đồ họa GPU từ:
> - **FabFilter Pro-Q 4 (`v4.13`)**: [`FabFilter Pro-Q 4.clap`](file:///f:/FL%20studio/extracted_targets/ffproq4/FabFilter%20Pro-Q%204.clap), [`FabFilter Pro-Q 4.dll`](file:///f:/FL%20studio/extracted_targets/ffproq4/FabFilter%20Pro-Q%204.dll), [`DigiPageTable.xml`](file:///f:/FL%20studio/extracted_targets/ffproq4/DigiPageTable.xml), [`FabFilter Pro-Q 4.chm`](file:///f:/FL%20studio/extracted_targets/ffproq4/FabFilter%20Pro-Q%204.chm).
> - **oeksound Soothe3 (`v1.0.5`)**: [`soothe3_symbols.map`](file:///f:/FL%20studio/extracted_targets/soothe3/soothe3_symbols.map) (26.18 MB, 142,863 dòng MSVC linker map), [`soothe3.vst3`](file:///f:/FL%20studio/extracted_targets/soothe3/soothe3.vst3), [`DigiPageTable.xml`](file:///f:/FL%20studio/extracted_targets/soothe3/DigiPageTable.xml), [`soothe3_manual.pdf`](file:///f:/FL%20studio/extracted_targets/soothe3/soothe3_manual.pdf), [`preset_cache.json`](file:///f:/FL%20studio/extracted_targets/soothe3/preset_cache.json) và 68 file JSON `.preset` trong [`Presets/`](file:///f:/FL%20studio/extracted_targets/soothe3/Presets).

---

## PHẦN 1: FABFILTER PRO-Q 4 (`v4.13`) — DEEP DSP & PARAMETER REVERSE ENGINEERING

### 1.1. Hệ thống 601 Tham số (Parameter System) & Cấu trúc 24-Band EQ

Từ [`DigiPageTable.xml`](file:///f:/FL%20studio/extracted_targets/ffproq4/DigiPageTable.xml) (14,186 dòng) và binary [`FabFilter Pro-Q 4.clap`](file:///f:/FL%20studio/extracted_targets/ffproq4/FabFilter%20Pro-Q%204.clap), Pro-Q 4 đăng ký chính xác **601 tham số** (`333` tham số rời rạc + `268` tham số liên tục):
- **576 Tham số Band**: `24 Bands × 24 Parameters/Band` (`Band 1` .. `Band 24`, ID từ `1` đến `576`).
- **25 Tham số Global**: ID từ `577` đến `601`.

#### A. Bảng 24 Tham số trên mỗi Band (`Band k`, $k \in \{1..24\}$, Base Index $= 24(k-1)$)

| Offset | Tên tham số (`DigiPageTable.xml`) | Kiểu / Bậc rời rạc | Dải giá trị UI & Công thức ánh xạ nội bộ (`0x18010F7E0..0x180110500`) |
|---|---|---|---|
| `+1` | `Band k Used` | Bool (`2` steps) | `0 = Unused`, `1 = Used` |
| `+2` | `Band k Enabled` | Bool (`2` steps) | `0 = Disabled`, `1 = Enabled` |
| `+3` | `Band k Frequency` | Continuous (`float`) | $10.0\text{ Hz} \dots 30,000.0\text{ Hz}$. Lưu nội bộ theo **Octave ($\log_2 f_{\text{Hz}}$)**: $f_{\text{oct}} \in [3.321928095, 14.87267488]$. Công thức: $f_{\text{oct}} = 3.321928095 + 11.550746785 \cdot x_{\text{norm}}$, $f_{\text{Hz}} = 2^{f_{\text{oct}}}$. |
| `+4` | `Band k Gain` | Continuous (`float`) | $-30.0\text{ dB} \dots +30.0\text{ dB}$ (tuyến tính theo dB: $G_{\text{dB}} = 60.0 \cdot x_{\text{norm}} - 30.0$). |
| `+5` | `Band k Q` | Continuous (`float`) | $0.025 \dots 40.0$ (mặc định $x_{\text{norm}} = 0.5 \Rightarrow Q_{\text{UI}} = 1.0$). Ánh xạ logarit: $Q_{\text{UI}} = 40^{2 x_{\text{norm}} - 1}$. Nội bộ nhân $\frac{1}{\sqrt{2}} = 0.70710678$: $Q_{\text{internal}} = 0.70710678 \cdot Q_{\text{UI}}$ (với Shelf/Tilt dùng $Q_{\text{shelf}} \in [0.125, 4.0]$). |
| `+6` | `Band k Shape` | Discrete (`10` UI steps) | UI: `0=Bell`, `1=Low Shelf`, `2=High Shelf`, `3=Low Cut`, `4=High Cut`, `5=Notch`, `6=Band Pass`, `7=Tilt Shelf`, `8=Flat Tilt`, `9=All Pass`. (Internal enum `0..12` trong `AnalogFilterPrototype`). |
| `+7` | `Band k Slope` | Discrete (`10` steps) / Fractional | `0=6 dB/oct`, `1=12 dB/oct`, `2=18 dB/oct`, `3=24 dB/oct`, `4=30 dB/oct`, `5=36 dB/oct`, `6=48 dB/oct`, `7=72 dB/oct`, `8=96 dB/oct`, `9=Brickwall` (chỉ cho Low/High Cut). Hỗ trợ **Fractional Slope** liên tục (`0.0..96.0 dB/oct`). |
| `+8` | `Band k Stereo Placement` | Discrete (`5` steps) | `0=Stereo`, `1=Left`, `2=Right`, `3=Mid`, `4=Side` (tự động đổi nhãn theo `Channel Mode`). |
| `+9` | `Band k Speakers` | Discrete (`65` steps) | Bitmask định tuyến kênh Surround / Dolby Atmos (L, R, C, LFE, Ls, Rs, Ltf, Rtf...). |
| `+10` | `Band k Dynamics Enabled` | Bool (`2` steps) | Bật/tắt xử lý Dynamic EQ / Spectral trên band. |
| `+11` | `Band k Dynamic Range` | Continuous (`float`) | $-30.0\text{ dB} \dots +30.0\text{ dB}$ ($<0$: Downward Compression; $>0$: Upward Expansion). |
| `+12` | `Band k Dynamics Auto` | Bool (`2` steps) | `1 = Auto Threshold & Auto Attack/Release`, `0 = Manual Attack/Release`. |
| `+13` | `Band k Threshold` | Continuous (`float`) | $-90.0\text{ dB} \dots 0.0\text{ dB}$ (hoặc Auto Gain Sensitivity khi `Dynamics Auto = 1`). |
| `+14` | `Band k Attack` | Continuous (`float`) | $0\% \dots 100\%$ scaling hệ số thời gian Attack (chỉ mở khi `Dynamics Auto = 0`). |
| `+15` | `Band k Release` | Continuous (`float`) | $0\% \dots 100\%$ scaling hệ số thời gian Release (chỉ mở khi `Dynamics Auto = 0`). |
| `+16` | `Band k External Side Chain` | Bool (`2` steps) | `0 = Internal`, `1 = External Side Chain` trigger. |
| `+17` | `Band k Side Chain Filtering` | Bool (`2` steps) | Bật/tắt bộ lọc bandpass sidechain riêng cho band. |
| `+18` | `Band k Side Chain Low Frequency` | Continuous (`float`) | Tần số cắt dưới của bộ lọc Sidechain ($10\text{ Hz} \dots 30\text{ kHz}$). |
| `+19` | `Band k Side Chain High Frequency` | Continuous (`float`) | Tần số cắt trên của bộ lọc Sidechain ($10\text{ Hz} \dots 30\text{ kHz}$). |
| `+20` | `Band k Side Chain Audition` | Bool (`2` steps) | Nghe thử tín hiệu Sidechain đã lọc. |
| `+21` | `Band k Spectral Enabled` | Bool (`2` steps) | Chuyển band từ Wideband Dynamic EQ sang **Spectral Dynamics** (chỉ cho Bell, Low/High Shelf, Tilt Shelf). |
| `+22` | `Band k Spectral Density` | Continuous (`float`) | $0\% \dots 100\%$ (Focal/Selectivity của xử lý Spectral Dynamics). |
| `+23` | `Band k Solo` | Bool (`2` steps) | Cô lập nghe thử dải tần của band. |
| `+24` | `Band k Invert Gain` | Bool (`2` steps) | Đảo dấu `Gain` và `Dynamic Range` ($\times -1$). |

#### B. Bảng 25 Tham số Global (`ID 577 .. 601`)

| ID | Tên tham số | Kiểu / Steps | Ý nghĩa & Giải thuật liên kết |
|---|---|---|---|
| `577` | `Processing Mode` | Discrete (`3`) | `0 = Zero Latency`, `1 = Natural Phase`, `2 = Linear Phase`. |
| `578` | `Processing Resolution` | Discrete (`5`) | Độ phân giải FIR cho Linear Phase: `0 = Low`, `1 = Medium` (default), `2 = High`, `3 = Very High`, `4 = Maximum`. |
| `579` | `Character` | Discrete (`3`) | `0 = Clean` (default), `1 = Subtle`, `2 = Warm` (Analog saturation trên từng band & output). |
| `580` | `Gain Scale` | Continuous | $0\% \dots 200\%$ (mặc định $100\%$). Nhân tỷ lệ lên toàn bộ `Band k Gain` và `Band k Dynamic Range`. |
| `581` | `Output Level` | Continuous | $-\infty\text{ dB} \dots +36.0\text{ dB}$ (mặc định $0.0\text{ dB}$). |
| `582` | `Output Pan` | Continuous | $-100\%\text{ (L)} \dots +100\%\text{ (R)}$. |
| `583` | `Output Pan Mode` | Discrete (`2`) | `0 = Left/Right`, `1 = Mid/Side`. |
| `584` | `Bypass` | Bool (`2`) | Soft-bypass toàn plugin (có crossfade chống click). |
| `585` | `Output Invert Phase` | Bool (`2`) | Đảo cực tính (polarity $\times -1$) tín hiệu đầu ra. |
| `586` | `Auto Gain` | Bool (`2`) | Tự động bù âm lượng cảm thụ theo 512-bin perceptual weighting LUT (`0x18019BD20`). |
| `587` | `Analyzer Show Pre-Processing` | Bool (`2`) | Hiển thị phổ FFT trước EQ. |
| `588` | `Analyzer Show Post-Processing` | Bool (`2`) | Hiển thị phổ FFT sau EQ. |
| `589` | `Analyzer Show Side Chain` | Bool (`2`) | Hiển thị phổ FFT của External Sidechain. |
| `590` | `Analyzer Range` | Discrete (`3`) | `0 = 60 dB`, `1 = 90 dB`, `2 = 120 dB`. |
| `591` | `Analyzer Resolution` | Discrete (`4`) | `0 = Low`, `1 = Medium`, `2 = High`, `3 = Maximum`. |
| `592` | `Analyzer Speed` | Discrete (`4`) | `0 = Very Slow`, `1 = Slow`, `2 = Medium`, `3 = Fast`. |
| `593` | `Analyzer Tilt` | Discrete (`5`) | Độ nghiêng phổ hiển thị: `0 = 0.0`, `1 = 1.5`, `2 = 3.0`, `3 = 4.5` (default), `4 = 6.0 dB/oct`. |
| `594` | `Analyzer Freeze` | Bool (`2`) | Đóng băng đường cong phổ cực đại. |
| `595` | `Spectrum Grab` | Bool (`2`) | Tự động bắt đỉnh cộng hưởng trên phổ để kéo thành EQ band. |
| `596` | `Display Range` | Discrete (`5`) | Thang chia lưới EQ: `0 = 3 dB`, `1 = 6 dB`, `2 = 12 dB`, `3 = 30 dB`, `4 = 60 dB`. |
| `597` | `Receive Midi` | Bool (`2`) | Nhận MIDI Note để điều khiển tần số band ($f = 440 \cdot 2^{(n-69)/12}$). |
| `598` | `Solo Gain` | Continuous | $-40.0\text{ dB} \dots +20.0\text{ dB}$ mức nghe thử khi giữ nút Solo. |
| `599` | `Channel Mode` | Discrete (`2`) | `0 = Left/Right`, `1 = Mid/Side`. |
| `600` | `Drift` | Continuous | $0.0\% \dots 100.0\%$ — biến thiên ngẫu nhiên chậm (Subtle random walk) cho Frequency/Q/Gain giữa các kênh để mô phỏng linh kiện analog. |
| `601` | `Show Collisions` | Bool (`2`) | Phát hiện va chạm phổ (Spectral Collision Detection) giữa các instance qua `SharedInstanceState`. |

---

### 1.2. Kiến trúc C++ RTTI, VTables & Bố cục Bộ nhớ (`FabFilter Pro-Q 4.clap`)

Quét toàn bộ bảng RTTI `TypeDescriptor` $\to$ `CompleteObjectLocator` $\to$ `VTable` trong [`FabFilter Pro-Q 4.clap`](file:///f:/FL%20studio/extracted_targets/ffproq4/FabFilter%20Pro-Q%204.clap) (ImageBase `0x180000000`) xác định chính xác sơ đồ class DSP và GPU:

| Tên Class C++ (RTTI Mangled Name) | Địa chỉ VTable | Hàm khởi tạo / Xử lý chính | Kích thước & Vai trò trong Engine |
|---|---|---|---|
| `ProQProcessor` (`.?AVProQProcessor@@`) | `0x1802324A8` | Ctor: `0x18019B8F0`<br>Update: `0x18019CBE0`<br>AutoGain: `0x18019BD20`<br>BandUpdate: `0x18019E300`<br>ConvolverInit: `0x18019EAE0` | Object size `0x23F0` (`9,200 bytes`). Điều phối toàn bộ 24 band, 16 kênh âm thanh (`0x180210F20`), Zero/Natural/Linear Phase, Spectral Dynamics, Character, Auto Gain, EQ Match. |
| `EQFilter` (`.?AVEQFilter@@`) | `0x180221958` | Update: `0x18010F7E0`<br>Gain-Q: `0x18010FF70`<br>FracSlope: `0x180112260` | Object size `0x1268` (`4,712 bytes`). Bộ lọc IIR đa tầng (Cascaded Biquad / SVF) hỗ trợ 13 kiểu lọc nội bộ (`0..12`) và Fractional Slope crossfading. |
| `NaturalPhaseFrequencyResponseSource` (`.?AVNaturalPhaseFrequencyResponseSource@@`) | `0x180221838` | Eval: `0x180112880`<br>Poly: `0x180112900` | Tính đáp tuyến phức (biên độ + pha) khớp mẫu bộ lọc tương tự (`0x180111450..0x180111687`). |
| `DynamicEQFilter` (`.?AVDynamicEQFilter@@`) | `0x180221A20` | Process: `0x1801134A0` | Object size `0x1420` (`5,152 bytes`). Bộ lọc IIR có hệ số gain điều chế theo từng mẫu/khối bởi tín hiệu từ `Detector` + `Capacitor`. |
| `Detector` (`.?AVDetector@@`) | `0x180221778` | Ctor: `0x180109520`<br>Process: `0x1801095A0` | Object size `0x6B0` (`1,712 bytes`). Đo mức năng lượng sidechain, tính hàm truyền tĩnh (Threshold, Dynamic Range, Auto Threshold, Soft Knee bậc 2) bằng đa thức `exp2` SIMD. |
| `Capacitor` (`.?AVCapacitor@@`) | `0x180221710` | Ctor: `0x18010A910`<br>Process: `0x18010A9B0`<br>Ballistics: `0x18010B2C0` | Object size `0x7D8` (`2,008 bytes`). Mạch RC phi tuyến bậc cao (Double-precision Program-Dependent Attack/Release Envelope Follower). |
| `WaveShaper` / `FunctionTableClipper` | `0x180221BF8` / `0x1802217F8` | Setup: `0x18019DA1F` | Xử lý méo hài phi tuyến cho `Character` (`Subtle` và `Warm`) bằng bảng nội suy hàm truyền bão hòa. |
| `AnalyzerProcessor` (`.?AVAnalyzerProcessor@@`) | `0x1802216B0` | Process: `0x180107800` | Phân tích phổ FFT thời gian thực (Pre, Post, Sidechain, Collision spectrum, Spectrum Grab). |
| `SharedInstanceState` (`.?AVSharedInstanceState@@`) | `0x180230738` | Sync: `0x18018B200` | Bộ nhớ chia sẻ giữa mọi instance Pro-Q 4 trong DAW phục vụ Instance List, EQ Sketch và Collision Detection. |
| `gui::OpenGLRenderer` (`.?AVOpenGLRenderer@gui@@`) | `0x180216520` | Init/Draw: `0x180055A90` | Engine dựng hình OpenGL 2D có khử răng cưa hình học (analytical edge feathering) và VBO batching. |

---

### 1.3. Giải mã Chi tiết các Thuật toán DSP Cốt lõi của Pro-Q 4

#### A. Zero Latency Mode (`EQFilter` — `0x18010F7E0`, `0x18010FF70`, `0x180112260`)
1. **13 Kiểu Bộ lọc Nội bộ (`AnalogFilterPrototype`, `[rcx+0x880]`)**:
   - Trong `EQFilter::updateCoefficients` (`0x18010F7E0`), Pro-Q 4 kiểm tra kiểu lọc thông qua hằng số bitmask **`0x10F6`** (`0b0001_0000_1111_0110`, tức các bit `1, 2, 4, 5, 6, 7, 12` tương ứng với các bộ lọc phụ thuộc Gain: Bell, Low Shelf, High Shelf, Tilt Shelf, Flat Tilt). Các bit còn lại (`0, 3, 8, 9, 10, 11`) là các bộ lọc không phụ thuộc Gain (Low Cut, High Cut, Notch, Band Pass, All Pass).
2. **Công thức Tần số, Gain và Gain-Q Interaction (`0x18010FF70`)**:
   - Tần số góc chuẩn hóa được tính ở độ chính xác kép (`double`):
     $$\omega_0 = \frac{2\pi f_0}{f_s}, \quad g = 10^{G_{\text{dB}} / 40}$$
   - Khi tùy chọn **Gain-Q Interaction** bật (`0x18010FF70`), đối với các kiểu lọc khớp bitmask `0x1074` (Bell & Shelf), hệ số $Q$ được tự động điều chỉnh phụ thuộc vào biên độ $|G_{\text{dB}}|$ (giới hạn đỉnh tại `12.0 dB`):
     $$Q_{\text{effective}} = Q_{\text{UI}} \cdot c^{\min(|G_{\text{dB}}|, 12.0) \cdot s + 0.5}$$
     (gọi hàm `powf` tại `0x1801C4F60`), tạo hành vi kiểu bàn trộn analog cổ điển (khi tăng/giảm gain lớn thì dải thông tự động thu hẹp để giữ độ mượt ở chân đáp tuyến).
3. **Thuật toán Fractional Slope (`0x180112260`)**:
   - Pro-Q 4 cho phép độ dốc bất kỳ giữa `0.0 dB/oct` và `96.0 dB/oct` (ví dụ `14.5 dB/oct`).
   - Disassembly tại `0x180112260` chứng minh Pro-Q 4 **không** dùng bộ lọc bậc phân số (fractional-order calculus), mà duy trì **hai chuỗi cascade Biquad bậc nguyên liền kề** tại offset `rbx + 0xFB0` (bậc $N_1 = \lfloor S / 6 \rfloor$) và `rbx + 0x10F0` (bậc $N_2 = N_1 + 1$).
   - Tại `0x180112470..0x180112517`, đầu ra của hai chuỗi cascade được trộn tuyến tính theo phần lẻ $\alpha = \text{frac}(S / 6) \in [0, 1)$, kết hợp bảng bù biên độ/Q tại `0x1802118E0..0x18021193C`:
     $$y[n] = (1 - \alpha)\, y_{N_1}[n] + \alpha\, y_{N_2}[n]$$

#### B. Natural Phase Mode (`NaturalPhaseFrequencyResponseSource` — `0x180112880..0x180112F80`)
- Trong chế độ `Zero Latency`, phép biến đổi song tuyến tính (Bilinear Transform) bị nén tần số và méo pha gần Nyquist ($\omega \to \pi$).
- Khi `Processing Mode = 1` (`Natural Phase`), Pro-Q 4 kích hoạt `NaturalPhaseFrequencyResponseSource` (`0x180112880`) cùng bộ lọc bù/oversampling ngắn:
  1. Tại `0x180111450..0x180111687`, nó giải hệ phương trình khớp trực tiếp cả **đáp tuyến biên độ và đáp tuyến pha miền tương tự (s-plane)** tại tần số tâm $\omega_0$, Nyquist $\pi$ và điểm tham chiếu trung gian.
  2. Hàm `0x180112900` đánh giá thương số hai đa thức phức bậc cao (tử số tại `[rcx+0x10]`, mẫu số tại `[rcx+0x30]` theo biến $z^{-1} = e^{-j\omega}$) bằng phép nhân đa thức phức nối tiếp.
  3. Độ trễ của `Natural Phase` (`0x18018D9E6..0x18018DB17`) chỉ từ **`0` đến `12` samples** (tùy bậc oversampling 2x/4x nội bộ), trừ khi có band dùng độ dốc `Brickwall` (cộng thêm `320` samples).

#### C. Linear Phase Mode & Non-Uniform Partitioned FFT Convolver (`0x18018D820`, `0x18019EAE0`, `0x1801A0120`)
Từ disassembly hàm tính độ trễ `0x18018D820` và hàm khởi tạo convolver `0x18019EAE0`:
1. **Hệ số tỷ lệ tần số lấy mẫu (`scale`)**:
   - Đọc hằng số `0x1802326C4` $= 2.0833333 \times 10^{-5} = \frac{1}{48,000}$.
   - Tính $s_{\text{raw}} = \max\!\left(1, \text{round}(f_s / 48000.0)\right)$, sau đó làm tròn lên lũy thừa của 2 gần nhất:
     $$\text{scale} = 2^{\lceil \log_2(s_{\text{raw}}) \rceil} = \begin{cases} 1 & \text{nếu } f_s \in \{44.1, 48\}\text{ kHz} \\ 2 & \text{nếu } f_s \in \{88.2, 96\}\text{ kHz} \\ 4 & \text{nếu } f_s \in \{176.4, 192\}\text{ kHz} \end{cases}$$
2. **Bảng chiều dài FIR (`0x180212910`) & Độ trễ Linear Phase**:
   - Tại `0x18019EC7D` (`imul ebx, dword ptr [r8+rax*4+212910h]`), chiều dài xung FIR $N_{\text{FIR}}$ bằng $\text{scale}$ nhân với bảng `TapTable` tại `0x180212910`, và độ trễ bằng đúng một nửa chiều dài FIR ($\text{Latency} = N_{\text{FIR}} / 2$, lệnh `sar eax, 1` tại `0x18018D88A`):

   | `Processing Resolution` | Giá trị bảng `0x180212910` ($N_{\text{FIR}}$ @ 1x) | Latency @ 44.1 / 48 kHz (`scale = 1`) | Latency @ 88.2 / 96 kHz (`scale = 2`) | Latency @ 176.4 / 192 kHz (`scale = 4`) |
   |---|---|---|---|---|
   | `0 = Low` | `4,096` taps | `2,048` samples (~42.7 ms) | `4,096` samples | `8,192` samples |
   | `1 = Medium` (Default) | `8,192` taps | `4,096` samples (~85.3 ms) | `8,192` samples | `16,384` samples |
   | `2 = High` | `16,384` taps | `8,192` samples (~170.7 ms) | `16,384` samples | `32,768` samples |
   | `3 = Very High` | `32,768` taps | `16,384` samples (~341.3 ms) | `32,768` samples | `65,536` samples |
   | `4 = Maximum` | `131,072` taps | `65,536` samples (~1365.3 ms) | `131,072` samples | `262,144` samples |

   - **Trường hợp Brickwall Slope (`Slope = 9`) trong Zero Latency / Natural Phase**: Tại `0x18019EC88` & `0x18019ECFE`, đặt $N_{\text{FIR}} = \text{0x200} = 512$ taps và kích thước phân hoạch đầu $P_0 = \text{0x40} = 64$ samples, tạo độ trễ đúng $256 + 64 = \mathbf{320\text{ samples}}$ (`0x140`, lệnh `lea eax, [rbx+rbx*4]; shl eax, 6` tại `0x18018D9CB`).
3. **Cửa sổ Kaiser (`60.0 dB`) & Bộ chập Phân hoạch Không đồng đều Đệ quy (`0x1801A0120`)**:
   - Tại `0x18019ED3D..0x18019ED93`, đáp tuyến biên độ tổng hợp của 24 band được biến đổi IFFT về miền thời gian, dịch tâm đối xứng $\tau = N_{\text{FIR}}/2$, và nhân với **cửa sổ Kaiser** có tham số suy hao dải chắn **`60.0` (`0x404E000000000000` tại `0x180232D00`)** sinh bởi hàm Bessel $I_0$ tại `0x180102A10`.
   - Hàm `0x1801A0120` xây dựng **Recursive Non-Uniform Partitioned Overlap-Save Convolver**: bắt đầu từ kích thước phân hoạch nhỏ nhất `edi = 64` (`0x40`); nếu $N_{\text{FIR}} \ge 32 \times \text{edi}$ (`shl eax, 5`), nó nhân kích thước phân hoạch lên $\times 8$ (`lea edi, [rdi*8]` $\Rightarrow 64 \to 512 \to 4096 \to 32768$) và gọi đệ quy `0x1801A0120` (`0x1801A024F`) để xử lý phần đầu xung (head partition). Khi thay đổi tham số EQ trên UI, hàm `0x1801A15B0..0x1801A16F5` thực hiện **crossfade tuyến tính từng mẫu** giữa đầu ra của xung FIR cũ và xung FIR mới để chống tiếng nổ (zipper-free IR swap).

#### D. Wideband Dynamic EQ: `Detector` (`0x1801095A0`) & `Capacitor` (`0x18010A9B0`)
Mỗi band bật `Dynamics Enabled = 1` (và `Spectral Enabled = 0`) chạy pipeline 3 khối:
1. **Sidechain Bandpass Isolation**: Lọc tín hiệu kích hoạt (Internal hoặc External Sidechain) qua bộ lọc bandpass tương ứng với tần số/Q của band (hoặc tần số `Side Chain Low/High Frequency` nếu `Side Chain Filtering = 1`).
2. **`Detector` (`0x1801095A0..0x18010A366`) — Đo năng lượng & Đường cong Nén/Giãn Soft-Knee**:
   - Chuyển đổi biên độ sang miền log2/dB bằng xấp xỉ đa thức bậc 3 vector hóa (`0x180109C20..0x180109CFB`).
   - Khi `Dynamics Auto = 1`, ngưỡng hiệu dụng $T_{\text{eff}}$ tự động trượt theo mức nền trung bình dài hạn của tín hiệu (High-pass/crest-factor detector).
   - Tính độ lợi mục tiêu $\Delta G_{\text{target}}$ với nội suy **Soft-Knee bậc 2** (`0x180109F70` & `0x18010A102`), giới hạn asymptote bởi `Dynamic Range` ($\Delta G_{\max} \in [-30, +30]\text{ dB}$).
3. **`Capacitor` (`0x18010A9B0..0x18010B65E`) — Mạch Envelope Follower Phụ thuộc Chương trình (Program-Dependent RC Ballistics)**:
   - Chạy hoàn toàn ở độ chính xác kép (`double`, các thanh ghi `xmm6..xmm15`).
   - Tại `0x18010B2C0`, nó so sánh độ chênh lệch mức năng lượng hiện tại với trạng thái tụ điện (`comisd xmm6, [0x180233098]`):
     - **Nhánh Attack (`0x18010B2D3..0x18010B3F9`)**: Tra bảng hằng số thời gian phi tuyến tại `[rbx+0x7D0]` theo biên độ vượt ngưỡng, nhân với hệ số `Band k Attack` (`[rbx+0x28]`), và tính hệ số làm mượt $\alpha_{\text{att}} = \exp(-\Delta t / \tau_{\text{att}})$ qua `call 0x1801E3520` (`exp`).
     - **Nhánh Release (`0x18010B3FE..0x18010B611`)**: Phóng điện kép (dual-stage program-dependent release) — các gai transient ngắn được nhả rất nhanh trong khi âm ngân dài được nhả mượt hơn, tính $\alpha_{\text{rel}} = \exp(-\Delta t / \tau_{\text{rel}})$ qua `call 0x1801E3520` tại `0x18010B5D4`.

#### E. Thuật toán Spectral Dynamics (`0x18019F1D0`, `0x1801A1F30`, `0x1801A28F0`)
Khi bất kỳ band nào bật `Spectral Enabled = 1` (`[band + 0x1D48] != 0`):
1. **Kích thước cửa sổ STFT & Độ trễ (`0x18019F1D0` & `0x18018D8B9`)**:
   - Tại `0x18019F288`, kích thước khung STFT được đặt bằng:
     $$N_{\text{STFT}} = \text{scale} \ll 11 = 2048 \times \text{scale}$$
     (tức `2048` samples tại 44.1/48 kHz, `4096` tại 88.2/96 kHz, `8192` tại 176.4/192 kHz).
   - Tại `0x18018D8B9`, độ trễ cộng thêm khi có ít nhất 1 band Spectral Dynamics là:
     $$\text{Latency}_{\text{spectral}} = \text{scale} \ll 10 = 1024 \times \text{scale}\text{ samples}$$
     (tức `1,024` samples $\approx 21.3\text{ ms}$ ở 48 kHz).
2. **Xử lý Từng Bin Tần số (`0x1801A28F0..0x1801A2B6E`)**:
   - Tín hiệu được phân tích STFT chồng lấp 75% (hop size $H = N_{\text{STFT}} / 4 = 512 \times \text{scale}$).
   - Hàm `0x1801A2C4B..0x1801A2D08` ánh xạ đáp tuyến của các band bật Spectral Dynamics thành mặt nạ trọng số tần số $W_{\text{band}}[k]$ và tính bán kính làm mượt phổ theo tham số **`Spectral Density`** (`[rbx+0x98]`, $0\% \dots 100\%$). Khi `Spectral Density` thấp, mặt nạ gain được làm mượt rộng hơn trên trục tần số; khi `Spectral Density` cao, mỗi bin phản ứng độc lập hơn.
   - Tại vòng lặp SIMD `0x1801A2990..0x1801A2AE6`, trên từng cụm 4 bin tần số (`shr r8, 2`), thuật toán so sánh năng lượng bin hiện tại (`xmm3`) với ngưỡng tham chiếu động (`xmm0`), tính tỷ số nén phân thức bậc hai:
     $$G_{\text{bin}}[k] = \text{clamp}\!\left(\left(\frac{1}{1 - D[k]}\right)^2 \cdot E[k],\; E_{\min}[k],\; E_{\max}[k]\right)$$
     và nhân mặt nạ gain phổ vào phổ phức trước khi tổng hợp IFFT + Overlap-Add.

#### F. Character Modes (`0 = Clean`, `1 = Subtle`, `2 = Warm` — `0x18019DA1F..0x18019DB82`)
Trong `ProQProcessor::updateParameters` (`0x18019CBE0` tại `0x18019DA1F`):
- Đọc tham số `Character` tại `[rdi+0x648]`:
  - `0 = Clean`: Tắt bộ bão hòa (`WaveShaper` / `FunctionTableClipper`), đường truyền tuyến tính tuyệt đối.
  - `1 = Subtle`: Gọi `0x18018DDB0` lấy hệ số drive nhẹ, tính đường cong bão hòa bậc hai đối xứng/bất đối xứng nhẹ tại `0x18019DA47` (hằng số `0x180232BA4 = 1.975`, `0x180232C2C = 2.1419137`, `0x1802330C8 = -0.074325`).
  - `2 = Warm`: Tăng hệ số drive và độ lệch hài bậc chẵn (even/odd harmonic saturation kiểu biến áp & đèn điện tử) tại `0x18019DA86` (hằng số `0x1802328AC = 0.562745`, `0x180232B90`, `0x180232BAC = 1.9856194`, `0x180232BE0 = 2.607522`).

#### G. Thuật toán Auto Gain (`0x18019BD20` & `0x18019C1B0`)
- Khi khởi tạo lần đầu, hàm `0x18019C1B0` tạo bảng trọng số cảm thụ **512 bin** (`0x200` floats tại `0x1802AD678`) trải đều theo octave trên đoạn $[20\text{ Hz}, 20,000\text{ Hz}]$ ($\log_2 f \in [4.321928, 14.287712]$) bằng cách nội suy tuyến tính trên **8 nút tần số - trọng số** tại `0x180212988` và `0x1802129A8`:

  | Knot Index | `0` | `1` | `2` | `3` | `4` | `5` | `6` | `7` |
  |---|---|---|---|---|---|---|---|---|
  | **Tần số (Hz)** (`0x180212988`) | `10.0` | `20.0` | `30.0` | `60.0` | `90.0` | `6,250.0` | `18,000.0` | `20,000.0` |
  | **Trọng số $w_k$** (`0x1802129A8`) | `0.000` | `0.035` | `0.200` | `0.850` | `1.000` | `1.000` | `0.400` | `0.000` |

- Sau khi chuẩn hóa $\widetilde{W}[i] = \frac{W[i]}{\sum_{j=0}^{511} W[j]}$, hàm `0x18019BD20` tính tổng biên độ đáp tuyến tổng hợp $|H_{\text{total}}(f_i)| = \sqrt{|H_{\text{total}}(f_i)|^2}$ bằng lệnh SIMD `sqrtps`, `mulps`, `addps` và trả về hệ số bù:
  $$G_{\text{AutoGain}} = \frac{1.0}{\max\!\left(10^{-9},\; \sum_{i=0}^{511} |H_{\text{total}}(f_i)| \cdot \widetilde{W}[i]\right)}$$

#### H. Đồ họa OpenGL (`gui::OpenGLRenderer` — `0x180055A90..0x180059C00`)
- Pro-Q 4 sử dụng `gui::OpenGLRenderer` (VTable `0x180216520`) kết hợp `gui::OpenGLTexture` (`0x1802163E0`) và `gui::GlyphTextureAtlas` (`0x180216C70`).
- Thay vì dùng MSAA phần cứng nặng nề, `gui::OpenGLRenderer` tesselate đường cong đáp tuyến EQ và phổ FFT thành dải tam giác (triangle strips) có biên mở rộng `1.0 px` ở hai phía, gán alpha $= 1.0$ ở tâm đường và alpha $= 0.0$ ở mép ngoài (`0x180057200..0x180058F00`), đồng thời tô vùng dưới đường cong (curve fill) bằng gradient alpha mượt.

---

## PHẦN 2: OEKSOUND SOOTHE3 (`v1.0.5`) — DEEP DSP, SYMBOLS MAP & SHADER REVERSE ENGINEERING

### 2.1. Hệ thống 143 Tham số (`DigiPageTable.xml` & 68 JSON Presets)

Phân tích [`DigiPageTable.xml`](file:///f:/FL%20studio/extracted_targets/soothe3/DigiPageTable.xml), bảng chuỗi `.rdata` của [`soothe3.vst3`](file:///f:/FL%20studio/extracted_targets/soothe3/soothe3.vst3), [`soothe3_manual.pdf`](file:///f:/FL%20studio/extracted_targets/soothe3/soothe3_manual.pdf) và toàn bộ **68 file `.preset`** xác nhận mỗi file `.preset` chứa chính xác **143 tham số trong `params`**:
- **26 Tham số Global**: Điều khiển thuật toán STFT, Depth, Detail, Attack, Release, Tilts, Mix/Trim, Stereo/Sidechain.
- **72 Tham số 8-Band Sidechain EQ**: `8 Bands × 9 Params/Band` (`Band 1` .. `Band 8`).
- **45 Tham số Multichannel / Dolby Atmos**: `16` cờ bật kênh (`Channel 1..16 enabled`), `25` tham số cho 5 nhóm loa (`LR`, `Center`, `LFE`, `Surround`, `Ceiling` × 5 thuộc tính `enabled`, `depth`, `delta`, `solo`, `link all`), và `4` tham số liên kết nhóm (`Link all`, `Group link LCR`, `Group link surround`, `Group link ceiling`).

#### A. Nhóm Tham số Thuật toán Chính (Global Parameters)

| Tên tham số (`.preset` JSON Key / `DigiPageTable.xml`) | Dải giá trị & Mặc định | Đơn vị / Bậc rời rạc | Vai trò trong Pipeline DSP (`soothe3_manual.pdf` tr. 71–99) |
|---|---|---|---|
| `Mode` (`Process Mode`) | `0` (`0`, `1`) | `0 = soft`, `1 = hard` | Chọn giải thuật nén cộng hưởng: `soft` (tự thích ứng, giữ màu âm tự nhiên) hoặc `hard` (phụ thuộc ngưỡng tuyệt đối, cắt mạnh). |
| `Delta` (`Processing Type`) | `0` (`0`, `1`) | `0 = suppress`, `1 = delta/solo` | `0`: cắt cộng hưởng; `1`: chỉ nghe phần năng lượng bị triệt tiêu (Delta). |
| `Depth` | `0.0 .. 10.0` (def `2.5`) | Tham số không thứ nguyên | Độ sâu triệt tiêu năng lượng cộng hưởng (Drive vào bộ tính ngưỡng động). |
| `Detail` | `0.0 .. 10.0` (def `5.0`) | Tham số không thứ nguyên | Hợp nhất 2 tham số `Sharpness` & `Selectivity` của Soothe2: điều khiển đồng thời độ hẹp vết cắt ($Q$) và mật độ chọn lọc đỉnh cộng hưởng. |
| `Detail tilt low` | `0.1 .. 10.0` (def `1.0`) | Multiplier ($\times$) | Nhân tỷ lệ `Detail` cho miền tần số thấp ($f < 500\text{ Hz}$, tối đa dưới $300\text{–}400\text{ Hz}$). |
| `Detail tilt high` | `0.1 .. 10.0` (def `1.0`) | Multiplier ($\times$) | Nhân tỷ lệ `Detail` cho miền tần số cao ($f > 2,000\text{ Hz}$). |
| `Attack` | `0.0 .. 10.0` (def `2.0`) | Tốc độ phản ứng | Tốc độ giảm gain khi phát hiện cộng hưởng mới. |
| `Attack tilt low` | `0.1 .. 10.0` (def `1.0`) | Multiplier ($\times$) | Nhân tỷ lệ `Attack` cho miền tần số thấp ($f < 500\text{ Hz}$). |
| `Attack tilt high` | `0.1 .. 10.0` (def `1.0`) | Multiplier ($\times$) | Nhân tỷ lệ `Attack` cho miền tần số cao ($f > 2,000\text{ Hz}$). |
| `Release` | `0.0 .. 10.0` (def `2.0`) | Tốc độ hồi phục | Tốc độ nhả gain về `0 dB` khi cộng hưởng biến mất. |
| `Release tilt low` | `0.1 .. 10.0` (def `1.0`) | Multiplier ($\times$) | Nhân tỷ lệ `Release` cho miền tần số thấp ($f < 500\text{ Hz}$). |
| `Release tilt high` | `0.1 .. 10.0` (def `1.0`) | Multiplier ($\times$) | Nhân tỷ lệ `Release` cho miền tần số cao ($f > 2,000\text{ Hz}$). |
| `Max cut` | `0.0 .. 36.0` (UI `-36.0 .. 0.0 dB`) | dB | Giới hạn trần mức cắt giảm tối đa trên mỗi bin tần số (ở `36 dB` / `-36 dB` là không giới hạn). |
| `Mix` | `0.0 .. 100.0` (def `100.0`) | `%` (Dry/Wet) | Tỷ lệ pha trộn giữa tín hiệu gốc (đã bù trễ) và tín hiệu đã xử lý. |
| `Out trim` / `Wet trim` | `-24.0 .. +24.0` (def `0.0`) | dB | Độ lợi đầu ra tổng (`Out trim`) và độ lợi nhánh Wet (`Wet trim`). |
| `Inverse` | `0` (`0`, `1`) | Bool | Đảo ngược xử lý: giữ lại phần cộng hưởng / nền tĩnh và loại bỏ phần kia (Unmasking / Ducking). |
| `Sidechain` / `Sidechain listen` | `0` (`0`, `1`) | Bool | Sử dụng tín hiệu External Sidechain để tính mặt nạ giảm gain / nghe thử Sidechain. |
| `Linear phase` | `0` (`0`, `1`) | Bool | Chuyển bộ tái tổng hợp từ Minimum-Phase (`2304` samples) sang Linear-Phase (`3456` samples). |
| `Low latency` | `0` (`0`, `1`) | Bool | Chế độ Zero-Latency (`0` samples @ 44.1/48k, `96` @ 88.2/96k, `192` @ 176.4/192k). |
| `Quality` | `0` (`0`, `1`, `2`) | `0=normal`, `1=high`, `2=ultra` | Độ phân giải thời gian (Time Resolution / Hop-size density) của bộ xử lý STFT thực thời. |
| `Stereo mode` (`Routing Mode`) | `0` (`0`, `1`) | `0 = L/R`, `1 = M/S` | Chế độ xử lý kênh Stereo (Left/Right hoặc Mid/Side). |
| `Stereo link` (`Channel link`) | `0.0 .. 1.0` (`0..100%`) | Ratio / `%` | Mức độ liên kết mặt nạ giảm gain giữa hai kênh L/R hoặc M/S. |
| `Stereo focus` (`Channel balance`) | `0.0 .. 1.0` (def `0.5`) | Balance (`-1..+1`) | Trọng số ưu tiên xử lý giữa Left vs Right hoặc Mid vs Side (`0.5` = cân bằng). |
| `Bypass` / `Auto gain` | `0` (`0`, `1`) | Bool | Soft-bypass toàn plugin / Tự động bù năng lượng đầu ra (`dsp::ShortTermGainMeasure`). |

#### B. Hệ thống 8-Band Sidechain EQ (`72 Parameters`: `8 Bands × 9 Params`)
Khác với Soothe2 (chỉ có 6 band), Soothe3 nâng cấp lên **8 Band Sidechain EQ** (`Band 1` .. `Band 8`). Mỗi band có **9 tham số** trong `.preset`:
1. `Band b used` (`0/1`): Trạng thái kích hoạt node trên đồ thị.
2. `Band b enabled` (`0/1`): Bật/tắt band.
3. `Band b shape` (**8 kiểu lọc `0..7`**, xác minh từ `.rdata` offset `33779224` và 68 presets):
   - `0 = Low cut`
   - `1 = Low shelf`
   - `2 = Bell`
   - `3 = Bandpass` *(Khi bật nhiều band `Bandpass`, Soothe3 định tuyến chúng **song song / parallel** để cô lập nhiều vùng tần số rời rạc)*
   - `4 = Band reject` *(Notch rộng)*
   - `5 = Tilt`
   - `6 = High shelf`
   - `7 = High cut`
4. `Band b frequency`: $20.0\text{ Hz} \dots 20,000.0\text{ Hz}$ (thang logarit).
5. `Band b depth` (`Band b gain (db)`): $-10.0 \dots +10.0$ (tương ứng $-24\text{ dB} \dots +24\text{ dB}$ độ nhạy phát hiện cộng hưởng tại vùng tần số đó).
6. `Band b q`: $0.1 \dots 16.0$ (mặc định `0.707`).
7. `Band b slope`: `0.0 .. 10.0` (độ dốc liên tục cho Low/High Cut & Shelf, mặc định `5.0`).
8. `Band b focus` (`Band b balance`): `0.0 .. 1.0` (mặc định `0.5`, cân bằng độ nhạy L/R hoặc M/S riêng cho band).
9. `Band b channels` (`Band b stereo mode`): Định tuyến nhóm kênh/M-S riêng cho từng band.

---

### 2.2. Kiến trúc C++ từ `soothe3_symbols.map` (26.18 MB, 142,863 dòng)

Phân tích toàn bộ [`soothe3_symbols.map`](file:///f:/FL%20studio/extracted_targets/soothe3/soothe3_symbols.map) cho thấy kiến trúc phân tầng sạch sẽ của oeksound (viết bằng C++20 + Rust `clap_wrapper` + PFFFT SIMD + Intel IPP + EnTT ECS + Chokidar/SQLite preset cache):

```
oeksound::Soothe3Plugin (soothe3_plugin.cpp.obj)
 ├── aani::ProcessEngine / aani::AudioBlockTopology (Quản lý luồng audio, crossfade topology, bypass)
 ├── dsp::MainProcessor (main_processor.cpp.obj — Trái tim DSP của Soothe3)
 │    ├── dsp::Resampler<16, float, 16384> (Chuẩn hóa 2x/4x sample rate về 1x base rate 44.1/48 kHz)
 │    ├── dsp::WeightFilter / dsp::SidechainEQ (8-band IIR SVF/Biquad + Parallel Bandpass routing)
 │    ├── dsp::StftProcessor / PFFFT (pffft.c.obj — SIMD Radix-2/4 Real FFT, AVX/SSE2 + IPPCODE)
 │    ├── dsp::SpectralAnalyzer & ResonanceSuppressor (Soft/Hard threshold, Detail, Tilts, Max cut)
 │    ├── dsp::MinimumPhaseReconstructor (Cepstral / Hilbert minimum-phase conversion)
 │    └── dsp::ShortTermGainMeasure (Đo năng lượng LUFS/RMS ngắn hạn cho Auto Gain)
 ├── vu::Entity / EnTT Registry (ECS quản lý state UI)
 ├── kali::Renderer / kali::Shader (Engine đồ họa GPU đa backend OpenGL / Direct3D 11 / Metal)
 └── visu::GraphRenderer / visu::SpectrumAnalyzer (Render phổ FFT và đường cong Reduction)
```

---

### 2.3. Giải mã Chi tiết Pipeline STFT & Thuật toán Dynamic Resonance Suppression của Soothe3

#### A. Chuẩn hóa Tốc độ Lấy mẫu (`dsp::Resampler<16, float, 16384>`) & Bảng Độ trễ Chính xác
- Soothe3 thiết kế lõi STFT hoạt động ở tốc độ lấy mẫu nền $f_{\text{base}} \in [44.1, 48.0]\text{ kHz}$.
- Khi chạy ở $88.2/96\text{ kHz}$ (2x) hoặc $176.4/192\text{ kHz}$ (4x), `dsp::Resampler<16, float, 16384>` tách dải tần, hạ mẫu (downsample) về $f_{\text{base}}$ bằng bộ lọc polyphase FIR 16-tap/branch để tính toán và áp dụng mặt nạ triệt cộng hưởng (vốn nằm dưới $20\text{ kHz}$), sau đó nâng mẫu (upsample) trở lại và ghép với phần siêu âm ($> f_{\text{base}}/2$).
- Độ trễ chính xác trên từng chế độ (`soothe3_manual.pdf` trang 100):

  | Chế độ xử lý (`Linear phase` / `Low latency`) | Độ trễ @ `44.1 / 48 kHz` (1x) | Độ trễ @ `88.2 / 96 kHz` (2x) | Độ trễ @ `176.4 / 192 kHz` (4x) |
  |---|---|---|---|
  | **Regular (Minimum-Phase)** (`Linear=0, LowLat=0`) | **`2,304` samples** (~48.0 ms @ 48k) | **`4,704` samples** ($2 \times 2304 + 96$) | **`9,408` samples** ($4 \times 2304 + 192$) |
  | **Low Latency (Minimum-Phase)** (`Linear=0, LowLat=1`) | **`0` samples** (Zero latency / IIR-hybrid) | **`96` samples** (chỉ trễ Resampler) | **`192` samples** (chỉ trễ Resampler) |
  | **Linear-Phase Mode** (`Linear=1, LowLat=0`) | **`3,456` samples** (~72.0 ms @ 48k) | **`7,008` samples** ($2 \times 3456 + 96$) | **`14,016` samples** ($4 \times 3456 + 192$) |

- Cấu trúc khung STFT tại 1x (`44.1/48 kHz`):
  - Kích thước FFT chuẩn: $N_{\text{FFT}} = 4096$ samples ($2049$ bin phức từ DC đến Nyquist, độ phân giải tần số $\Delta f \approx 11.72\text{ Hz/bin}$).
  - Trong chế độ `Linear-Phase`, độ trễ `3,456` samples tương ứng với cửa sổ phân tích/tổng hợp đối xứng $\frac{3}{4} N_{\text{FFT}} + 384 = 3456$ samples.
  - Tham số **`Quality`** (`0 = normal`, `1 = high`, `2 = ultra`) điều chỉnh **Time Resolution (bước nhảy khung STFT $H$)**:
    - `normal`: $H = 512$ samples (8x overlap)
    - `high`: $H = 256$ samples (16x overlap)
    - `ultra`: $H = 128$ samples (32x overlap)

#### B. Pipeline Phát hiện và Triệt tiêu Cộng hưởng Động (Dynamic Resonance Suppression)
Với mỗi khung STFT $m$ và mỗi bin tần số $k \in \{0 \dots N_{\text{FFT}}/2\}$ tại tần số $f_k = \frac{k \cdot f_{\text{base}}}{N_{\text{FFT}}}$:

1. **Bước 1: Đáp tuyến Trọng số Sidechain EQ 8-Band ($W_{\text{EQ}}[k]$)**:
   - Tín hiệu kích hoạt (Input hoặc External Sidechain) đi qua 8 band EQ:
     - Các band có `shape != 3` (`Low cut, Low shelf, Bell, Band reject, Tilt, High shelf, High cut`) được mắc **nối tiếp (cascade)**: $H_{\text{series}}(f_k) = \prod_{b \notin \mathcal{B}_{\text{BP}}} H_b(f_k)$.
     - Nếu có các band chọn `shape == 3` (`Bandpass`, tập $\mathcal{B}_{\text{BP}} \neq \emptyset$), chúng được mắc **song song (parallel)** rồi nhân với nhánh nối tiếp:
       $$W_{\text{EQ}}(f_k) = H_{\text{series}}(f_k) \cdot \sum_{b \in \mathcal{B}_{\text{BP}}} H_b(f_k)$$
   - Phổ năng lượng đầu vào của bộ phát hiện là:
     $$P_{\text{SC}}[m, k] = |X_{\text{SC}}[m, k]|^2 \cdot |W_{\text{EQ}}(f_k)|^2$$

2. **Bước 2: Đường cong Tilt Tần số cho `Detail`, `Attack`, `Release`**:
   - Với mỗi tham số $P \in \{\text{Detail}, \text{Attack}, \text{Release}\}$, hệ số nghiêng tần số dùng hai hàm chuyển tiếp mượt kiểu shelf quanh $f_{\text{low}} = 500\text{ Hz}$ và $f_{\text{high}} = 2000\text{ Hz}$:
     $$\mathcal{T}_P(f_k) = \left(T_{P,\text{low}}\right)^{S_{\text{low}}(f_k)} \cdot \left(T_{P,\text{high}}\right)^{S_{\text{high}}(f_k)}$$
     trong đó $S_{\text{low}}(f_k) = \frac{1}{1 + (f_k / 500)^2} \in [0, 1]$ và $S_{\text{high}}(f_k) = \frac{(f_k / 2000)^2}{1 + (f_k / 2000)^2} \in [0, 1]$. Khi $T_{P,\text{low}} = T_{P,\text{high}} = 1.0$ (mặc định), $\mathcal{T}_P(f_k) \equiv 1.0$.
   - Giá trị hiệu dụng tại bin $k$:
     $$D_{\text{eff}}[k] = \text{Detail} \cdot \mathcal{T}_{\text{Detail}}(f_k), \quad A_{\text{eff}}[k] = \text{Attack} \cdot \mathcal{T}_{\text{Attack}}(f_k), \quad R_{\text{eff}}[k] = \text{Release} \cdot \mathcal{T}_{\text{Release}}(f_k)$$

3. **Bước 3: Ước lượng Phổ Nền Tham chiếu $\overline{P}_{\text{ref}}[m, k]$ & Tách Đỉnh Cộng hưởng**:
   - Một đỉnh cộng hưởng (resonance) là nơi mật độ năng lượng cục bộ vượt trội bất thường so với vùng tần số lân cận (local spectral neighborhood).
   - Soothe3 làm mượt phổ năng lượng $P_{\text{SC}}[m, k]$ trên thang tần số cảm thụ (ERB / Bark scale) với bán kính cửa sổ rộng $\sigma_{\text{ref}}$ để thu được **đường bao phổ nền (Local Spectral Baseline)** $\overline{P}_{\text{ref}}[m, k]$, và với bán kính hẹp $\sigma_{\text{det}}(D_{\text{eff}}[k])$ (tỷ lệ nghịch với $D_{\text{eff}}[k]$) để thu được **phổ chi tiết** $P_{\text{det}}[m, k]$.

4. **Bước 4: Chế độ `Soft` vs `Hard` và Tính Mức Cắt Mục tiêu ($\Delta G_{\text{target}}[m, k]$)**:
   - Chuyển sang miền dB: $L_{\text{det}}[m, k] = 10 \log_{10}(P_{\text{det}}[m, k] + \epsilon)$, $L_{\text{ref}}[m, k] = 10 \log_{10}(\overline{P}_{\text{ref}}[m, k] + \epsilon)$.
   - **Chế độ `Soft` (`Mode = 0`) — Level-Independent Spectral Crest Suppression**:
     - Chỉ phụ thuộc vào **độ nhô tương đối (Spectral Prominence)** $\Delta L_{\text{prom}}[m, k] = L_{\text{det}}[m, k] - L_{\text{ref}}[m, k]$ kết hợp ngưỡng chọn lọc phụ thuộc $D_{\text{eff}}[k]$:
       $$E_{\text{soft}}[m, k] = \max\!\left(0,\; \Delta L_{\text{prom}}[m, k] - \Theta_{\text{sel}}(D_{\text{eff}}[k])\right)$$
     - Nhân với hệ số dẫn động theo `Depth`:
       $$\Delta G_{\text{raw}}[m, k] = -\,\kappa_{\text{soft}}(\text{Depth}) \cdot \text{SoftKnee}\!\left(E_{\text{soft}}[m, k]\right)$$
   - **Chế độ `Hard` (`Mode = 1`) — Level-Dependent Dynamic Threshold Suppression**:
     - Kết hợp cả độ nhô tương đối $\Delta L_{\text{prom}}[m, k]$ và mức năng lượng tuyệt đối $L_{\text{det}}[m, k]$ so với ngưỡng tuyệt đối hạ xuống theo `Depth`:
       $$E_{\text{hard}}[m, k] = \max\!\left(0,\; \alpha \Delta L_{\text{prom}}[m, k] + (1-\alpha)\left(L_{\text{det}}[m, k] - \Theta_{\text{abs}}(\text{Depth})\right)\right)$$
       $$\Delta G_{\text{raw}}[m, k] = -\,\kappa_{\text{hard}}(\text{Depth}) \cdot E_{\text{hard}}[m, k]$$

5. **Bước 5: Làm mượt Phổ theo `Detail` & Giới hạn `Max cut`**:
   - Mặt nạ giảm gain $\Delta G_{\text{raw}}[m, k]$ được lọc thông thấp trên trục tần số với bề rộng nhân (kernel width) tỷ lệ nghịch với $D_{\text{eff}}[k]$ (khi `Detail` cao, vết cắt sắc nhọn như bộ lọc Notch $Q$ cao; khi `Detail` thấp, vết cắt rộng và mượt).
   - Nếu `Max cut < 36.0 dB` (tức giới hạn trần được kích hoạt), áp dụng giới hạn tại $M_{\text{cut}} = -|\text{Max cut}|$:
     $$\Delta G_{\text{clamped}}[m, k] = \max\!\left(M_{\text{cut}},\; \Delta G_{\text{smooth}}[m, k]\right)$$

6. **Bước 6: Ballistics Thời gian (`Attack` / `Release`) & Liên kết Kênh (`Stereo link` / Atmos Groups)**:
   - Với mỗi bin $k$, chuyển đổi tham số `Attack` và `Release` đã nhân Tilt ($A_{\text{eff}}[k], R_{\text{eff}}[k]$) thành hệ số làm mượt một cực theo bước nhảy khung $\Delta t = H / f_{\text{base}}$:
     $$\Delta G_{\text{env}}[m, k] = \begin{cases} \alpha_{\text{att}}[k]\,\Delta G_{\text{env}}[m-1, k] + (1 - \alpha_{\text{att}}[k])\,\Delta G_{\text{clamped}}[m, k] & \text{nếu } \Delta G_{\text{clamped}} < \Delta G_{\text{env}} \\ \alpha_{\text{rel}}[k]\,\Delta G_{\text{env}}[m-1, k] + (1 - \alpha_{\text{rel}}[k])\,\Delta G_{\text{clamped}}[m, k] & \text{ngược lại} \end{cases}$$
   - Áp dụng `Stereo link` ($\lambda \in [0, 1]$) giữa kênh $c_1, c_2$:
     $$\Delta G_{c_1}^{\text{linked}}[m, k] = (1 - \lambda)\,\Delta G_{c_1}[m, k] + \lambda \min\!\left(\Delta G_{c_1}[m, k], \Delta G_{c_2}[m, k]\right)$$

7. **Bước 7: Chuyển đổi Minimum-Phase (`dsp::MinimumPhaseReconstructor`), `Inverse` và `Delta` (Solo)**:
   - Biên độ tuyến tính của bộ lọc triệt cộng hưởng: $G_{\text{lin}}[m, k] = 10^{\Delta G_{\text{env}}[m, k] / 20} \in (0, 1]$.
   - Nếu **`Inverse = 1`**: Đảo ngược mặt nạ thành $G_{\text{inv}}[m, k] = 1.0 - G_{\text{lin}}[m, k]$ (khi không có External Sidechain, giữ lại phần cộng hưởng; khi có External Sidechain, cắt tín hiệu chính tại đúng các tần số nổi bật của Sidechain để tạo khoảng trống — spectral unmasking).
   - Nếu **`Linear phase = 1`**: Nhân trực tiếp biên độ thực $G[m, k]$ vào phổ phức $X[m, k]$ (pha bằng 0 $\Rightarrow$ đáp tuyến xung đối xứng).
   - Nếu **`Linear phase = 0`** (Regular Minimum-Phase): Sử dụng **phương pháp Cepstrum thực (Hilbert Transform trên $\ln G[m, k]$)** để khôi phục pha tối thiểu $\phi_{\min}[m, k] = -\mathcal{H}\{\ln G[m, k]\}$, tạo đáp tuyến phức $H_{\min}[m, k] = G[m, k] e^{j\phi_{\min}[m, k]}$ giúp loại bỏ hoàn toàn hiện tượng pre-ringing trên transient!
   - Nếu **`Delta = 1` (`solo`)**: Xuất phần hiệu số $Y_{\text{delta}}[m, k] = X[m, k] \cdot (1 - H[m, k])$.

---

### 2.4. Toàn bộ 6 Cặp Shader Đồ họa GPU Nhúng trong `soothe3.vst3` (`.rdata`)

Từ vùng `.rdata` của [`soothe3.vst3`](file:///f:/FL%20studio/extracted_targets/soothe3/soothe3.vst3) (offsets `33235488 .. 34028000`), engine đồ họa `kali::Renderer` nhúng trực tiếp **6 cặp Vertex/Fragment Shader** đa nền tảng (tự động dịch sang HLSL D3D11, GLSL OpenGL và MSL Metal thông qua các macro `BEGIN_CONSTANTS`, `SAMPLE_TEXTURE_2D`, `float4`):

1. **`reductionCurveVertexShader` & `reductionCurveFragmentShader` (Offset `33235488`)**:
   - **Vai trò**: Vẽ đường cong cắt giảm cộng hưởng màu xanh/tím thời gian thực (`visu::ReductionCurve`).
   - **Cơ chế toán học**: Nhận mảng hằng số `float4 ys[256]` (**1,024 điểm tần số** được đóng gói 4 float/vector: `ys[idx / 4][idx % 4]`). Trong Fragment Shader, tính đạo hàm không gian màn hình `float2 d = float2(fwidth(input.x), fwidth(input.y))` và giải khoảng cách có dấu từ pixel tới đoạn thẳng nối 2 điểm liền kề `(x0, y0) -> (x1, y1)`:
     ```hlsl
     float2 p1p0 = float2(x1 - x0, y1 - y0) / d;
     float signed_dist_px = (-p1p0.y * p0.x + p1p0.x * p0.y) / length(p1p0);
     float t = smoothstep(-1.0, 1.0, signed_dist_px);
     return lerp(fill_color, line_color, exp(-signed_dist_px * signed_dist_px * 0.2));
     ```
2. **`outlineVertexShader` & `outlineFragmentShader` (Offset `33985472`)**:
   - **Vai trò**: Vẽ đường viền đáp tuyến EQ và phổ FFT từ 2 texture 1D (`tex0`, `tex1`).
   - **Cơ chế toán học**: Trong Vertex Shader, đọc 3 điểm liên tiếp `p0, p1, p2` từ `SAMPLE_TEXTURE_2D_LOD`, tính vector pháp tuyến đơn vị của 2 đoạn thẳng và tìm giao điểm của hai đường thẳng song song cách một khoảng `offset` (`calculate_offset(p0, p1, p2)`) để mở rộng bề dày nét vẽ (miter join) mà không bị biến dạng góc nhọn.
3. **`renderTargetVertexShader` & `renderTargetFragmentShader` (Offset `33995376`)**:
   - **Vai trò**: Compositing framebuffer offscreen với hiệu ứng làm mờ/khử bão hòa khi mở popup hoặc bypass.
   - **Cơ chế toán học**: Tính độ chói chuẩn Rec.709 `float3 bw = dot(color.rgb, float3(0.2126, 0.7152, 0.0722))` và nội suy `lerp(color.rgb, bw, desaturate) * brightness`.
4. **`shapeVertexShader` & `shapeFragmentShader` (Offset `34002400`)**:
   - **Vai trò**: Uber-shader dựng toàn bộ các nút vặn (knobs), thanh trượt, nút bấm, nhãn chữ MSDF và bóng đổ (drop shadows) của giao diện Soothe3.
   - **Cơ chế toán học**: Đánh giá chuỗi phép toán hình học CSG (`op`: `0=union`, `1=intersect`, `2=subtract`, `3=replace`) trên tối đa `MAX_SHAPES` nguyên hình SDF (`0 = rounded_rect`, `1 = semi_circle_ring`, `2 = rect`, `3 = Newton-Raphson exact ellipse`, `4 = MSDF text glyph`, `5 = SDF sprite`, `6 = squircle`), kết hợp bóng đổ hàm mũ `exp(-outside / shadow_sigma)`.
5. **`textureVertexShader` & `textureFragmentShader` (Offset `34018480`)**:
   - **Vai trò**: Vẽ quad/nine-patch phủ texture có premultiplied alpha và mask.
6. **`gradientVertexShader` & `gradientFragmentShader` (Offset `34025232`)**:
   - **Vai trò**: Tô màu chuyển sắc đa điểm dừng (`stops[8]`, tối đa 8 stops) theo 4 chế độ (`0 = Linear`, `1 = Radial`, `2 = Linear LUT texture`, `3 = Radial LUT texture`).

---

## PHẦN 3: ĐẶC TẢ TOÁN HỌC CLEAN-ROOM & THIẾT KẾ KIẾN TRÚC RUST (`PRO-Q 4` & `SOOTHE3`)

Dưới đây là thiết kế kiến trúc Rust chuẩn `no_std`/real-time audio-safe (không cấp phát bộ nhớ trên luồng âm thanh, hỗ trợ SIMD `f32x4`/`f64x2`) sẵn sàng cho giai đoạn triển khai Engine:

### 3.1. Module 1: `proq4_engine` (Pure-Rust Clean-Room Architecture)

```rust
//! Clean-Room Rust Architecture for FabFilter Pro-Q 4 DSP Engine (`proq4_engine`)
//! Real-time audio thread safe: zero heap allocations in `process_block`.

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessingMode {
    ZeroLatency = 0,
    NaturalPhase = 1,
    LinearPhase = 2,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearPhaseResolution {
    Low = 0,      // 4096 taps @ 1x (2048 samples latency)
    Medium = 1,   // 8192 taps @ 1x (4096 samples latency)
    High = 2,     // 16384 taps @ 1x (8192 samples latency)
    VeryHigh = 3, // 32768 taps @ 1x (16384 samples latency)
    Maximum = 4,  // 131072 taps @ 1x (65536 samples latency)
}

impl LinearPhaseResolution {
    pub const TAP_TABLE_1X: [usize; 5] = [4096, 8192, 16384, 32768, 131072];

    #[inline]
    pub fn fir_length(self, sample_rate: f32) -> usize {
        let scale = ((sample_rate / 48_000.0).round().max(1.0) as usize).next_power_of_two();
        scale * Self::TAP_TABLE_1X[self as usize]
    }

    #[inline]
    pub fn latency_samples(self, sample_rate: f32) -> usize {
        self.fir_length(sample_rate) >> 1
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CharacterMode {
    Clean = 0,
    Subtle = 1,
    Warm = 2,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterShape {
    Bell = 0,
    LowShelf = 1,
    HighShelf = 2,
    LowCut = 3,
    HighCut = 4,
    Notch = 5,
    BandPass = 6,
    TiltShelf = 7,
    FlatTilt = 8,
    AllPass = 9,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StereoPlacement {
    Stereo = 0,
    Left = 1,
    Right = 2,
    Mid = 3,
    Side = 4,
}

#[derive(Clone, Copy, Debug)]
pub struct EqBandParams {
    pub used: bool,
    pub enabled: bool,
    pub freq_hz: f32,          // 10.0 ..= 30_000.0 Hz (internal: log2(freq_hz))
    pub gain_db: f32,          // -30.0 ..= +30.0 dB
    pub q_ui: f32,             // 0.025 ..= 40.0 (internal: 0.70710678 * q_ui)
    pub shape: FilterShape,
    pub slope_db_oct: f32,     // Continuous 0.0 ..= 96.0 dB/oct, or Brickwall
    pub is_brickwall: bool,
    pub placement: StereoPlacement,
    pub speaker_mask: u64,
    pub dyn_enabled: bool,
    pub dyn_range_db: f32,     // -30.0 ..= +30.0 dB
    pub dyn_auto: bool,
    pub dyn_threshold_db: f32, // -90.0 ..= 0.0 dB
    pub dyn_attack_pct: f32,   // 0.0 ..= 100.0 %
    pub dyn_release_pct: f32,  // 0.0 ..= 100.0 %
    pub ext_sidechain: bool,
    pub sc_filtering: bool,
    pub sc_low_hz: f32,
    pub sc_high_hz: f32,
    pub sc_audition: bool,
    pub spectral_enabled: bool,
    pub spectral_density: f32, // 0.0 ..= 100.0 %
    pub solo: bool,
    pub invert_gain: bool,
}

/// Dual-cascade fractional-slope IIR filter (`EQFilter` @ 0x18010F7E0 & 0x180112260)
pub struct FractionalCascadedBiquad {
    pub cascade_low_order: [[f64; 5]; 8],  // Up to 8 biquads (96 dB/oct)
    pub cascade_high_order: [[f64; 5]; 8],
    pub state_low: [[f64; 2]; 8],
    pub state_high: [[f64; 2]; 8],
    pub num_sections_low: usize,
    pub num_sections_high: usize,
    pub frac_mix: f64, // alpha in [0.0, 1.0)
}

/// Program-dependent double-precision Attack/Release RC follower (`Capacitor` @ 0x18010A9B0)
pub struct ProgramDependentCapacitor {
    pub env_state_db: f64,
    pub slow_crest_state_db: f64,
    pub attack_coeff_base: f64,
    pub release_coeff_base: f64,
}

impl ProgramDependentCapacitor {
    #[inline]
    pub fn step(&mut self, target_gr_db: f64, sample_rate: f64) -> f64 {
        let delta = target_gr_db - self.env_state_db;
        if delta.abs() <= 1e-7 {
            return self.env_state_db;
        }
        // Program-dependent time constant modulation (`0x18010B2C0..0x18010B611`)
        let coeff = if target_gr_db.abs() > self.env_state_db.abs() {
            let crest_factor = (target_gr_db.abs() - self.slow_crest_state_db).max(0.0);
            let tau_att = self.attack_coeff_base / (1.0 + 0.25 * crest_factor);
            (-1.0 / (tau_att * sample_rate)).exp()
        } else {
            let transient_ratio = (self.env_state_db.abs() - self.slow_crest_state_db).max(0.0);
            let tau_rel = self.release_coeff_base / (1.0 + 0.5 * transient_ratio);
            (-1.0 / (tau_rel * sample_rate)).exp()
        };
        self.env_state_db = coeff * self.env_state_db + (1.0 - coeff) * target_gr_db;
        let slow_alpha = (-1.0 / (0.150 * sample_rate)).exp();
        self.slow_crest_state_db =
            slow_alpha * self.slow_crest_state_db + (1.0 - slow_alpha) * self.env_state_db.abs();
        self.env_state_db
    }
}

/// 512-Bin Perceptual Loudness Auto-Gain Calculator (`0x18019BD20` & `0x18019C1B0`)
pub struct PerceptualAutoGain {
    pub log_freq_octaves: [f32; 512],
    pub normalized_weights: [f32; 512],
}

impl PerceptualAutoGain {
    /// Exact 8-knot table extracted from `.rdata` `0x180212988` & `0x1802129A8`
    pub const KNOT_FREQS_HZ: [f32; 8] = [10.0, 20.0, 30.0, 60.0, 90.0, 6250.0, 18000.0, 20000.0];
    pub const KNOT_WEIGHTS: [f32; 8] = [0.0, 0.035, 0.20, 0.85, 1.0, 1.0, 0.40, 0.0];

    pub fn new() -> Self {
        let mut log_freq_octaves = [0.0f32; 512];
        let mut weights = [0.0f32; 512];
        let min_oct = 20.0f32.log2();    // 4.321928 (`0x180212984`)
        let max_oct = 20000.0f32.log2(); // 14.287712 (`0x1802129C8`)
        let step = (max_oct - min_oct) / 511.0;
        let mut sum_w = 0.0f32;

        for i in 0..512 {
            let oct = min_oct + (i as f32) * step;
            let hz = oct.exp2();
            log_freq_octaves[i] = oct;
            let mut w = 0.0f32;
            for k in 0..7 {
                let f0 = Self::KNOT_FREQS_HZ[k];
                let f1 = Self::KNOT_FREQS_HZ[k + 1];
                if hz >= f0 && hz <= f1 {
                    let t = (hz - f0) / (f1 - f0);
                    w = Self::KNOT_WEIGHTS[k] + t * (Self::KNOT_WEIGHTS[k + 1] - Self::KNOT_WEIGHTS[k]);
                    break;
                }
            }
            weights[i] = w;
            sum_w += w;
        }
        let inv_sum = if sum_w > 0.0 { 1.0 / sum_w } else { 1.0 };
        for w in &mut weights {
            *w *= inv_sum;
        }
        Self { log_freq_octaves, normalized_weights: weights }
    }

    #[inline]
    pub fn compute_compensation_linear(&self, combined_mag_sq_512: &[f32; 512]) -> f32 {
        let mut weighted_sum = 0.0f32;
        for i in 0..512 {
            weighted_sum += combined_mag_sq_512[i].sqrt() * self.normalized_weights[i];
        }
        1.0 / weighted_sum.max(1e-9)
    }
}
```

---

### 3.2. Module 2: `soothe3_engine` (Pure-Rust Clean-Room Architecture)

```rust
//! Clean-Room Rust Architecture for oeksound Soothe3 Resonance Suppressor (`soothe3_engine`)
//! Implements Dual-Mode (Soft/Hard) Spectral Crest Suppression, 8-Band Hybrid Series/Parallel
//! Sidechain EQ, Tilt Curves, and Cepstral Minimum-Phase Reconstruction.

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SootheProcessMode {
    Soft = 0, // Level-independent relative spectral prominence detector
    Hard = 1, // Level-dependent dynamic threshold suppressor
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SidechainFilterShape {
    LowCut = 0,
    LowShelf = 1,
    Bell = 2,
    Bandpass = 3, // Routed in PARALLEL when multiple Bandpass bands are enabled
    BandReject = 4,
    Tilt = 5,
    HighShelf = 6,
    HighCut = 7,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeResolutionQuality {
    Normal = 0, // Hop = 512 @ 1x (8x overlap, N_FFT = 4096)
    High = 1,   // Hop = 256 @ 1x (16x overlap, N_FFT = 4096)
    Ultra = 2,  // Hop = 128 @ 1x (32x overlap, N_FFT = 4096)
}

#[derive(Clone, Copy, Debug)]
pub struct TiltPair {
    pub low_mult: f32,  // 0.1 ..= 10.0 (default 1.0, acts below 500 Hz)
    pub high_mult: f32, // 0.1 ..= 10.0 (default 1.0, acts above 2000 Hz)
}

impl TiltPair {
    #[inline]
    pub fn evaluate(&self, freq_hz: f32) -> f32 {
        let r_low = freq_hz / 500.0;
        let s_low = 1.0 / (1.0 + r_low * r_low);
        let r_high = freq_hz / 2000.0;
        let r_high_sq = r_high * r_high;
        let s_high = r_high_sq / (1.0 + r_high_sq);
        self.low_mult.powf(s_low) * self.high_mult.powf(s_high)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Soothe3GlobalParams {
    pub mode: SootheProcessMode,
    pub delta_solo: bool,
    pub depth: f32,        // 0.0 ..= 10.0 (default 2.5)
    pub detail: f32,       // 0.0 ..= 10.0 (default 5.0)
    pub detail_tilt: TiltPair,
    pub attack: f32,       // 0.0 ..= 10.0 (default 2.0)
    pub attack_tilt: TiltPair,
    pub release: f32,      // 0.0 ..= 10.0 (default 2.0)
    pub release_tilt: TiltPair,
    pub max_cut_db: f32,   // -36.0 ..= 0.0 dB (-36.0 = inf)
    pub mix_pct: f32,      // 0.0 ..= 100.0 %
    pub trim_db: f32,      // -24.0 ..= +24.0 dB
    pub inverse: bool,
    pub ext_sidechain: bool,
    pub linear_phase: bool,
    pub low_latency: bool,
    pub quality: TimeResolutionQuality,
    pub mid_side_mode: bool,
    pub channel_link_pct: f32, // 0.0 ..= 100.0 %
    pub channel_balance: f32,  // -1.0 ..= +1.0
    pub auto_gain: bool,
}

pub const STFT_FFT_SIZE: usize = 4096;
pub const STFT_NUM_BINS: usize = (STFT_FFT_SIZE / 2) + 1; // 2049 bins

pub struct Soothe3SpectralCore {
    pub sc_eq_weight_sq: [f32; STFT_NUM_BINS],
    pub detail_per_bin: [f32; STFT_NUM_BINS],
    pub attack_alpha_per_bin: [f32; STFT_NUM_BINS],
    pub release_alpha_per_bin: [f32; STFT_NUM_BINS],
    pub env_gr_db: [[f32; STFT_NUM_BINS]; 2], // Stereo / M-S envelopes
    pub cepstrum_scratch: [f32; STFT_FFT_SIZE],
}

impl Soothe3SpectralCore {
    /// Computes per-bin dynamic resonance suppression in dB for one channel
    pub fn compute_frame_reduction_db(
        &mut self,
        ch: usize,
        sc_power_bins: &[f32; STFT_NUM_BINS],
        baseline_power_bins: &[f32; STFT_NUM_BINS],
        params: &Soothe3GlobalParams,
    ) {
        let eps = 1e-12f32;
        let depth_norm = (params.depth * 0.1).clamp(0.0, 1.0);
        let unlimited_cut = params.max_cut_db <= -35.99;

        for k in 0..STFT_NUM_BINS {
            let p_weighted = sc_power_bins[k] * self.sc_eq_weight_sq[k];
            let p_base = baseline_power_bins[k] * self.sc_eq_weight_sq[k];
            let l_det_db = 10.0 * (p_weighted + eps).log10();
            let l_ref_db = 10.0 * (p_base + eps).log10();
            let prominence_db = (l_det_db - l_ref_db).max(0.0);

            let detail_k = self.detail_per_bin[k].clamp(0.1, 20.0);
            let selectivity_thresh_db = 6.0 / (0.5 + 0.25 * detail_k);

            let raw_cut_db = match params.mode {
                SootheProcessMode::Soft => {
                    let excess = (prominence_db - selectivity_thresh_db).max(0.0);
                    let knee = (excess * excess) / (excess + 3.0);
                    -(depth_norm * 2.4) * knee
                }
                SootheProcessMode::Hard => {
                    let abs_thresh_db = -12.0 - 48.0 * depth_norm;
                    let abs_excess = (l_det_db - abs_thresh_db).max(0.0);
                    let combined = 0.65 * prominence_db + 0.35 * abs_excess;
                    let excess = (combined - 0.7 * selectivity_thresh_db).max(0.0);
                    -(depth_norm * 3.2) * excess
                }
            };

            let clamped_cut_db = if unlimited_cut {
                raw_cut_db.max(-60.0)
            } else {
                raw_cut_db.max(params.max_cut_db)
            };

            let prev_db = self.env_gr_db[ch][k];
            let alpha = if clamped_cut_db < prev_db {
                self.attack_alpha_per_bin[k]
            } else {
                self.release_alpha_per_bin[k]
            };
            self.env_gr_db[ch][k] = alpha * prev_db + (1.0 - alpha) * clamped_cut_db;
        }
    }
}
```
