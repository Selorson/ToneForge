// Nội dung hướng dẫn sử dụng nhanh cho từng khu vực của ToneForge Studio.
// Viết riêng cho người mới hoàn toàn — chưa từng dùng FL Studio, Pro-Q 4, Soothe3, Ozone 12 hay UVR5.
//
// Mọi mục dùng chung một khuôn chuẩn từ Co-Writer:
//   title          — tên ngắn gọn của khu vực
//   what           — khu vực này dùng để làm gì (1–2 câu tiếng Việt bình dân, dễ hiểu)
//   steps          — các bước thao tác 1, 2, 3... bám sát đúng tên nút/thanh trượt trên màn hình
//   gotcha         — "Dễ sai:" những lỗi người mới hay gặp và cách tránh
//   next           — nút chuyển nhanh sang bước tiếp theo trong quy trình làm nhạc
//   handbookAnchor — mục tương ứng trong tab "📖 Cẩm nang" để đọc bảng giải thích từng nút

export type WorkspaceTabKey =
  | 'all'
  | 'playlist'
  | 'mixer'
  | 'proq4'
  | 'soothe3'
  | 'ozone12'
  | 'uvr5'
  | 'render'
  | 'handbook';

export type Guide = {
  title: string;
  what: string;
  steps: string[];
  gotcha: string;
  next: { tab: WorkspaceTabKey; label: string };
  handbookAnchor: { sectionId: string; label: string };
};

export const GUIDES: Record<string, Guide> = {
  unified: {
    title: 'Tổng quan quy trình 5 bước (Dành cho người chưa từng dùng FL Studio)',
    what:
      'ToneForge Studio gom 5 phần mềm phòng thu phức tạp nhất thế giới (UVR5 tách nhạc, FL Studio 26 soạn nhạc/trộn âm, FabFilter Pro-Q 4 lọc tần số, Soothe3 khử chói và iZotope Ozone 12 nâng âm lượng AI) vào chung một màn hình duy nhất. Anh không cần tự nối dây hay cài từng plugin rời — mọi thứ đã được nối sẵn và có nút chọn Preset (cấu hình làm sẵn) chỉ việc bấm là nghe hay.',
    steps: [
      'Bấm nút "▶ Play" ở góc trên bên trái để nghe thử tiếng nhạc đang chạy thực tế trên máy, hoặc đổi tốc độ nhanh/chậm ở ô "BPM".',
      'Nếu muốn tách lời/nhạc nền từ một bài hát có sẵn: vào "UVR5 Stem Studio" → chọn cấu hình → bấm "Run Ensemble & Bounce to Mixer 1..4".',
      'Nếu muốn đổi giai điệu hoặc bộ trống/nhạc cụ: ở "Playlist & Piano Roll", chọn "FL26 Sample & Synth Kit" và "FL26 Piano Roll Score (.fsc)".',
      'Làm sạch và làm mượt tiếng bằng Preset có sẵn ở "Pro-Q 4" (chọn Vocals / Mixbus) và "Soothe3" (chọn That Grammy balance preset).',
      'Cuối cùng: bấm "✨ AI Master Assistant (Auto-Match)" ở khung "Ozone 12 Console" để AI tự cân chỉnh âm lượng chuẩn quốc tế, rồi bấm "Export Studio WAV" ở cuối trang để lấy file .wav hoàn chỉnh.',
    ],
    gotcha:
      'Không cần chỉnh tay hàng chục thanh trượt cùng lúc! Người mới chỉ cần chọn đúng "Factory Preset" (cấu hình mẫu của hãng) ở mỗi khung là âm thanh đã đạt 90% chất lượng phòng thu chuyên nghiệp.',
    next: {
      tab: 'handbook',
      label: 'Chưa từng dùng FL Studio? Mở 📖 Cẩm nang hướng dẫn từ A-Z',
    },
    handbookAnchor: {
      sectionId: 'phan-a',
      label: 'Xem sơ đồ vòng lặp 5 bước trong Cẩm nang',
    },
  },

  playlist: {
    title: 'Soạn giai điệu, chọn bộ tiếng FL26 & Vẽ đường cong tự động (Playlist & Piano Roll)',
    what:
      'Đây là nơi tạo ra nhịp trống và giai điệu bài hát (giống khu vực Channel Rack + Piano Roll + Playlist của FL Studio 26). Ngay cả khi anh không biết nhạc lý hay đánh đàn, anh vẫn có thể chọn sẵn các bộ tiếng thật (.wav) và các bản nhạc mẫu (.fsc) để máy tự chơi ngay lập tức.',
    steps: [
      'Chọn bộ nhạc cụ ở ô "FL26 Sample & Synth Kit": thử "808 Studio Kit (Real .wav)" (trống HipHop/Trap trầm ấm), "909 Analog Kit" (trống EDM sôi động), hoặc "Symphonic Strings" (Strings — dàn dây giao hưởng).',
      'Chọn giai điệu mẫu có sẵn ở ô "FL26 Piano Roll Score (.fsc)": thử "FL26 Trance Arp (.fsc)" (Trance), "FL26 Oxygene Progression (.fsc)" (Oxygene) hoặc "FL26 4-Steps Groove (.fsc)" (4 Steps) — các ô nốt nhạc bên phải sẽ tự động xếp hình ngay.',
      'Thêm/Sửa nốt nhạc trên bảng kẻ ô (Piano Roll bên phải): Bấm chuột trái vào ô trống để thêm nốt; kéo giữa nốt để dời vị trí; kéo mép phải nốt để ngân dài ra; bấm chuột phải vào nốt để xoá.',
      'Muốn tiếng đàn/bộ lead luyến mượt từ nốt thấp lên nốt cao (kiểu 808 Glide hay đàn bầu): bật nút "⊿ Slide / Portamento (ON)" trước khi chấm nốt, hoặc tích ô "⊿ Slide Note" ở thanh dưới cùng.',
      'Ở khung "TensionSpline" bên trái: Nhấp đúp vào khung đồ thị để thêm điểm uốn, kéo điểm tròn lên/xuống hoặc chỉnh thanh "Tension" để âm lượng/hiệu ứng tự động vặn to nhỏ theo thời gian.',
    ],
    gotcha:
      'Nốt có ký hiệu "⊿ Slide" không tự phát ra tiếng mới mà dùng để kéo trượt cao độ của nốt đang kêu ngay trước nó. Vì vậy hãy đặt nốt Slide gối lên hoặc nối tiếp một nốt bình thường thì mới nghe thấy tiếng luyến.',
    next: {
      tab: 'proq4',
      label: 'Soạn giai điệu xong → Sang Bước 2: Gọt sạch âm thanh với Pro-Q 4',
    },
    handbookAnchor: {
      sectionId: 'khu-playlist',
      label: 'Giải thích từng nút của Playlist & Piano Roll',
    },
  },

  proq4: {
    title: 'Điêu khắc âm sắc 24 dải tần (FabFilter Pro-Q 4 Equalizer)',
    what:
      'EQ (Equalizer) giống như bộ chỉnh Bass / Treble trên dàn loa gia đình nhưng chi tiết gấp 24 lần: bên trái màn hình là âm Trầm (Bass/Sub), ở giữa là âm Trung (Giọng hát/Nhạc cụ), bên phải là âm Cao (Treble/Độ sáng). Anh dùng khu vực này để cắt bỏ phần âm trầm bị ồm đục và nâng phần giọng hát cho sáng rõ.',
    steps: [
      'Cách nhanh nhất cho người mới: Mở ô "Factory Preset (171 Real .ffp)" ở góc phải trên → chọn "Vocals / Lead Vocal" (cho giọng hát), "Drums / Kick Punch" (cho trống lực hơn) hoặc "Mixbus / Dynamic Smiley Curve ST" (cho cả bài nhạc).',
      'Chỉnh trực tiếp trên màn hình đen: Kéo các nút tròn đánh số 1..7 sang Trái/Phải để chọn vùng âm trầm hay cao (Hz); kéo Lên trên vạch giữa để Tăng (dB) hoặc kéo Xuống dưới vạch giữa để Giảm/Cắt bớt (dB). Nhấp đúp vào vùng trống bất kỳ để tạo thêm điểm chỉnh mới.',
      'Chọn chế độ pha ở góc phải: dùng "ZeroLatency" khi đang vừa gõ nhạc vừa nghe để không bị trễ; dùng "NaturalPhase" khi chỉnh giọng hát/nhạc cụ để tiếng tự nhiên nhất; dùng "LinearPhase" khi gọt toàn bộ bài nhạc (Master) để giữ nguyên độ chuẩn pha.',
      'Nếu một vùng tần số chỉ thỉnh thoảng mới bị chói khi hát to: tích chọn ô "Spectral Dynamics" ở thanh dưới cùng để Pro-Q 4 chỉ nén xuống đúng lúc âm thanh vượt ngưỡng.',
    ],
    gotcha:
      'Đừng kéo các nút tròn lên quá cao (+10 dB đến +20 dB) vì sẽ làm tiếng bị gắt và vỡ (clipping). Quy tắc vàng của kỹ sư phòng thu: "Muốn tiếng sáng hơn, hãy ưu tiên cắt bớt vùng bị đục (kéo nhẹ xuống -2 dB đến -4 dB ở vùng 200–400 Hz) trước khi nâng vùng cao".',
    next: {
      tab: 'soothe3',
      label: 'EQ xong → Sang Bước 3: Khử chói & êm giọng tự động với Soothe3',
    },
    handbookAnchor: {
      sectionId: 'khu-proq4',
      label: 'Giải thích từng nút của FabFilter Pro-Q 4',
    },
  },

  soothe3: {
    title: 'Người dọn dẹp tự động các tần số chói tai, ồm và hú (oeksound Soothe3)',
    what:
      'Khác với EQ thông thường cắt cố định một chỗ làm giọng dễ bị mỏng, Soothe3 là "người dọn dẹp tự động các tần số chói tai, ồm, hú mà không làm mất chất giọng": nó lắng nghe hàng nghìn tần số mỗi giây, phát hiện chỗ nào đang bị chói gắt (tiếng xì "s/x", tiếng hú, tiếng thùng đàn ồm) thì tự động dìm đúng chỗ đó xuống rồi trả lại nguyên vẹn ngay khi hết chói.',
    steps: [
      'Chọn cấu hình mẫu ở ô "Factory Preset (68 Real .preset)": thử ngay "That Grammy balance preset" (cân bằng mượt mà cho cả bài), "SM7 Vocals" (làm mượt giọng thu mic), hoặc "General de-ess" / "De-ess normal" (giảm tiếng xì chói của giọng hát).',
      'Kéo thanh "Depth" (từ 0 đến 10): số càng cao thì Soothe3 càng dọn dẹp mạnh tay hơn (thường để khoảng 3.0 – 5.5 là vừa đẹp, tự nhiên).',
      'Chỉnh "Detail (Sharpness/Selectivity)": kéo cao lên nếu muốn cắt thật hẹp các tiếng hú/rít sắc nhọn; để vừa phải nếu muốn làm êm tổng thể.',
      'Bấm thử nút "Delta Solo" (màu vàng): khi bật nút này, anh sẽ nghe thấy CHỈ NHỮNG TẦN SỐ XẤU/CHÓI đang bị Soothe3 lọc bỏ. Nghe kiểm tra xong nhớ bấm tắt "Delta Solo" để nghe lại bài nhạc hoàn chỉnh!',
    ],
    gotcha:
      'Nhiều người mới bấm bật "Delta Solo" để nghe phần âm thanh rác bị cắt đi, rồi quên tắt nó nên tưởng bài nhạc bị hỏng hoặc nghe mỏng tang như radio cũ! Nhớ tắt "Delta Solo" sau khi kiểm tra xong.',
    next: {
      tab: 'mixer',
      label: 'Khử chói xong → Sang Bước 4: Bàn trộn 502-Track Mixer & Hiệu ứng FL26',
    },
    handbookAnchor: {
      sectionId: 'khu-soothe3',
      label: 'Giải thích từng nút của oeksound Soothe3',
    },
  },

  mixer: {
    title: 'Bàn trộn 502 kênh, Tự động bù trễ 5-Point APDC & Hiệu ứng FL26 (.fst)',
    what:
      'Đây là bàn trộn trung tâm (502-Track Mixer & Patcher) nơi mọi giọng hát, trống, bass và nhạc cụ hội tụ về trước khi ra loa. ToneForge tích hợp sẵn hệ thống "5-Point APDC" tự động căn chỉnh độ trễ của mọi hiệu ứng đến từng mẫu âm thanh (sample) để tiếng trống và nhạc không bao giờ bị lệch nhịp hay ngược pha (out of phase).',
    steps: [
      'Chọn nhóm kênh muốn xem bằng các nút "Tracks 0..31", "32..63"... ở góc phải trên. Kênh #0 là "Master" (kênh tổng của cả bài); các kênh #1..#4 mặc định nhận 4 nhánh âm thanh (Vocals, Drums, Bass, Other).',
      'Bấm vào một ô kênh (ví dụ Track #1) rồi kéo thanh trượt trong ô đó để tăng/giảm âm lượng riêng cho kênh đó. Nhìn chỉ số màu xanh (dB) để biết độ lớn và dòng chữ vàng (smp) để xem số mẫu trễ đã được bù tự động.',
      'Ở khung bên trái phía dưới ("10 FX Insert Slots & Routing"): mở ô "FL26 Native FX Preset (.fst)" để nạp ngay hiệu ứng bản quyền của FL Studio 26 như "Maximus: Clear Master.fst" (Maximus Clear Master), "Emphasis: Pop (heavy).fst" (Emphasis Pop Heavy), "Fruity Soft Clipper.fst" (Fruity Soft Clipper), "Fruity Limiter: Ducking music behind voice.fst" hoặc "LuxeVerb: Shimmer Cathedral.fst".',
      'Nếu muốn gửi âm thanh từ kênh đang chọn sang kênh khác (hoặc làm hiệu ứng nhạc nền tự nhỏ xuống khi có tiếng hát / tiếng trống đập): nhập số kênh vào "Send to Track", tích "Sidechain Only" nếu chỉ gửi tín hiệu kích hoạt, rồi bấm "Apply Route".',
    ],
    gotcha:
      'Khi tích chọn "Sidechain Only" rồi bấm "Apply Route", âm thanh của kênh đó sẽ KHÔNG phát chồng lên kênh đích (tránh bị gấp đôi âm lượng) mà chỉ gửi nhịp đập sang để điều khiển bộ nén (như Fruity Limiter Ducking).',
    next: {
      tab: 'ozone12',
      label: 'Trộn kênh xong → Sang Bước 5: Hoàn thiện AI Mastering với Ozone 12',
    },
    handbookAnchor: {
      sectionId: 'khu-mixer',
      label: 'Giải thích từng nút của 502-Track Mixer & Patcher',
    },
  },

  ozone12: {
    title: 'Bàn Mastering AI & Cân chỉnh lại Giọng hát / Trống thời gian thực (iZotope Ozone 12)',
    what:
      'Mastering là bước "đánh bóng và nâng âm lượng cuối cùng" trước khi xuất bản để bài nhạc của anh nghe to, dày, ấm và đạt chuẩn âm lượng quốc tế như trên Spotify, Apple Music hay YouTube. Đặc biệt, công nghệ BSRNN 120ms ở góc phải cho phép anh vặn to tiếng Hát hoặc giảm tiếng Trống ngay cả trên một file nhạc đã ghép sẵn!',
    steps: [
      'Cách dễ nhất (1 cú nhấp chuột): Bấm nút màu xanh "✨ AI Master Assistant (Auto-Match)". Trí tuệ nhân tạo sẽ tự động đo âm thanh và thiết lập toàn bộ bộ nâng âm lượng (IRC V), độ ấm bóng đèn (12AX7) và độ rộng không gian (Imager).',
      'Hoặc chọn màu âm theo thể loại nhạc: Mở ô "Genre Target Curve (36 Real IZRESEMBED)" (chọn Pop, EDM, HipHop, LoFi, KPopJPop...) và ô "Global Mastering Preset (82 Real XML)" (như "Streaming Master (-14 LUFS)" hoặc "Modern Clarity").',
      'Muốn giọng hát nổi bật hơn hoặc tiếng trống mạnh hơn ngay trong bản nhạc tổng: Nhìn vào khung thứ 4 bên phải "120ms Real-Time BSRNN Master Rebalance" → kéo thanh "Vocals" lên (+1.5 đến +3.0 dB) hoặc chỉnh "Drums", "Bass", "Other" theo ý thích.',
      'Quan sát dãy số đo ở góc trên bên phải: "I: -14.0 LUFS" là độ to chuẩn cho Spotify/YouTube, và "TP L/R" (True Peak) nên luôn nằm từ -1.0 dBTP trở xuống để không bao giờ bị rè tiếng.',
    ],
    gotcha:
      'Đừng kéo thanh "Gain" của IRC V Maximizer quá cao (trên +12 dB) nếu thấy chỉ số "IRC V GR" (Gain Reduction) vượt quá 6 dB, vì ép âm lượng quá mức sẽ làm bài nhạc bị bí, phẳng lì và mất độ nảy tự nhiên của tiếng trống.',
    next: {
      tab: 'render',
      label: 'Mastering hoàn tất → Sang Bước cuối: Xuất file WAV phòng thu (Offline Render)',
    },
    handbookAnchor: {
      sectionId: 'khu-ozone12',
      label: 'Giải thích từng nút của iZotope Ozone 12 Console',
    },
  },

  uvr5: {
    title: 'Phòng tách lời & nhạc nền bằng 3 mô hình AI kết hợp (UVR5 Stem Studio)',
    what:
      'Khu vực này giúp anh đưa bất kỳ bài hát hoàn chỉnh nào vào và dùng 3 mô hình AI mạnh nhất của UVR5 (MDX23C + HTDemucs v4 + VR CascadedASPPNet) chạy cùng lúc để tách riêng thành 4 nhánh âm thanh độc lập: Giọng hát (Vocals), Trống (Drums), Bass và Nhạc cụ khác (Other) — cực kỳ hữu ích khi làm nhạc Karaoke, Remix hoặc lấy Acapella.',
    steps: [
      'Chọn cấu hình AI ở ô "UVR5 Model Profile (143 Real Configs)": chọn "Ensemble: MDX23C + HTDemucs4 + VR-ASPP (3-Model)" để tách sạch nhất cả 4 nhánh, hoặc "Kim_Vocal_2" / "UVR-MDX-NET Karaoke / Vocals" nếu chuyên tách giọng hát và nhạc beat.',
      'Kiểm tra đường dẫn file nhạc đầu vào ở ô "Input File" và giữ nguyên tích chọn "Auto-Bounce 4 Stems onto Mixer Insert Tracks 1..4" để tách xong máy tự động đưa thẳng 4 nhánh vào bàn trộn Mixer.',
      'Bấm nút màu xanh "Run Ensemble & Bounce to Mixer 1..4". Khi chạy xong, khung bên phải sẽ hiện rõ 4 file .wav đã tách (Track 1: Vocals, Track 2: Drums, Track 3: Bass, Track 4: Other).',
    ],
    gotcha:
      'Nếu thấy tiếng nhạc cụ vẫn còn vương nhẹ bóng giọng hát (hoặc ngược lại), hãy bật cả hai ô "MDX23C 2-Pass Phase-Inverted Bias Denoising" và "Phase-Inverted Residual Subtraction" để AI khử nhiễu pha lần 2 sạch bóng.',
    next: {
      tab: 'mixer',
      label: 'Đã tách 4 nhánh vào Kênh 1..4 → Sang 502-Track Mixer để chỉnh từng nhánh',
    },
    handbookAnchor: {
      sectionId: 'khu-uvr5',
      label: 'Giải thích từng nút của UVR5 Stem Studio',
    },
  },

  render: {
    title: 'Xuất file nhạc WAV phòng thu & Kiểm định chuẩn Spotify / YouTube (Offline Render)',
    what:
      'Đây là bước cuối cùng để đóng gói tác phẩm của anh thành file âm thanh `.wav` chất lượng cao lưu vào ổ cứng, đồng thời máy sẽ tự động chấm điểm xem file nhạc đã đạt chuẩn phát hành lên Spotify, Apple Music và YouTube mà không bị các nền tảng đó tự động bóp nhỏ âm lượng hay chưa.',
    steps: [
      'Chọn kiểu xuất ở ô "Render Mode": chọn "Render Master" để xuất 1 file bài nhạc hoàn chỉnh; chọn "Split Mixer Tracks" nếu muốn xuất rời từng kênh; hoặc "UVR5 4-Stem Mastering" để xuất bộ 4 nhánh đã qua xử lý.',
      'Chọn chất lượng file ở ô "Bit Depth & Dither": khuyên dùng "24-Bit PCM + Ozone 12 MBIT+ Noise-Shaping Dither" (chuẩn phát hành nhạc số phổ biến nhất) hoặc "32-Bit IEEE Float" (nếu cần lưu trữ gốc tuyệt đối không vỡ tiếng).',
      'Bấm nút màu xanh "Export Studio WAV & Analyze LUFS/dBTP" ở góc phải trên.',
      'Đọc bảng kết quả ở khung thứ 2 và thứ 3: nếu các nhãn Spotify, Apple Music, YouTube hiện chữ "PASS" màu xanh lá tức là file nhạc đã đạt chuẩn vàng để đăng tải ngay!',
    ],
    gotcha:
      'Nếu nhãn Spotify hoặc YouTube hiện chữ "TRIM" màu vàng kèm số âm (ví dụ -1.8 dB), file nhạc của anh KHÔNG bị lỗi — điều đó chỉ báo hiệu bài nhạc đang hơi to hơn mức -14 LUFS nên khi đăng lên mạng họ sẽ tự vặn nhỏ đi bấy nhiêu dB. Muốn đạt PASS xanh lá tuyệt đối, chỉ cần quay lại Ozone 12 giảm nhẹ thanh "Gain" đi 1–2 dB rồi xuất lại.',
    next: {
      tab: 'handbook',
      label: 'Xem 4 kịch bản thực hành cầm tay chỉ việc trong 📖 Cẩm nang',
    },
    handbookAnchor: {
      sectionId: 'khu-render',
      label: 'Giải thích từng chỉ số của Offline Render & Bounce',
    },
  },
};
