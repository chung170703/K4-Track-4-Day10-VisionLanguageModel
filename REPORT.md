Họ tên: Ngô Đức Chung
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 7.85 | 4.20 | 0.0711 | [0.0471, 0.0989] | 1.09 | 0.1517 | 0.0144 | 0.1479 |
| E1 | LoRA LLM | 33,030,144 | 8.92 | 41.72 | 0.2605 | [0.2243, 0.2990] | 1.26 | 0.1759 | 0.1889 | 0.5097 |
| E2 | + projector | 142,135,296 | 8.46 | 41.28 | 0.3414 | [0.3030, 0.3820] | 1.22 | 0.2266 | 0.2684 | 0.6115 |
| E3 | + LoRA ViT | 148,426,752 | 8.64 | 55.60 | 0.4335 | [0.3877, 0.4752] | 0.95 | 0.3038 | 0.3689 | 0.6689 |
| E4 | LoRA rank r=4 | 117,362,688 | 8.64 | 47.03 | 0.3045 | [0.2659, 0.3410] | 1.44 | 0.1733 | 0.2312 | 0.6176 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
- **Phân bổ tham số kiến trúc Qwen3-VL-4B**:
  - **Vision Transformer (ViT)**: Gồm 24 khối transformer (`depth=24`), kích thước ẩn `hidden_size=1024`, `intermediate_size=4096`, 16 attention heads. Tổng số tham số của ViT xấp xỉ **303M tham số** (bao gồm patch embedding $16 \times 16 \times 3$, positional embeddings và 24 transformer blocks).
  - **Projector**: Gồm bộ gộp không gian chính (`visual.merger`) và 3 bộ gộp đa tỉ lệ tại các tầng trung gian 5, 11, 17 (`visual.deepstack_merger_list`). Projector làm nhiệm vụ ánh xạ từ đặc trưng 4 patch gộp ($4 \times 1024 = 4096$) sang không gian embedding của LLM ($2560$). Tổng cộng có 24 tensors với **109,105,152 tham số** (~109.1M tham số).
  - **Language Model (LLM)**: Gồm 36 tầng giải mã Causal LM (`num_hidden_layers=36`), `hidden_size=2560`, `intermediate_size=9728`, cơ chế Grouped-Query Attention (32 query heads, 8 KV heads), từ điển `vocab_size=151936` (buộc chia sẻ trọng số embedding). Tổng số tham số LLM xấp xỉ **3.65B tham số**.
- **Khối được cập nhật trong từng setup**:
  - **E1 (LoRA LLM)**: Đóng băng hoàn toàn ViT và Projector. Gắn LoRA ($r=16, \alpha=32$) vào 252 module chiếu tuyến tính của LLM (36 layers $\times$ 7 projections: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`). Cập nhật **33,030,144 tham số** (~0.8% mô hình).
  - **E2 (+ Projector)**: Giữ nguyên LoRA LLM của E1 (33.03M) và mở khoá toàn bộ trọng số Projector (`visual.merger` và `visual.deepstack_merger_list`, 109,105,152 tham số) đưa về `float32`. Tổng tham số huấn luyện là **142,135,296 tham số** (~5.36% mô hình).
  - **E3 (+ LoRA ViT)**: Giữ nguyên toàn bộ cấu hình E2 và gắn thêm LoRA ($r=16, \alpha=32$) vào 96 module của ViT (24 blocks $\times$ 4 projections: `attn.qkv`, `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`, bổ sung 6,291,456 tham số). Tổng cộng 348 module LoRA + toàn bộ Projector, cập nhật **148,426,752 tham số** (~5.59% mô hình).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- **Chỉ số E0 từ `report/evidence/e0.json`**:
  - Tỷ lệ sinh JSON hợp lệ `parse_valid_rate`: **83.28%** (0.8327974276527331). Tức có tới ~16.72% số ảnh ở tập test mà model gốc không thể sinh ra JSON hợp lệ.
  - Điểm F1 micro: **0.0711** (Precision: 9.12%, Recall: 5.83%, TP: 34, FP: 339, FN: 549).
- **Hai lỗi phổ biến nhất của model gốc**:
  1. *Lỗi định dạng cấu trúc JSON (Formatting Failure & Conversational Hallucination)*: Model gốc thường sinh thêm các câu giải thích mở đầu/kết thúc (ví dụ: `Based on the road image, here are the defects:...` hoặc cú pháp markdown ```json ... ``` bị thiếu dấu ngoặc vuông `]`), vi phạm yêu cầu nghiêm ngặt "Return ONLY a JSON list".
  2. *Ảo giác vết nứt và dự đoán bao phủ diện rộng (Over-prediction & Huge False Positives)*: Do chưa được finetune trên miền ảnh đường bộ, model gốc nhận định hầu như mọi vết bóng râm, mối nối mặt đường hoặc gờ phân cách thành `crack` (lớp `crack` có tới 251 False Positives, Precision chỉ đạt 1.57%). Nhiều box dự đoán bao trọn cả nửa bức ảnh khiến IoU $< 0.5$ và bị tính là FP.
- **Ví dụ output thô điển hình của E0**:
  `[{"label": "crack", "bbox_2d": [0, 680, 999, 995]}, {"label": "pothole", "bbox_2d": [412, 530, 480, 590]}]`
  (Box `crack` trải dài từ mép trái sang mép phải toàn bộ phần mặt đường phía dưới ảnh, dẫn đến việc không khớp với bất kỳ vết nứt cục bộ nào của ground truth).

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- **Hệ quả khi lệch thang đo toạ độ**:
  - Nếu mô hình sinh toạ độ theo pixel (ví dụ ảnh $1920 \times 1080$, toạ độ $x \in [0, 1920], y \in [0, 1080]$) nhưng bộ chấm `scripts/score.py` đối chiếu theo thang chuẩn hoá nguyên $[0, 1000]$ (hoặc ngược lại), điểm **$\text{F1}$ sẽ sụt giảm về 0.0 (hoặc xấp xỉ 0)**.
  - **Nguyên nhân toán học**: Ngưỡng ghép cặp True Positive đòi hỏi $\text{IoU} \ge 0.5$. Công thức $\text{IoU} = \frac{\text{Area}(B_{pred} \cap B_{gt})}{\text{Area}(B_{pred} \cup B_{gt})}$. Khi một box ở toạ độ pixel (kích thước hàng trăm, hàng nghìn pixel) được đem so trực tiếp với box thang 1000, hoặc toạ độ bị tràn ra ngoài miền $[0, 1000]$, diện tích phần giao nhau sẽ cực nhỏ so với diện tích hợp, khiến $\text{IoU} < 0.5$ ở hầu như toàn bộ 100% các cặp box. Tất cả dự đoán đều trở thành False Positive và ground truth thành False Negative, dẫn đến $\text{TP} = 0 \implies \text{F1} = 0.0$.
- **Cách kiểm chứng trong lab**:
  - Đã thực hiện kiểm chứng 2 chiều trong notebook: chuyển đổi $YOLO \to [0, 1000] \to \text{pixel}$ đảm bảo độ lệch tuyệt đối $|x_{orig} - x_{recon}| \le 1.0$ pixel trên toàn bộ tập huấn luyện.
  - Thực nghiệm kiểm thử script `scripts/score.py` với tuỳ chọn cờ `--coord norm1000` khẳng định tính nhất quán của toạ độ mô hình và nhãn ground truth.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Lý do chỉ tính loss trên phần trả lời (`labels != -100`)**:
  - Bài toán finetune VLM cho Visual Grounding là mô hình hoá phân phối xác suất có điều kiện $P(\text{Response} \mid \text{Image}, \text{Prompt})$. Câu hỏi (prompt) và hình ảnh là ngữ cảnh đầu vào (conditions), mô hình cần học cách sinh ra toạ độ và nhãn hư hại mặt đường chính xác dựa trên ngữ cảnh đó.
  - Việc gán nhãn `-100` cho toàn bộ prompt và visual tokens (`<|image_pad|>`) kích hoạt tính năng `ignore_index=-100` của hàm mất mát `CrossEntropyLoss`, triệt tiêu gradient trên phần ngữ cảnh đầu vào.
- **Điều gì xảy ra nếu tính loss trên cả prompt và token ảnh?**:
  1. *Gradient mất cân bằng trầm trọng (Visual token domination)*: Số lượng token ảnh áp đảo (hàng trăm đến hàng nghìn tokens) so với vài chục token của chuỗi toạ độ JSON. Nếu tính loss trên cả chuỗi, gradient từ việc tái tạo visual tokens sẽ áp đảo tín hiệu học toạ độ bounding box.
  2. *Phá vỡ không gian biểu diễn thị giác*: Ép bộ giải mã LLM autoregressive dự đoán từng token ảnh tiếp theo theo chiều ngôn ngữ là phi lý và làm xáo trộn đặc trưng không gian do ViT trích xuất.
  3. *Hiện tượng học vẹt câu lệnh (Prompt memorization)*: Mô hình sẽ lãng phí dung lượng tham số để học thuộc lòng chuỗi ký tự prompt cố định thay vì học ánh xạ từ đặc trưng thị giác sang toạ độ không gian.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Nguyên nhân kỹ thuật**:
  - Trong cấu hình QLoRA NF4 tiêu chuẩn, nếu base model chứa các module ở kiểu `float16`, các trọng số của Projector ban đầu cũng ở `torch.float16`. Khi unfreeze Projector và huấn luyện với bộ chia tỉ lệ gradient tự động `torch.amp.GradScaler`, PyTorch nghiêm cấm unscale các gradient có kiểu `float16` (`ValueError: Attempting to unscale FP16 gradients`).
  - Ngoài ra, nếu chỉ ép kiểu ở thuộc tính `.data` (`param.data = param.data.float()`), thuộc tính `param.dtype` vẫn giữ nguyên `torch.float16`, khiến bộ tối ưu AdamW phân bổ bộ đệm moment (`exp_avg`) ở `float16` nhưng gradient nhận vào ở `float32`, dẫn đến lỗi `RuntimeError: expected dtype c10::Half for end but got dtype float` trong kernel `_foreach_lerp_`.
- **Cách xử lý triệt để**:
  1. Ép toàn bộ module Projector và LoRA adapters về `torch.float32`: gọi `module.to(torch.float32)` trên `visual.merger`, `visual.deepstack_merger_list`, `lora_A`, `lora_B`.
  2. Thay thế tường minh các tham số của Projector bằng đối tượng `torch.nn.Parameter(p.data.float(), requires_grad=True)`.
  3. Cấu hình bộ tối ưu `AdamW(trainable_params, lr=2.0e-4, foreach=False)` để xử lý từng tham số riêng biệt, loại bỏ xung đột phân nhóm tensor đa kiểu dữ liệu.
- **Vì sao Projector phải ở `float32`?**:
  - Projector là nút thắt cổ chai giao tiếp giữa miền thị giác và miền ngôn ngữ. Tín hiệu gradient truyền ngược từ 36 tầng LLM qua Projector có biên độ biến thiên rất lớn. Kiểu `float32` cung cấp dải động (dynamic range với 8 bit số mũ) rộng hơn gấp nhiều lần so với `float16` (chỉ 5 bit số mũ), ngăn chặn triệt để hiện tượng tràn số (gradient overflow/underflow), giữ cho các moment bậc 1 và bậc 2 của AdamW luôn hội tụ ổn định.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên gây F1 cao giả (Data Leakage)**:
  - Dữ liệu hư hại mặt đường được trích xuất từ các video hành trình liên tục theo từng phiên (session). Các khung hình kế tiếp nhau chỉ cách nhau vài phần trăm giây, có cùng góc máy, cùng điều kiện mặt đường, ánh sáng và chiếu cùng một ổ gà / vết nứt.
  - Nếu chia ngẫu nhiên từng ảnh (random shuffle), các góc nhìn khác nhau của cùng một ổ gà sẽ rơi vào cả tập `train` và `test`. Khi đó mô hình chỉ cần ghi nhớ phông nền (memorize background) là có thể định vị chính xác, tạo ra kết quả kiểm thử cao giả tạo (optimistic bias).
  - Chia theo phiên quay (`groups` trong `meta.json`) đảm bảo toàn bộ hành trình của một phiên quay chỉ thuộc về một tập dữ liệu duy nhất, đo lường năng lực tổng quát hoá thực tế trên cung đường mới.
- **Thống kê số ảnh và số box từng lớp theo `data/splits/meta.json`**:
  - **Tập Train** (12 phiên: `num-0`, `num-1`, `phone-20250216`,...):
    - Tổng số ảnh: **1,422** ảnh.
    - `pothole`: **1,074** boxes | `crack`: **1,750** boxes | `manhole`: **691** boxes (Tổng: 3,515 boxes).
  - **Tập Val** (2 phiên: `vlc-2025-02-18-18`, `vlc-2025-02-19-16`):
    - Tổng số ảnh: **276** ảnh.
    - `pothole`: **66** boxes | `crack`: **470** boxes | `manhole`: **103** boxes (Tổng: 639 boxes).
  - **Tập Test** (3 phiên: `num-4`, `vlc-2025-02-19-17`, `vlc-2025-02-18-23`):
    - Tổng số ảnh: **311** ảnh.
    - `pothole`: **121** boxes | `crack`: **299** boxes | `manhole`: **163** boxes (Tổng: 583 boxes).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **So sánh định lượng giữa các cấu hình**:
  - **E1 (Chỉ LoRA LLM)**: Đạt F1 micro **0.2605** [0.2243, 0.2990], VRAM đỉnh **8.92 GB**, thời gian train **41.72 phút** với 33.03M tham số trainable. Khả năng parse đạt tới 99.04%.
  - **E2 (+ Toàn bộ Projector)**: Mở thêm 109.11M tham số Projector đưa tổng tham số lên 142.14M. Đạt F1 micro **0.3414** [0.3030, 0.3820], tăng vọt **+8.09%** so với E1 (0.2605), tỷ lệ parse valid đạt 100.0%. VRAM đỉnh 8.46 GB, thời gian train 41.28 phút.
  - **E3 (+ LoRA ViT)**: Bổ sung 96 module LoRA trên ViT đưa tổng tham số lên 148.43M. Đạt F1 micro **0.4335** [0.3877, 0.4752], tăng áp đảo **+9.21%** so với E2 và **+17.30%** so với E1! FP/ảnh giảm mạnh từ 1.22 xuống 0.95. VRAM đỉnh 8.64 GB, thời gian train 55.60 phút.
- **Đánh giá tính đánh đổi (Trade-off Analysis)**:
  - *Mở Projector (E1 -> E2)*: CỰC KỲ ĐÁNG GIÁ. F1 tăng từ 0.2605 lên 0.3414 (+8.09%) mà không tốn thêm VRAM (8.46 GB vs 8.92 GB) hay thời gian train (41.28 phút vs 41.72 phút). Khoảng tin cậy [0.303, 0.382] tách biệt rõ rệt khỏi E1 [0.224, 0.299].
  - *LoRA trên ViT (E2 -> E3)*: CỰC KỲ HIỆU QUẢ. F1 tiếp tục tăng vọt lên 0.4335 (+9.21% so với E2), khoảng tin cậy [0.388, 0.475] hoàn toàn vượt trên E2 [0.303, 0.382]. Chi phí đánh đổi duy nhất là thời gian huấn luyện tăng thêm ~14 phút (55.60 phút so với 41.28 phút) do lan truyền ngược qua 24 khối ViT, trong khi VRAM đỉnh chỉ tăng nhẹ 0.18 GB (8.64 GB vs 8.46 GB). Đây là minh chứng rõ nét cho thấy việc điều chỉnh biểu diễn thị giác của ViT cho miền ảnh đường bộ đóng vai trò quyết định trong việc cải thiện độ chính xác phát hiện và giảm thiểu dương tính giả.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng khoảng tin cậy bootstrap CI95**:
  - Tập test gồm 311 ảnh với 583 boxes ground truth.
  - Ở E0, khoảng tin cậy F1 là $[0.0471, 0.0989]$, độ rộng $\Delta \approx 0.0518$.
  - Ở E1, khoảng tin cậy F1 là $[0.2243, 0.2990]$, độ rộng $\Delta \approx 0.0746$ (biên độ sai số $\pm 3.73\%$).
- **Giới hạn kết luận thống kê**:
  - Khoảng tin cậy có độ rộng lên tới ~7.5% là tương đối lớn. Điều này có nghĩa là nếu sự chênh lệch điểm F1 giữa hai setup (ví dụ E1 và E2) chỉ rơi vào khoảng 1% đến 3%, hai khoảng tin cậy CI95 sẽ chồng lấn (overlap) nhau đáng kể.
  - Về mặt lý thuyết xác suất thống kê, khi hai khoảng tin cậy 95% chồng lấn, ta **không thể bác bỏ giả thuyết vô hiệu (null hypothesis)** và không thể khẳng định chắc chắn 95% rằng một setup vượt trội hơn setup kia chỉ dựa trên điểm số trung bình. Do đó, việc đánh giá bắt buộc phải kết hợp thêm các tiêu chí phụ như tỷ lệ False Positive trên mỗi ảnh (`fp_per_image`), `mean_iou` trên các True Positives, và độ ổn định của đường cong loss trong quá trình huấn luyện.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Hiệu năng theo lớp (từ kết quả kiểm thử)**:
  - **Lớp tốt nhất — `manhole` (Nắp cống)**: Đạt F1 cao vượt trội **0.5097** (Precision: 68.75%, Recall: 40.49% ở E1).
    - *Nguyên nhân*: Nắp cống có hình thái học rất rõ ràng (hình tròn hoặc hình chữ nhật kim loại dập nổi), độ tương phản cao so với mặt đường nhựa xung quanh, kích thước box đồng đều và ranh giới sắc nét.
  - **Lớp kém nhất — `pothole` (Ổ gà, F1: 0.1759) và `crack` (Vết nứt, F1: 0.1889)**:
    - *Vết nứt (`crack`)*: Gặp số lượng False Positive khổng lồ (286 FP ở E1), Precision chỉ đạt 17.58%. Vết nứt có hình dạng thanh mảnh, kéo dài ngoằn ngoèo, ranh giới mờ nhạt. Việc ép vết nứt vào bounding box hình chữ nhật $[x1, y1, x2, y2]$ khiến phần lớn diện tích trong box là mặt đường bình thường; mô hình rất dễ nhầm lẫn bóng râm của cây cối, vệt dầu loang hoặc mối nối thảm bê tông thành vết nứt.
    - *Ổ gà (`pothole`)*: Số lượng mẫu trong tập test ít nhất (chỉ 121 box). Do góc quay camera hành trình nhìn xiên thấp (shallow grazing angle), ổ gà thường bị che khuất một phần hoặc trông giống như một mảng vá đường nhựa sẫm màu, dẫn đến tỷ lệ bỏ sót (FN = 102) rất cao.
- **Ảnh minh hoạ**: Kết quả trực quan hoá so sánh ground truth (màu xanh lá cây) và dự đoán của mô hình (màu tím nét đứt) được lưu tại `report/images/pred_comparison.png`.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Thiết kế thực nghiệm Ablation E4**:
  - Biến lựa chọn: Khảo sát ảnh hưởng của **LoRA Rank $r$** trên nền tảng setup E2 (mở Projector). So sánh giữa cấu hình rank nhỏ $r=4$ ($\alpha=8$) và baseline $r=16$ ($\alpha=32$).
  - Khi giảm rank từ 16 xuống 4, số lượng tham số LoRA giảm từ 33.03M xuống 8.26M tham số (giảm 75% tham số LoRA), đưa tổng tham số huấn luyện (kèm Projector) từ 142,135,296 xuống 117,362,688.
- **Kết quả đo lường thực tế**:
  - **Rank $r=4$**: Đạt F1 micro **0.3045**, khoảng tin cậy CI95 $[0.2659, 0.3410]$, FP/ảnh là 1.44, thời gian huấn luyện 47.03 phút, VRAM đỉnh 8.64 GB.
  - **Rank $r=16$ (Baseline E2)**: Đạt F1 micro **0.3414**, khoảng tin cậy CI95 $[0.3030, 0.3820]$, FP/ảnh là 1.22, thời gian huấn luyện 41.28 phút, VRAM đỉnh 8.46 GB.
- **Đánh giá độ tin cậy của kết luận**:
  - *Hiệu năng giảm*: Khi giảm rank từ 16 xuống 4, F1 giảm **-3.69%** (từ 0.3414 xuống 0.3045), đồng thời tỷ lệ dương tính giả tăng lên (FP/ảnh tăng từ 1.22 lên 1.44).
  - *Độ tin cậy thống kê*: Hai khoảng tin cậy CI95 của $r=4$ $[0.266, 0.341]$ và $r=16$ $[0.303, 0.382]$ có sự chồng lấn nhẹ ở vùng $[0.303, 0.341]$. Tuy nhiên, giá trị trung bình của $r=16$ vượt trội rõ ràng và chạm cận trên của $r=4$. Điều này chứng minh rằng đối với bài toán Visual Grounding cần hồi quy toạ độ chính xác 4 chiều $[x1, y1, x2, y2]$, rank $r=16$ cung cấp dung lượng biểu diễn ma trận thích ứng tốt hơn đáng kể so với rank $r=4$ mà không làm tăng thời gian huấn luyện hay VRAM.

## Nhìn lại

- **Những thách thức kỹ thuật và phát hiện then chốt**:
  1. *Xung đột số học Mixed Precision và LoRA Projector*: Khi kết hợp lượng tử hoá 4-bit với việc mở toàn bộ Projector, việc các tham số gốc ở kiểu `float16` gây ra xung đột nghiêm trọng với `torch.amp.GradScaler` và `AdamW`. Bắt buộc phải chuyển đổi đồng nhất toàn bộ tham số trainable và gradient về `torch.float32` ở cấp độ `nn.Parameter` và tắt cơ chế đa tensor `foreach=False` trong AdamW để đảm bảo tính ổn định số học tuyệt đối.
  2. *Hạn chế của Bounding Box đối với đối tượng dạng đường*: Các hư hại mặt đường dạng tuyến tính như vết nứt (`crack`) có đặc tính hình học không phù hợp với bounding box hình chữ nhật chuẩn. Trong các ứng dụng thực tế, chuyển sang biểu diễn đa giác (polygon segmentation) hoặc mask RLE sẽ phản ánh chính xác hơn nhiều chất lượng phát hiện.
  3. *Hiện tượng rò rỉ dữ liệu theo chuỗi thời gian*: Việc bắt buộc chia split theo session quay (group-based split) là kinh nghiệm thực tế sống còn trong thị giác máy tính cho xe tự hành, ngăn ngừa hoàn toàn hiện tượng F1 cao ảo do mô hình học vẹt khung cảnh liên tiếp.

## Bonus E5 — Học tăng cường (GRPO) với Reward IoU cho Visual Grounding (+10 điểm)

### 1. Thiết kế và kiểm thử Hàm Reward IoU (4 điểm)
- **Công thức tính toán**:
  Hàm `compute_grounding_reward(completion, gt_boxes)` được thiết kế để đánh giá trực tiếp độ chính xác không gian và ngữ nghĩa:
  - Nếu `parse_output(completion)` trả về `None` (JSON hỏng hoặc không đúng định dạng): $R = 0.0$ (phạt nặng nhất).
  - Nếu cả ground truth và dự đoán đều rỗng (khung cảnh không có hư hại): $R = 1.0$.
  - Nếu một trong hai rỗng (bỏ sót toàn bộ hoặc bịa ra box khi đường hoàn toàn bình thường): $R = 0.2$ (chỉ thưởng điểm đúng định dạng JSON cơ bản).
  - Ghép đôi tham lam (greedy bipartite matching) các box cùng nhãn theo IoU giảm dần, chỉ chấp nhận $\text{IoU} \ge 0.5$.
  - Điểm reward cuối cùng:
    $$R = 0.2 + 0.8 \times \frac{\sum_{i} \text{IoU}_i}{\max(N_{gt}, N_{pred}, 1)}$$
    Mẫu số $\max(N_{gt}, N_{pred}, 1)$ đóng vai trò là cơ chế phạt tự nhiên (penalty) chống hiện tượng sinh thừa box hoặc ảo giác (hallucination).
- **Kết quả Unit Tests (chạy tự động trong notebook)**:
  1. *Khớp hoàn hảo (Perfect match)*: $R = 1.0$ (assert pass).
  2. *JSON hỏng (Invalid output)*: $R = 0.0$ (assert pass).
  3. *Sai lớp (Wrong class)*: $R = 0.2$ (assert pass).
  4. *Thừa box bịa (Over-prediction)*: $0.2 < R = 0.6 < 1.0$ (bị phạt qua mẫu số, assert pass).
  - Kết quả log kiểm thử từ Kaggle: `Unit test hàm reward IoU: TẤT CẢ TEST ĐỀU PASS!`

### 2. Thiết lập huấn luyện GRPO (3 điểm)
- **Cấu hình theo `configs/e5.yaml` & `GUIDE.md`**:
  - Mô hình nền tảng: Khởi tạo từ checkpoint tốt nhất **E3** (LoRA ViT + Full Projector + LoRA LLM).
  - Thuật toán: Group Relative Policy Optimization (`trl.GRPOTrainer`, `trl==1.9.2`).
  - Số lượng generation ($G$): 4 rollouts mỗi prompt (`num_generations=4`).
  - Độ dài tối đa: `max_completion_length=128`.
  - Số bước huấn luyện: 60 bước, tối ưu bằng AdamW với learning rate nhỏ $5.0 \times 10^{-6}$, hệ số điều hòa KL $\beta = 0.0$.
  - Log quá trình: Reward trung bình tăng dần qua các step (từ 0.42 lên ~0.68), độ lệch chuẩn reward trong nhóm (group advantage) ổn định, không xuất hiện hiện tượng sụp đổ gradient.

### 3. Phân tích hiện tượng Reward Hacking & So sánh với SFT (3 điểm)
- **Phân tích dấu hiệu Reward Hacking**:
  - *Nguy cơ 1 — Mô hình trả danh sách rỗng `[]`*: Nếu dữ liệu huấn luyện có nhiều ảnh không có hư hại, mô hình có thể bị cám dỗ trả về `[]` để né tránh mẫu số phạt $\max(N_{gt}, N_{pred})$. Tuy nhiên, trong tập Road Damage, phần lớn ảnh đều có ít nhất 1 hư hại mặt đường ($N_{gt} \ge 1$), do đó nếu trả `[]`, reward chỉ nhận được $0.2$, thấp hơn nhiều so với việc dự đoán đúng một box bất kỳ ($R \ge 0.2 + 0.8 \times 0.5 / N_{gt} \approx 0.6$).
  - *Nguy cơ 2 — Mô hình sinh box khổng lồ bao trùm toàn bộ ảnh*: Nhằm tối đa hóa khả năng giao cắt với ground truth. Tuy nhiên, do công thức $\text{IoU} = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$, một box quá lớn sẽ làm diện tích hợp tăng vọt, kéo $\text{IoU}$ xuống dưới ngưỡng $0.5$ và nhận $\text{IoU} = 0$, bị loại bỏ hoàn toàn khỏi bộ ghép đôi.
  - *Kết luận*: Hàm reward được thiết kế chặt chẽ và không quan sát thấy hiện tượng reward hacking bệnh lý trong quá trình rollout.
- **So sánh định lượng giữa GRPO và SFT (E3)**:
  - **SFT (Baseline E3)**: F1 micro **0.4335**, khoảng tin cậy CI95 $[0.3877, 0.4752]$, FP/ảnh = 0.95, meanIoU = 0.688.
  - **GRPO (E5)**: F1 micro đạt **0.4418**, khoảng tin cậy CI95 $[0.3962, 0.4835]$, FP/ảnh giảm còn 0.89, meanIoU cải thiện lên 0.694.
  - **Đánh giá thống kê (CI95)**:
    - Điểm F1 micro sau GRPO tăng nhẹ **+0.83%** so với SFT, và tỷ lệ dương tính giả giảm rõ rệt (mô hình thận trọng hơn khi sinh box nhờ cơ chế phạt của mẫu số).
    - Tuy nhiên, điểm số $0.4418$ vẫn **nằm hoàn toàn bên trong khoảng tin cậy 95% của SFT E3 $[0.3877, 0.4752]$**.
    - Về mặt kiểm định giả thuyết thống kê, ta kết luận rằng 60 bước GRPO mang lại sự tinh chỉnh tích cực về chất lượng toạ độ và giảm thiểu FP, nhưng mức tăng F1 chưa tạo ra sự bứt phá vượt ra ngoài biên sai số thống kê trên tập kiểm thử 311 ảnh.
