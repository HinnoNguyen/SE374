# Kế hoạch đồ án

**Đề tài:** Bone Suppression in Chest X-rays for Improving Abnormality Detection on the VinDr-CXR Dataset

**Câu hỏi nghiên cứu:** *Xóa bóng xương có giúp phát hiện bất thường trên ảnh X-quang ngực tốt hơn không, và với những loại bệnh nào?*

**Điểm mới (theo nhận xét của giảng viên):** các nghiên cứu trước chứng minh lợi ích của xóa xương trên bài toán **phân loại** (ví dụ Rajaraman et al. 2021, lao phổi);
đồ án kiểm tra trên bài toán **phát hiện vùng bất thường** (abnormal region detection) với 14 loại bất thường của VinDr-CXR.

## Góp ý của giảng viên (10/2026) → việc cần làm

| # | Góp ý | Việc cần làm |
|---|---|---|
| 1 | Tìm related works: đã có nghiên cứu nào *improve detection by bone suppression* chưa, nhất là trên VinDr-CXR | Điền [`related_works.md`](related_works.md) |
| 2 | Train bone suppression tốn thời gian/chi phí → **dùng trước các mô hình có sẵn checkpoint**, áp lên VinDr, train YOLO xem có cải thiện không; so sánh nhiều mô hình có sẵn (gợi ý GL-LCM, MICCAI 2025) | Thí nghiệm B1, B2 (và B3 nếu được) |
| 3 | Nếu tự train: PSNR/SSIM chỉ cần đo trên JSRT/BSE-JSRT. Trên VinDr đánh giá bằng **mAP@50, mAP@[0.5:0.95]** | `evaluate_yolo` + `compare` (cột AP50, AP50_95) |

## Thí nghiệm

Tất cả dùng **cùng cách chia tập** (`data/vindr_splits.csv`) và **cùng cấu hình YOLO**; chỉ khác ảnh đầu vào.

| Mã | Đầu vào YOLO | Mô hình xóa xương | Chạy ở đâu | Bắt buộc? |
|---|---|---|---|---|
| **A** | Ảnh gốc | Không có (baseline) | Local | ✅ |
| **B1** | Ảnh xóa xương | **ResNet-BS** (Rajaraman et al. 2021), đầu vào 256px | Local | ✅ |
| **B2** | Ảnh xóa xương | **AE** (Gusarev et al. 2017, bản của hmchuong), đầu vào 1024px | Local | ✅ |
| B3 | Ảnh xóa xương | **GL-LCM** (Sun et al., MICCAI 2025), checkpoint SZCH-X-Rays | Kaggle miễn phí / local qua WSL2 | Nếu kịp |
| B4 | Ảnh xóa xương | Mô hình tự train (U-Net / ResNet-BS) trên JSRT | Local | Nếu kịp |
| C | 3 kênh [gốc, xóa xương, CLAHE] | Mô hình tốt nhất trong B | Local | Nếu kịp |

### Bảng kết quả cần điền (tập test tự chia từ train của Kaggle)
| Mã | mAP@50 | mAP@[0.5:0.95] | Ghi chú |
|---|---|---|---|
| A | | | |
| B1 | | | |
| B2 | | | |

Phân tích thêm theo từng loại bệnh (`results/detection/compare_*.md`). Dự đoán: nốt/khối, đông đặc, xơ phổi có thể tốt lên;
vôi hóa có thể tệ đi (bị xóa cùng xương); tim to, giãn động mạch chủ ít thay đổi.

## Mô hình có sẵn đã kiểm tra

| Mô hình | Checkpoint | Đã kiểm tra | Lưu ý |
|---|---|---|---|
| ResNet-BS (Rajaraman 2021) | `resnet_bs.h5` (Keras) → `checkpoints/pretrained/rajaraman_resnet_bs.pt` | Bản PyTorch khớp Keras (sai số < 1e-5); tái tạo ảnh mẫu của tác giả với SSIM 0,98 | Repo không ghi giấy phép → chỉ dùng cho nghiên cứu, trích dẫn bài báo |
| AE (hmchuong / Gusarev 2017) | TF1 checkpoint → `checkpoints/pretrained/hmchuong_ae.pt` | Bản PyTorch khớp TensorFlow (sai số < 1e-5); tái tạo ảnh mẫu với SSIM 0,999 | Train trên **ảnh âm bản**: `infer.py` tự đảo màu. MIT license |
| GL-LCM (MICCAI 2025) | Google Drive (VQGAN + UNet, bản SZCH-X-Rays và JSRT) | Chưa chạy | Cần Linux; README ghi GPU ≥ 32 GB; ~8,5 giây/ảnh với 50 bước. Chưa có script cho ảnh tùy ý |
| BS-LDM (IEEE JBHI 2025) | Google Drive | Chưa chạy | 1000 bước, ~85 giây/ảnh → không thực tế cho cả VinDr |

## Nền tảng chạy (đã thống nhất)

- **Toàn bộ train YOLO (A, B1, B2…) chạy trên Kaggle Notebook (T4), cùng cấu hình**, bằng `notebooks/vindr_experiments.ipynb`.
  Không trộn máy: phần cứng khác nhau làm kết quả lệch, khi đó so sánh A với B mất ý nghĩa.
- Máy cá nhân (Windows 4060 Ti, Mac M5 Pro): chạy thử `--limit`, xem ảnh xóa xương, phân tích kết quả, làm app.
- Sau mỗi thí nghiệm: commit `results/` và `data/vindr_splits.csv`; `best.pt` lên GitHub Releases. Train xong không cần chạy lại.

## Chi phí

| Việc | Ước lượng trên RTX 4060 Ti 16 GB |
|---|---|
| Xóa xương B1, B2 cho ~6.500 ảnh | Vài phút đến vài chục phút |
| Train 1 YOLO (yolo11s, imgsz 1024, 50 epoch) | Vài giờ. Muốn nhanh hơn: `--imgsz 640` (nhưng mọi thí nghiệm phải giống nhau) |
| GL-LCM cho ~6.500 ảnh | ~15–30 giờ ở 50 bước. Thử giảm số bước LCM (8–10), chạy thử 20–50 ảnh trước |

GL-LCM: nếu local không đủ VRAM thì dùng **Kaggle Notebook GPU miễn phí** (VinDr đã có sẵn trên Kaggle, không phải tải lên),
chia ảnh bằng `--shard i/n` cho 4 tài khoản của nhóm. Chỉ thuê máy khi cả hai cách trên không được.

## Công việc & phân công

| # | Công việc | Phụ trách | Phụ thuộc | Trạng thái |
|---|---|---|---|---|
| 1 | Cài môi trường, chạy `scripts/smoke_test.py --with-yolo` | Cả nhóm | | ⬜ |
| 2 | Tìm related works, điền `docs/related_works.md` | | | ⬜ |
| 3 | Tải dữ liệu VinDr (Kaggle), kiểm tra file meta kích thước gốc | | 1 | ⬜ |
| 4 | `convert_pretrained.py`, chạy thử `infer --limit 20` bằng B1/B2, **xem ảnh bằng mắt** | | 1 | ⬜ |
| 5 | Kaggle: `run_experiment --experiment A_original` → commit `data/vindr_splits.csv`, `data/vindr_train_sizes.csv`, `results/` | | 3 | ⬜ |
| 6 | Kaggle: `B1_rajaraman`, `B2_hmchuong` (chạy được song song trên 2 tài khoản, sau khi split đã commit) | | 4, 5 | ⬜ |
| 7 | Đánh giá, so sánh mAP@50 / mAP@[0.5:0.95] theo bệnh, vẽ hình | | 6 | ⬜ |
| 8 | (Tùy chọn) GL-LCM → B3 | | 6 | ⬜ |
| 9 | (Tùy chọn) Tự train xóa xương → B4; PSNR/SSIM trên JSRT | | 6 | ⬜ |
| 10 | App demo | | 6 | ⬜ |
| 11 | Báo cáo + slide; đưa checkpoints lên GitHub Releases | Cả nhóm | 7 | ⬜ |

## Rủi ro & cách xử lý
| Rủi ro | Cách xử lý |
|---|---|
| Lệch miền: mô hình có sẵn học trên JSRT (phim scan Nhật) / SZCH, VinDr là X-quang số Việt Nam | Xem ảnh bằng mắt trước khi chạy hết; so sánh nhiều mô hình chính là để đo điều này |
| Xóa xương làm mất vôi hóa / gãy xương sườn | Phân tích theo từng bệnh, đây là phát hiện có giá trị |
| Tập test tự chia từ train của Kaggle (nhãn test Kaggle bị ẩn) | Ghi rõ trong báo cáo; mọi thí nghiệm dùng chung một cách chia |
| Hết thời gian train | Giảm `imgsz`/model nhưng giữ mọi thí nghiệm cùng cấu hình |
