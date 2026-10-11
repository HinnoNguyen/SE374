# Kế hoạch đồ án

**Đề tài:** Xóa bóng xương sườn trên ảnh X-quang ngực nhằm cải thiện phát hiện bất thường trên bộ dữ liệu VinDr-CXR

**Câu hỏi nghiên cứu:** *Xóa bóng xương có giúp phát hiện bất thường trên X-quang ngực tốt hơn không? Với loại bệnh nào?*

## Tiêu chí chấm của giảng viên → sản phẩm tương ứng

| Tiêu chí | Sản phẩm |
|---|---|
| Hiểu vấn đề | Báo cáo: vì sao xương che tổn thương, cơ chế mô hình, phân tích bệnh nào tốt lên/tệ đi, hạn chế |
| Thực hiện được | Tự train lại toàn bộ mô hình bằng code trong `src/` |
| Có kết quả | Bảng PSNR/SSIM (Phần 1), bảng AP theo từng bệnh A/B/C (Phần 2), hình minh họa |
| Có mã nguồn | Repo này |
| Có checkpoints | `best.pt` của mô hình xóa xương và YOLO, đưa lên GitHub Releases |
| App demo (nếu có) | Thành viên khác phụ trách, dùng `src.bone_suppression.infer.suppress()` + YOLO |

## Thí nghiệm

### Phần 1: Xóa bóng xương (JSRT / BSE-JSRT, 512×512)
| Mô hình | Ghi chú | PSNR | SSIM | MS-SSIM |
|---|---|---|---|---|
| Identity (ảnh gốc so với ảnh đích) | mốc tham chiếu | | | |
| AE (Gusarev-style) | | | | |
| ResNet-BS (Rajaraman 2021) | train trên crop 256 | | | |
| U-Net | | | | |

→ Chọn mô hình tốt nhất (theo SSIM trên tập test) để xóa xương ảnh VinDr.

### Phần 2: Phát hiện bất thường (VinDr-CXR, YOLO11s, 1024px)
| Thí nghiệm | Đầu vào | mAP50 | mAP50-95 |
|---|---|---|---|
| A | ảnh gốc | | |
| B | ảnh đã xóa xương | | |
| C | 3 kênh [gốc, xóa xương, CLAHE] | | |

Cùng cách chia tập (`data/vindr_splits.csv`), cùng siêu tham số, chỉ khác đầu vào.
Phân tích theo từng bệnh bằng `src.detection.compare` (dự đoán: nốt/khối, đông đặc, xơ hóa có thể tốt lên; vôi hóa có thể tệ đi; tim to không đổi).

## Công việc & phân công

| # | Công việc | Phụ trách | Phụ thuộc | Trạng thái |
|---|---|---|---|---|
| 1 | Cài môi trường, chạy `scripts/smoke_test.py --with-yolo` | Cả nhóm | | ⬜ |
| 2 | Tải dữ liệu, kiểm tra cấu trúc thư mục & file meta | | 1 | ⬜ |
| 3 | `prepare_jsrt` + xem previews (căn chỉnh, đảo màu) | | 2 | ⬜ |
| 4 | Train + đánh giá 3 mô hình xóa xương | | 3 | ⬜ |
| 5 | `prepare_vindr` (original) + train YOLO **A** | | 2 | ⬜ |
| 6 | Xóa xương toàn bộ VinDr, kiểm tra bằng mắt ~30 ảnh | | 4 | ⬜ |
| 7 | Train YOLO **B** và **C** | | 5, 6 | ⬜ |
| 8 | Đánh giá, so sánh theo bệnh, vẽ hình | | 7 | ⬜ |
| 9 | App demo | | 4, 5 | ⬜ |
| 10 | Báo cáo + slide | Cả nhóm | 8 | ⬜ |
| 11 | Đưa checkpoints lên GitHub Releases, cập nhật README | | 4, 7 | ⬜ |

Bước 3–4 (Phần 1) và bước 5 (YOLO A) **chạy song song được**.

## Mốc thời gian gợi ý (6 tuần)

| Tuần | Mục tiêu |
|---|---|
| 1 | Bước 1–3, đọc 2 bài báo chính |
| 2 | Bước 4–5 |
| 3 | Bước 6 |
| 4 | Bước 7 |
| 5 | Bước 8–9 |
| 6 | Bước 10–11 |

## Ước lượng tài nguyên (RTX 4060 Ti 16 GB)
- Phần 1: ~200 ảnh train, 512px → vài phút/epoch với U-Net; ResNet-BS nặng hơn nên dùng `--crop 256`.
- Phần 2: ~3.000 ảnh bất thường + 20% ảnh bình thường, YOLO11s @1024, batch 8 → ước tính vài giờ cho mỗi thí nghiệm.
  Nếu thiếu thời gian: `--imgsz 640` hoặc `yolo11n.pt`, nhưng **A/B/C phải cùng cấu hình**.

## Rủi ro & cách xử lý
| Rủi ro | Cách xử lý |
|---|---|
| Lệch miền: JSRT (phim scan Nhật, cũ) ≠ VinDr (X-quang số Việt Nam) | Xem kỹ ảnh xóa xương; thử `--mode direct` và `residual-upsample`; ghi vào hạn chế |
| Không có "đáp án" xóa xương cho VinDr | Đánh giá gián tiếp qua mAP của YOLO (mục tiêu chính) |
| Xóa xương làm mất vôi hóa / gãy xương sườn | Phân tích theo từng bệnh, đây là phát hiện có giá trị |
| BSE-JSRT là ảnh do phần mềm tạo | Nêu trong hạn chế; Rajaraman đã kiểm tra chéo trên ảnh dual-energy thật |
| Hết thời gian train | Giảm `imgsz`/model nhưng giữ A/B/C cùng cấu hình |
