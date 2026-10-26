# Related works

Mục tiêu (góp ý số 1 của giảng viên): kiểm tra xem **đã có nghiên cứu nào dùng xóa bóng xương để cải thiện *phát hiện* (detection) chưa**,
và nắm các kết quả detection hiện có trên **VinDr-CXR**, để định vị đúng điểm mới của đồ án.

## Từ khóa tìm kiếm (Google Scholar, Semantic Scholar, arXiv, IEEE Xplore)
- `"bone suppression" chest X-ray detection`
- `"bone suppression" "object detection"` / `"bone suppression" YOLO`
- `"bone suppression" nodule detection`
- `"rib suppression" lesion detection`
- `"soft tissue" chest radiograph deep learning detection`
- `"VinDr-CXR" detection` / `"VinBigData" chest x-ray abnormalities detection`
- `"bone suppression" "VinDr"`

Mẹo: mở các bài bone suppression mới (GL-LCM, BS-LDM) và xem mục *Related work* + danh sách bài **trích dẫn chúng** (Google Scholar → "Cited by").

## A. Xóa bóng xương + bài toán chẩn đoán phía sau
| Bài | Năm / nơi đăng | Bài toán sau xóa xương | Dữ liệu | Kết quả chính | Khác đồ án ở đâu |
|---|---|---|---|---|---|
| Rajaraman et al. | 2021, Diagnostics | Phân loại lao | Shenzhen, Montgomery | AUC tăng có ý nghĩa thống kê | Classification, không phải detection |
| Freedman et al. *(cần kiểm tra lại trích dẫn)* | 2011, Radiology | Bác sĩ đọc tìm nốt phổi | — | Phần mềm xóa xương giúp bác sĩ phát hiện nốt tốt hơn | Người đọc, không phải mô hình tự động |
| | | | | | |

## B. Các mô hình xóa bóng xương (có checkpoint hay không)
| Bài | Năm / nơi đăng | Phương pháp | Checkpoint | Dùng trong đồ án |
|---|---|---|---|---|
| Gusarev et al. | 2017, IEEE CIBCB | Autoencoder | Có (bản tái hiện của hmchuong) | B2 |
| Rajaraman et al. | 2021, Diagnostics | ResNet-BS | Có | B1 |
| Sun et al. (BS-LDM) | 2025, IEEE JBHI | Latent diffusion | Có | Không (quá chậm) |
| Sun et al. (GL-LCM) | 2025, MICCAI | Latent consistency model | Có | B3 (tùy chọn) |
| | | | | |

## C. Phát hiện bất thường trên VinDr-CXR (để so baseline)
| Bài / giải pháp | Năm | Mô hình | mAP | Ghi chú (cách chia tập, IoU) |
|---|---|---|---|---|
| | | | | |

Lưu ý khi so sánh: cuộc thi Kaggle chấm **mAP@0.4**, còn đồ án dùng mAP@50 và mAP@[0.5:0.95] trên tập test tự chia, nên các con số không so trực tiếp được.

## Kết luận (điền sau khi tìm)
- Đã có nghiên cứu nào làm giống không? …
- Điểm khác biệt của đồ án: …
