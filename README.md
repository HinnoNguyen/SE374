<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>SE374 - PHÁT TRIỂN PHẦN MỀM XỬ LÝ ẢNH Y KHOA</b></h1>



## BẢNG MỤC LỤC
* [ Giới thiệu môn học](#gioithieumonhoc)
* [ Giảng viên hướng dẫn](#giangvien)
* [ Thành viên nhóm](#thanhvien)
* [ Đồ án môn học](#doan)


## GIỚI THIỆU MÔN HỌC
<a name="gioithieumonhoc"></a>
* **Tên môn học**: Phát triển phần mềm xử lý ảnh y khoa
* **Mã môn học**: SE374
* **Lớp học**: SE374.R11
* **Năm học**: 2026-2027


## GIẢNG VIÊN HƯỚNG DẪN
<a name="giangvien"></a>
* ThS. **Bùi Cao Doanh** - *doanhbc@uit.edu.vn*


## THÀNH VIÊN NHÓM
<a name="thanhvien"></a>
| STT    | MSSV          | Họ và Tên              | Github                                               | Email                   |
|:------:|:-------------:|:-----------------------|:-----------------------------------------------------|:------------------------|
| 1      | 24520237      | Nguyễn Cao Cường       | [CuongNguyen7983](https://github.com/CuongNguyen7983) | 24520237@gm.uit.edu.vn  |
| 2      | 23520588      | Nguyễn Lan Hương       | [Hinno](https://github.com/HinnoNguyen)              | 23520588@gm.uit.edu.vn  |
| 3      | 23521261      | Nguyễn Đình Hoàng Quân | [DSeaNguyen](https://github.com/DSeaNguyen)          | 23521261@gm.uit.edu.vn  |
| 4      | 23521548      | Dương Quốc Thuận       | [duongthuan0405](https://github.com/duongthuan0405)  | 23521548@gm.uit.edu.vn  |


## ĐỒ ÁN MÔN HỌC
<a name="doan"></a>

### 📌 Đề tài: *Bone Suppression in Chest X-rays for Improving Abnormality Detection on the VinDr-CXR Dataset*
*(Xóa bóng xương sườn trên ảnh X-quang ngực nhằm cải thiện phát hiện bất thường trên bộ dữ liệu VinDr-CXR)*

#### 📖 Giới thiệu đề tài
Trên X-quang ngực, xương sườn và xương đòn chồng lên nhu mô phổi và có thể che khuất tổn thương (nốt phổi, đông đặc, xơ hóa…).
Đồ án gồm hai phần:

1. **Xóa bóng xương (bone suppression):** huấn luyện mạng ảnh-sang-ảnh biến X-quang thường thành ảnh chỉ còn mô mềm,
   dựa theo Rajaraman et al. (2021), trên bộ JSRT / BSE-JSRT. Đánh giá bằng PSNR, SSIM, MS-SSIM.
2. **Phát hiện bất thường:** huấn luyện YOLO phát hiện 14 loại bất thường trên **VinDr-CXR** (dữ liệu Việt Nam, Nguyen et al. 2022),
   so sánh khi đầu vào là ảnh gốc, ảnh đã xóa xương, và ảnh ghép 3 kênh; phân tích bệnh nào được cải thiện, bệnh nào bị ảnh hưởng.

```
JSRT + BSE-JSRT ──► [Phần 1] U-Net / ResNet-BS / AE ──► mô hình xóa xương
                                                              │
VinDr-CXR (ảnh gốc) ──────────────────────────────────────────┤
        │                                                     ▼
        │                                          VinDr-CXR đã xóa xương
        ▼                                                     │
 [Phần 2] YOLO — A: ảnh gốc    B: ảnh xóa xương    C: 3 kênh [gốc, xóa xương, CLAHE]
        │
        ▼
 So sánh AP theo từng loại bệnh (A vs B vs C)
```

#### 📚 Bài báo tham khảo chính
1. S. Rajaraman et al. *Chest X-ray Bone Suppression for Improving Classification of Tuberculosis-Consistent Findings.* **Diagnostics** 11(5):840, 2021. [DOI](https://doi.org/10.3390/diagnostics11050840) · [PDF](https://www.mdpi.com/2075-4418/11/5/840/pdf) · [Code](https://github.com/sivaramakrishnan-rajaraman/CXR-bone-suppression)
2. H. Q. Nguyen et al. *VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations.* **Scientific Data** 9:429, 2022. [DOI](https://doi.org/10.1038/s41597-022-01498-w) · [arXiv](https://arxiv.org/abs/2012.15029)

Danh sách đầy đủ: [`docs/references.md`](docs/references.md) (BibTeX: [`docs/references.bib`](docs/references.bib)). Kế hoạch & phân công: [`docs/PLAN.md`](docs/PLAN.md).

#### 🗂️ Cấu trúc thư mục
```
SE374/
├── LT/                         # Tài liệu bài giảng
├── docs/                       # Tài liệu tham khảo, kế hoạch, báo cáo
├── scripts/
│   ├── download_data.py        # Tải dữ liệu bằng Kaggle CLI
│   └── smoke_test.py           # Kiểm thử toàn pipeline trên dữ liệu giả
├── src/
│   ├── common/                 # Đọc/ghi ảnh (PNG, DICOM), CLAHE, tiện ích
│   ├── bone_suppression/       # Phần 1: prepare_jsrt, dataset, models, losses, metrics, train, evaluate, infer
│   └── detection/              # Phần 2: prepare_vindr (WBF → YOLO), make_fused, train_yolo, evaluate_yolo, compare
├── data/                       # Dữ liệu (KHÔNG đưa lên git)
├── checkpoints/                # Trọng số mô hình xóa xương (KHÔNG đưa lên git)
├── runs/                       # Kết quả train YOLO (KHÔNG đưa lên git)
└── results/                    # Bảng số liệu, hình ảnh cho báo cáo
```

#### ⚙️ Cài đặt
```bash
# 1. PyTorch có CUDA: chọn lệnh phù hợp tại https://pytorch.org/get-started/locally/ , ví dụ:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# 2. Các thư viện còn lại
pip install -r requirements.txt
# 3. Kiểm tra môi trường (dữ liệu giả, chạy trên CPU vài phút)
python scripts/smoke_test.py --with-yolo
```

#### 🚀 Quy trình chạy
Mọi lệnh chạy từ thư mục gốc `SE374/`. Chi tiết tham số: thêm `--help` vào mỗi lệnh.

**0. Tải dữ liệu** (cần `kaggle.json` và bấm *Join competition* VinBigData trên Kaggle, xem [`data/README.md`](data/README.md))
```bash
python scripts/download_data.py --what all
```

**Phần 1: Xóa bóng xương**
```bash
# Ghép cặp JSRT/BSE-JSRT, resize 512, căn chỉnh (ECC), chia train/val/test
python -m src.bone_suppression.prepare_jsrt --source-dir data/raw/bse/JSRT --target-dir data/raw/bse/BSE_JSRT --out-dir data/processed/bse_512 --size 512 --register

# Huấn luyện 3 kiến trúc để so sánh
python -m src.bone_suppression.train --data-root data/processed/bse_512 --model unet --amp
python -m src.bone_suppression.train --data-root data/processed/bse_512 --model resnet_bs --crop 256 --batch-size 8 --amp
python -m src.bone_suppression.train --data-root data/processed/bse_512 --model ae --amp

# Đánh giá trên tập test (PSNR, SSIM, MS-SSIM + hình minh họa)
python -m src.bone_suppression.evaluate --ckpt checkpoints/bone_suppression/unet/best.pt --data-root data/processed/bse_512 --out-dir results/bone_suppression/unet
```

**Phần 2: Phát hiện bất thường trên VinDr-CXR**
```bash
# Xóa xương toàn bộ ảnh VinDr bằng mô hình tốt nhất ở Phần 1
python -m src.bone_suppression.infer --ckpt checkpoints/bone_suppression/unet/best.pt --input data/raw/vindr_1024/train --output data/vindr_suppressed/train --skip-existing
python -m src.detection.make_fused --orig-dir data/raw/vindr_1024/train --supp-dir data/vindr_suppressed/train --out-dir data/vindr_fused/train

# Chuyển sang định dạng YOLO (cùng một cách chia tập cho A/B/C)
python -m src.detection.prepare_vindr --csv data/raw/vindr/train.csv --meta-csv data/raw/vindr_1024/train_meta.csv --images-dir data/raw/vindr_1024/train --out-root data/yolo/original
python -m src.detection.prepare_vindr --csv data/raw/vindr/train.csv --meta-csv data/raw/vindr_1024/train_meta.csv --images-dir data/vindr_suppressed/train --out-root data/yolo/suppressed
python -m src.detection.prepare_vindr --csv data/raw/vindr/train.csv --meta-csv data/raw/vindr_1024/train_meta.csv --images-dir data/vindr_fused/train --out-root data/yolo/fused

# Huấn luyện (cùng siêu tham số)
python -m src.detection.train_yolo --data data/yolo/original/data.yaml --name A_original
python -m src.detection.train_yolo --data data/yolo/suppressed/data.yaml --name B_suppressed
python -m src.detection.train_yolo --data data/yolo/fused/data.yaml --name C_fused

# Đánh giá theo từng loại bệnh và so sánh
python -m src.detection.evaluate_yolo --weights runs/vindr/A_original/weights/best.pt --data data/yolo/original/data.yaml --out results/detection/A_original.csv
python -m src.detection.evaluate_yolo --weights runs/vindr/B_suppressed/weights/best.pt --data data/yolo/suppressed/data.yaml --out results/detection/B_suppressed.csv
python -m src.detection.evaluate_yolo --weights runs/vindr/C_fused/weights/best.pt --data data/yolo/fused/data.yaml --out results/detection/C_fused.csv
python -m src.detection.compare A=results/detection/A_original.csv B=results/detection/B_suppressed.csv C=results/detection/C_fused.csv --out results/detection/compare
```

#### 💾 Checkpoints
Trọng số **không** commit vào git (quá nặng). Sau khi train xong, đưa lên **GitHub Releases** của repo (hoặc Google Drive / Hugging Face) và cập nhật link tại đây:

| Mô hình | File | Link |
|---|---|---|
| Xóa xương (U-Net) | `checkpoints/bone_suppression/unet/best.pt` | *(cập nhật)* |
| Xóa xương (ResNet-BS) | `checkpoints/bone_suppression/resnet_bs/best.pt` | *(cập nhật)* |
| YOLO A / B / C | `runs/vindr/<run>/weights/best.pt` | *(cập nhật)* |

#### 🖥️ Ứng dụng demo
*(Đang phát triển, do thành viên khác phụ trách.)*

#### ⚠️ Lưu ý
- Dữ liệu VinDr-CXR chỉ dùng cho nghiên cứu/giáo dục và **không được chia sẻ lại**: không commit ảnh hay CSV dữ liệu lên GitHub.
- Kết quả của mô hình chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ.
