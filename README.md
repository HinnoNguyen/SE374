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
Các nghiên cứu trước (ví dụ Rajaraman et al. 2021) chỉ chứng minh xóa bóng xương giúp **phân loại**; đồ án này kiểm tra trên bài toán **phát hiện vùng bất thường**.

> **Câu hỏi nghiên cứu:** *Xóa bóng xương có giúp phát hiện bất thường trên ảnh X-quang ngực tốt hơn không, và với những loại bệnh nào?*

Cách làm (theo góp ý của giảng viên):
1. **Dùng các mô hình xóa xương đã công bố, có sẵn checkpoint** (ResNet-BS của Rajaraman et al. 2021, autoencoder của Gusarev et al. 2017; tùy chọn: GL-LCM, MICCAI 2025),
   áp dụng lên toàn bộ ảnh **VinDr-CXR** (dữ liệu Việt Nam, Nguyen et al. 2022).
2. **Huấn luyện YOLO** phát hiện 14 loại bất thường trên ảnh gốc và trên từng phiên bản ảnh đã xóa xương, với **cùng cấu hình**.
3. **Đánh giá bằng mAP@50 và mAP@[0.5:0.95]**, tổng thể và theo từng loại bệnh.
4. *(Tùy chọn)* Tự huấn luyện mô hình xóa xương trên JSRT/BSE-JSRT (đánh giá PSNR/SSIM trên JSRT), rồi thêm vào so sánh ở bước 3.

```
                         ┌─ ResNet-BS (Rajaraman 2021) ──► B1 ┐
VinDr-CXR (ảnh gốc) ─────┼─ AE (Gusarev 2017, hmchuong) ──► B2 ┤
        │                ├─ GL-LCM (MICCAI 2025, tùy chọn) ─► B3 ┤──► YOLO ──► mAP@50, mAP@[.5:.95]
        │                └─ Mô hình tự train (tùy chọn) ───► B4 ┘          theo từng loại bệnh
        └──────────────────────────── A (baseline) ────────────────► YOLO ──►     so với A
```

#### 📚 Bài báo tham khảo chính
1. S. Rajaraman et al. *Chest X-ray Bone Suppression for Improving Classification of Tuberculosis-Consistent Findings.* **Diagnostics** 11(5):840, 2021. [DOI](https://doi.org/10.3390/diagnostics11050840) · [PDF](https://www.mdpi.com/2075-4418/11/5/840/pdf) · [Code](https://github.com/sivaramakrishnan-rajaraman/CXR-bone-suppression)
2. H. Q. Nguyen et al. *VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations.* **Scientific Data** 9:429, 2022. [DOI](https://doi.org/10.1038/s41597-022-01498-w) · [arXiv](https://arxiv.org/abs/2012.15029)

Danh sách đầy đủ: [`docs/references.md`](docs/references.md) (BibTeX: [`docs/references.bib`](docs/references.bib)). Kế hoạch & phân công: [`docs/PLAN.md`](docs/PLAN.md). Related works: [`docs/related_works.md`](docs/related_works.md).

#### 🗂️ Cấu trúc thư mục
```
SE374/
├── LT/                         # Tài liệu bài giảng
├── docs/                       # Tài liệu tham khảo, kế hoạch, báo cáo
├── scripts/
│   ├── download_data.py        # Tải dữ liệu bằng Kaggle CLI
│   ├── convert_pretrained.py   # Tải + chuyển checkpoint đã công bố (ResNet-BS, AE) sang PyTorch
│   ├── run_experiment.py       # Chạy trọn 1 thí nghiệm (A/B1/B2…) giống nhau trên mọi máy
│   ├── dicom_sizes.py          # Lấy kích thước ảnh gốc từ header DICOM
│   └── smoke_test.py           # Kiểm thử toàn pipeline trên dữ liệu giả
├── notebooks/
│   └── vindr_experiments.ipynb # Notebook dùng chung cho Kaggle / Colab
├── src/
│   ├── common/                 # Đọc/ghi ảnh (PNG, DICOM), CLAHE, tiện ích
│   ├── bone_suppression/       # Xóa xương: infer (mô hình có sẵn/tự train), models, train, evaluate, prepare_jsrt
│   └── detection/              # Phát hiện: prepare_vindr (WBF → YOLO), train_yolo, evaluate_yolo, compare, make_fused
├── data/                       # Dữ liệu (KHÔNG đưa lên git)
├── checkpoints/                # Trọng số: pretrained/ (đã chuyển đổi) + tự train (KHÔNG đưa lên git)
├── runs/                       # Kết quả train YOLO (KHÔNG đưa lên git)
└── results/                    # Bảng số liệu, hình ảnh cho báo cáo (CÓ đưa lên git)
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

#### ☁️ Chạy trên Kaggle / Colab (cả nhóm dùng chung)
Mở [`notebooks/vindr_experiments.ipynb`](notebooks/vindr_experiments.ipynb) trên **Kaggle** (khuyên dùng: VinDr có sẵn, chạy nền được) hoặc Colab,
đặt `EXPERIMENT`, rồi chạy. Mỗi lần chạy gọi đúng **một** lệnh, và lệnh này cũng chạy được ở local:
```bash
python scripts/run_experiment.py --experiment A_original --auto-find /kaggle/input
```
**Quy tắc:** mọi thí nghiệm YOLO chạy trên **cùng một nền tảng** (Kaggle T4) với **cùng cấu hình** (`--model --imgsz --epochs --batch`),
vì phần cứng khác nhau cũng làm kết quả lệch.

**Lưu kết quả ở đâu**

| Thứ | Ở đâu | Vì sao |
|---|---|---|
| `data/vindr_splits.csv`, `data/vindr_train_sizes.csv` | **git** | Nhỏ, chỉ có mã ảnh/kích thước. Mọi người **bắt buộc** dùng chung một cách chia tập |
| `results/` (CSV mAP, biểu đồ, `run_info.json`) | **git** | Nhỏ, dùng cho báo cáo; train xong không cần chạy lại |
| `best.pt` / `last.pt` của YOLO, `checkpoints/*.pt` | **GitHub Releases** hoặc Drive | File lớn; git chặn sẵn `*.pt` |
| Ảnh VinDr (gốc hoặc đã xóa xương) | Chỉ trên Kaggle / máy cá nhân, có thể lưu thành **Kaggle Dataset riêng tư** để dùng lại | Giấy phép cấm chia sẻ lại dữ liệu |

Thí nghiệm đầu tiên sẽ tạo `data/vindr_splits.csv`: **commit file này trước** khi ai chạy thí nghiệm tiếp theo.

#### 🚀 Quy trình chạy (từng bước, chạy local)
Mọi lệnh chạy từ thư mục gốc `SE374/`. Chi tiết tham số: thêm `--help` vào mỗi lệnh.

**0. Tải dữ liệu** (cần `kaggle.json` và bấm *Join competition* VinBigData trên Kaggle, xem [`data/README.md`](data/README.md))
```bash
python scripts/download_data.py --what all
```

**Bước 1: Chuẩn bị mô hình xóa xương có sẵn** (chỉ cần chạy 1 lần; `hmchuong` cần `pip install tensorflow-cpu` lúc chuyển đổi)
```bash
python scripts/convert_pretrained.py --which all
# -> checkpoints/pretrained/rajaraman_resnet_bs.pt , checkpoints/pretrained/hmchuong_ae.pt
```

**Bước 2: Xóa xương toàn bộ ảnh VinDr bằng từng mô hình** (chạy thử `--limit 20` trước, xem ảnh rồi mới chạy hết)
```bash
python -m src.bone_suppression.infer --ckpt checkpoints/pretrained/rajaraman_resnet_bs.pt --input data/raw/vindr_1024/train --output data/vindr_bs/rajaraman/train --skip-existing
python -m src.bone_suppression.infer --ckpt checkpoints/pretrained/hmchuong_ae.pt --input data/raw/vindr_1024/train --output data/vindr_bs/hmchuong/train --skip-existing
```
`--shard 1/4` … `--shard 4/4` chia việc cho nhiều máy / nhiều tài khoản Kaggle.

**Bước 3: Chuyển sang định dạng YOLO** (cùng một cách chia tập cho mọi thí nghiệm, lưu ở `data/vindr_splits.csv`)
```bash
python -m src.detection.prepare_vindr --csv data/raw/vindr/train.csv --meta-csv data/raw/vindr_1024/train_meta.csv --images-dir data/raw/vindr_1024/train --out-root data/yolo/A_original
python -m src.detection.prepare_vindr --csv data/raw/vindr/train.csv --meta-csv data/raw/vindr_1024/train_meta.csv --images-dir data/vindr_bs/rajaraman/train --out-root data/yolo/B1_rajaraman
python -m src.detection.prepare_vindr --csv data/raw/vindr/train.csv --meta-csv data/raw/vindr_1024/train_meta.csv --images-dir data/vindr_bs/hmchuong/train --out-root data/yolo/B2_hmchuong
```

**Bước 4: Huấn luyện YOLO** (mọi thí nghiệm phải **cùng** `--model`, `--imgsz`, `--epochs`, `--batch`)
```bash
python -m src.detection.train_yolo --data data/yolo/A_original/data.yaml --name A_original
python -m src.detection.train_yolo --data data/yolo/B1_rajaraman/data.yaml --name B1_rajaraman
python -m src.detection.train_yolo --data data/yolo/B2_hmchuong/data.yaml --name B2_hmchuong
```

**Bước 5: Đánh giá mAP@50, mAP@[0.5:0.95] theo từng loại bệnh và so sánh**
```bash
python -m src.detection.evaluate_yolo --weights runs/vindr/A_original/weights/best.pt --data data/yolo/A_original/data.yaml --out results/detection/A_original.csv
python -m src.detection.evaluate_yolo --weights runs/vindr/B1_rajaraman/weights/best.pt --data data/yolo/B1_rajaraman/data.yaml --out results/detection/B1_rajaraman.csv
python -m src.detection.evaluate_yolo --weights runs/vindr/B2_hmchuong/weights/best.pt --data data/yolo/B2_hmchuong/data.yaml --out results/detection/B2_hmchuong.csv
python -m src.detection.compare A=results/detection/A_original.csv B1=results/detection/B1_rajaraman.csv B2=results/detection/B2_hmchuong.csv --metric AP50 --out results/detection/compare_AP50
python -m src.detection.compare A=results/detection/A_original.csv B1=results/detection/B1_rajaraman.csv B2=results/detection/B2_hmchuong.csv --metric AP50_95 --out results/detection/compare_AP50_95
```

<details>
<summary><b>Tùy chọn: tự huấn luyện mô hình xóa xương trên JSRT/BSE-JSRT</b></summary>

```bash
# Ghép cặp, resize 512, tự nhận ảnh âm bản và đảo lại, căn chỉnh (ECC), chia train/val/test
python -m src.bone_suppression.prepare_jsrt --source-dir data/raw/bse/JSRT --target-dir data/raw/bse/BSE_JSRT --out-dir data/processed/bse_512 --size 512 --register
python -m src.bone_suppression.train --data-root data/processed/bse_512 --model unet --amp
python -m src.bone_suppression.evaluate --ckpt checkpoints/bone_suppression/unet/best.pt --data-root data/processed/bse_512 --out-dir results/bone_suppression/unet
# Mô hình có sẵn cũng đánh giá được trên JSRT theo cùng cách (lưu ý: chúng đã thấy JSRT khi train)
python -m src.bone_suppression.evaluate --ckpt checkpoints/pretrained/rajaraman_resnet_bs.pt --data-root data/processed/bse_512 --out-dir results/bone_suppression/rajaraman
```
Sau đó xóa xương VinDr bằng `infer` như Bước 2 và thêm thí nghiệm B4.
</details>

#### 💾 Checkpoints
Trọng số **không** commit vào git (quá nặng). Sau khi train xong, đưa lên **GitHub Releases** của repo (hoặc Google Drive / Hugging Face) và cập nhật link tại đây:

| Mô hình | File | Link |
|---|---|---|
| Xóa xương có sẵn (ResNet-BS, AE) | `checkpoints/pretrained/*.pt` | Tạo lại bằng `scripts/convert_pretrained.py` |
| Xóa xương tự train (tùy chọn) | `checkpoints/bone_suppression/<model>/best.pt` | *(cập nhật)* |
| YOLO A / B1 / B2 / … | `runs/vindr/<run>/weights/best.pt` | *(cập nhật)* |

#### 🖥️ Ứng dụng demo
*(Đang phát triển, do thành viên khác phụ trách.)*

#### ⚠️ Lưu ý
- Dữ liệu VinDr-CXR chỉ dùng cho nghiên cứu/giáo dục và **không được chia sẻ lại**: không commit ảnh hay CSV dữ liệu lên GitHub.
- Kết quả của mô hình chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ.
