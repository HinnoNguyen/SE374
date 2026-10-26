# Dữ liệu

Thư mục `data/` **không được đưa lên git** (đã chặn trong `.gitignore`), trừ file này.

## 1. Chuẩn bị Kaggle
1. `pip install kaggle`
2. Kaggle → *Settings* → *API* → **Create New Token** → lưu `kaggle.json` vào `%USERPROFILE%\.kaggle\` (Windows).
   **Không commit file này.**
3. Mở trang cuộc thi https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection
   và bấm **Join Competition / I understand and accept** (bắt buộc để tải `train.csv`).

## 2. Tải
```bash
python scripts/download_data.py --what all
```

| Dữ liệu | Nguồn | Vị trí sau khi tải |
|---|---|---|
| JSRT + BSE-JSRT (247 cặp ảnh) | Kaggle dataset `hmchuong/xray-bone-shadow-supression` | `data/raw/bse/` |
| Nhãn VinDr-CXR (`train.csv`) | Kaggle competition `vinbigdata-chest-xray-abnormalities-detection` | `data/raw/vindr/` |
| Ảnh VinDr-CXR 1024×1024 PNG | Kaggle dataset `xhlulu/vinbigdata-chest-xray-resized-png-1024x1024` | `data/raw/vindr_1024/` |

Sau khi tải, **kiểm tra tên thư mục thực tế** (ví dụ `JSRT/` và `BSE_JSRT/` có thể nằm lồng thêm một cấp) rồi sửa đường dẫn trong lệnh cho đúng.

Trên **Kaggle** không cần tải: gắn competition và dataset PNG làm *Input* (xem `notebooks/vindr_experiments.ipynb`).

## 3. Lưu ý quan trọng
- **Kích thước gốc:** toạ độ khung trong `train.csv` tính theo pixel của ảnh DICOM gốc. `prepare_vindr.py` cần file meta có kích thước gốc
  (cột `width`/`height` hoặc `dim0`/`dim1`). Kiểm tra file meta đi kèm bộ PNG (ví dụ `train_meta.csv`) và truyền qua `--meta-csv`.
  Nếu không có, `scripts/dicom_sizes.py` đọc kích thước từ header DICOM (trên Kaggle DICOM có sẵn) → `data/vindr_train_sizes.csv` (commit lên git).
- **Không dùng bộ "augmented" có sẵn** của BSE-JSRT để train: bộ đó được tạo từ *toàn bộ* 247 ảnh, kể cả ảnh test, nên sẽ bị rò rỉ dữ liệu.
  Code đã augment trực tiếp khi train, chỉ trên tập train.
- **BSE-JSRT là ảnh do phần mềm tạo ra**, không phải ảnh chụp dual-energy thật. Cần nêu điều này trong phần hạn chế của báo cáo.
- **Giấy phép VinDr-CXR:** chỉ dùng cho nghiên cứu/giáo dục, không chia sẻ lại dữ liệu.

## 4. Cấu trúc sau khi xử lý
```
data/
├── raw/                       # dữ liệu tải về
├── processed/bse_512/         # source/, target/, splits.csv, previews/
├── vindr_fused/train/         # ảnh 3 kênh cho thí nghiệm C (tùy chọn)
├── vindr_splits.csv           # cách chia train/val/test DÙNG CHUNG cho mọi thí nghiệm (có trên git)
├── vindr_train_sizes.csv      # kích thước ảnh gốc (có trên git, nếu phải tạo từ DICOM)
├── vindr_bs/<thí nghiệm>/train/   # ảnh VinDr đã xóa xương (B1_rajaraman, B2_hmchuong, …)
└── yolo/<thí nghiệm>/         # images/, labels/, data.yaml (A_original, B1_rajaraman, …)
```
