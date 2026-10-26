# Tài liệu tham khảo

Đồ án dựa trên **2 bài báo chính**. Các bài còn lại là tài liệu bổ trợ cho từng thành phần.

> Ghi chú: [1], [2], DOI của [3], và [11], [12] (theo README của tác giả) đã được kiểm tra trực tuyến (10/2026).
> Các tài liệu bổ trợ [4]–[10] là các bài kinh điển, nên kiểm tra lại số trang / tập trước khi đưa vào báo cáo cuối.
> File BibTeX tương ứng: [`references.bib`](references.bib).

---

## Bài báo chính

### [1] Xóa bóng xương: Rajaraman et al., 2021 (Diagnostics)

**S. Rajaraman, G. Zamzmi, L. Folio, P. Alderson, S. Antani.**
*Chest X-ray Bone Suppression for Improving Classification of Tuberculosis-Consistent Findings.*
**Diagnostics**, 11(5):840, 2021. DOI: [10.3390/diagnostics11050840](https://doi.org/10.3390/diagnostics11050840)

- Trang bài (open access): https://www.mdpi.com/2075-4418/11/5/840
- PDF: https://www.mdpi.com/2075-4418/11/5/840/pdf
- Code + trọng số: https://github.com/sivaramakrishnan-rajaraman/CXR-bone-suppression

**Vai trò trong đồ án:** phương pháp gốc cho Phần 1, đồng thời là ý tưởng "xóa xương rồi mới chẩn đoán" mà đồ án mở rộng.

**Tóm tắt nội dung:**
- So sánh 4 mô hình xóa xương: AE-BS (autoencoder), ConvNet-BS, RL-BS (học phần dư) và **ResNet-BS (16 khối residual), là mô hình tốt nhất** (PSNR ≈ 34.07, MS-SSIM ≈ 0.983).
- Train trên JSRT/BSE-JSRT (247 ảnh, augment lên ~4.500 cặp), kiểm tra chéo trên NIH-CC-DES (ảnh chụp dual-energy thật).
- Ảnh đã xóa xương giúp **phân loại lao tốt hơn có ý nghĩa thống kê**: AUC Shenzhen 0.899 → 0.954, Montgomery 0.857 → 0.964.
- Đo bằng MAE, PSNR, SSIM, MS-SSIM.

### [2] Dữ liệu X-quang Việt Nam: Nguyen et al., 2022 (Scientific Data)

**H. Q. Nguyen, K. Lam, L. T. Le, H. H. Pham, D. Q. Tran, et al.**
*VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations.*
**Scientific Data**, 9:429, 2022. DOI: [10.1038/s41597-022-01498-w](https://doi.org/10.1038/s41597-022-01498-w)

- Trang bài (open access): https://www.nature.com/articles/s41597-022-01498-w
- arXiv (PDF): https://arxiv.org/abs/2012.15029
- Dữ liệu gốc (PhysioNet, cần tài khoản credentialed): https://doi.org/10.13026/3akn-b287
- Bản dùng trong đồ án (cuộc thi Kaggle, 14 nhãn): https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection

**Vai trò trong đồ án:** bộ dữ liệu cho Phần 2 (phát hiện bất thường).

**Tóm tắt nội dung:**
- 18.000 ảnh X-quang ngực thẳng (PA) từ 2 bệnh viện Việt Nam: 15.000 train, 3.000 test.
- Mỗi ảnh train được **3 bác sĩ gán nhãn độc lập**; ảnh test là đồng thuận của 5 bác sĩ.
- 22 nhãn khung (local) và 6 nhãn chẩn đoán (global). Bản Kaggle gộp còn **14 loại bất thường + "No finding"**.
- **Giấy phép:** chỉ dùng cho nghiên cứu, giáo dục; **không được chia sẻ lại dữ liệu** (không đưa ảnh lên GitHub).

---

## Tài liệu bổ trợ

| # | Trích dẫn | Dùng cho |
|---|---|---|
| [3] | M. Gusarev, R. Kuleev, A. Khan, A. Ramirez Rivera, A. M. Khattak. *Deep learning models for bone suppression in chest radiographs.* IEEE CIBCB, 2017. DOI: [10.1109/CIBCB.2017.8058543](https://doi.org/10.1109/CIBCB.2017.8058543). Code tái hiện: https://github.com/hmchuong/ML-BoneSuppression | Mô hình autoencoder xóa xương (so sánh) |
| [4] | J. Shiraishi et al. *Development of a digital image database for chest radiographs with and without a lung nodule.* American Journal of Roentgenology, 174(1):71–74, 2000. | Bộ dữ liệu JSRT |
| [5] | S. Juhász, Á. Horváth, L. Nikházy, G. Horváth. *Segmentation of anatomical structures on chest radiographs.* XII Mediterranean Conference on Medical and Biological Engineering and Computing (MEDICON), IFMBE Proceedings vol. 29, Springer, 2010. | Bộ ảnh đã xóa xương BSE-JSRT |
| [6] | O. Ronneberger, P. Fischer, T. Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, 2015. arXiv: [1505.04597](https://arxiv.org/abs/1505.04597) | Kiến trúc U-Net |
| [7] | H. Zhao, O. Gallo, I. Frosio, J. Kautz. *Loss Functions for Image Restoration with Neural Networks.* IEEE Transactions on Computational Imaging, 3(1):47–57, 2017. | Hàm loss MS-SSIM + L1 |
| [8] | Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli. *Image quality assessment: from error visibility to structural similarity.* IEEE Transactions on Image Processing, 13(4):600–612, 2004. | Chỉ số SSIM |
| [9] | R. Solovyev, W. Wang, T. Gabruseva. *Weighted boxes fusion: Ensembling boxes from different object detection models.* Image and Vision Computing, 107:104117, 2021. | Gộp khung của 3 bác sĩ (WBF) |
| [10] | G. Jocher, J. Qiu, A. Chaurasia. *Ultralytics YOLO*, 2023. https://github.com/ultralytics/ultralytics | Mô hình phát hiện YOLO |
| [11] | Y. Sun, Z. Chen, H. Zheng, Y. Lu, L. Duan, F. Fan, A. Elazab, X. Wan, C. Wang, R. Ge. *GL-LCM: Global-Local Latent Consistency Models for Fast High-Resolution Bone Suppression in Chest X-Ray Images.* MICCAI 2025, pp. 222–232, Springer. arXiv: [2508.03357](https://arxiv.org/abs/2508.03357). Code + checkpoint: https://github.com/diaoquesang/GL-LCM | Mô hình xóa xương có sẵn (B3, giảng viên gợi ý) |
| [12] | Y. Sun, Z. Chen, H. Zheng, R. Ge, J. Liu, W. Min, A. Elazab, X. Wan, C. Wang. *BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models.* IEEE Journal of Biomedical and Health Informatics, 2025. arXiv: [2412.15670](https://arxiv.org/abs/2412.15670). Code + checkpoint: https://github.com/diaoquesang/BS-LDM | Mô hình xóa xương có sẵn (tham khảo) |

Code + checkpoint của [1] và [3] được chuyển sang PyTorch bằng `scripts/convert_pretrained.py` (đã kiểm tra khớp với bản gốc).
