# Nhận dạng hoạt động và ước lượng nhịp tim trên PPG-DaLiA

**Sinh viên:** Đặng Minh Tài — **MSSV:** 23110148 — **Lớp:** IOT-01CLC  
**Môn học:** Trí tuệ nhân tạo cho IoT  
**Giảng viên:** Th.S Hồ Nhựt Minh

## Nội dung

- `M1_PPG_DaLiA_DangMinhTai_Nop_bai.ipynb`: notebook chính A–C, gồm dữ liệu, tiền xử lý, chia 5 fold theo người, B0/B1/B2/M1, khảo sát λ, đánh giá và đo tài nguyên.
- `M1_PPG_DaLiA_DangMinhTai_DEF_Nop_bai.ipynb`: notebook bổ sung D–F đã chạy; D dùng CE thường, E chỉ dùng PPG, F nhận dạng hoạt động chỉ bằng ACC.
- `results/M1_results/`: CSV, JSON, lịch sử, dự đoán Test và hình của thí nghiệm chính.
- `results/M1_results/DEF/`: kết quả của 15 mô hình bổ sung, gồm cấu hình, trạng thái hoàn thành và phép đo cùng phiên.
- `SHA256SUMS.txt`: mã kiểm tra tính toàn vẹn của các file mã và kết quả.

## Chạy trên Google Colab

1. Mở notebook chính bằng Colab. Chọn **Runtime → Change runtime type → GPU** (thực nghiệm đã dùng Tesla T4).
2. Chạy cell theo thứ tự và cấp quyền mount Google Drive. Notebook tự tải PPG-DaLiA từ UCI; không cần tải dữ liệu thủ công lên Colab.
3. Giữ `SMOKE=False` để thực hiện thí nghiệm đầy đủ. Chế độ SMOKE chỉ kiểm tra luồng chạy, không dùng số liệu này để báo cáo.
4. Kết quả chính và checkpoint nằm tại `/content/drive/MyDrive/M1_results`. Sau khi bị ngắt, chạy lại các cell để khôi phục checkpoint tương thích.
5. Khi thí nghiệm chính hoàn tất, mở notebook D–F và chạy theo thứ tự. Nó đọc cấu hình và λ đã chọn từ `M1_results`, ghi kết quả vào `M1_results/DEF`.

Notebook có thanh tiến độ `tqdm`. Có thể cần nhiều phiên GPU; thời lượng và khả năng cấp GPU phụ thuộc Colab. Notebook D–F đặt `SESSION_TRAIN_MINUTES=300` để tạm dừng có lưu checkpoint; giá trị này không bảo đảm Colab cấp đủ thời gian.

## Chỉ xem và đối chiếu kết quả đã có

Không cần huấn luyện lại để đọc báo cáo, output notebook và các CSV. Nếu dùng một Drive mới, có thể chép **nội dung** `results/M1_results/` vào `MyDrive/M1_results/` để có cấu hình và bảng kết quả đúng vị trí.

Gói nộp không chứa dữ liệu thô, cache `.npz` hoặc trọng số `.pt`. CSV đủ để kiểm tra các số liệu dự đoán đã lưu; chúng không thay thế checkpoint cho suy luận hoặc tiếp tục huấn luyện. Nếu Drive không còn checkpoint, chạy cell huấn luyện sẽ huấn luyện lại mô hình còn thiếu. Để tiếp tục đúng trạng thái cũ, phải giữ các `.pt` và checkpoint đang chạy trong Drive ban đầu.

## Giao thức thực nghiệm

- 15 người, 46.763 cửa sổ hợp lệ; cửa sổ 8 giây, bước 2 giây.
- Năm nhóm người; mỗi fold: 9 Train / 3 Validation / 3 Test. Đây là 5-fold theo nhóm người, không phải LOSO.
- Thống kê chuẩn hóa và trọng số lớp chỉ tính từ Train.
- M1 khảo sát λ = 0,1; 1; 10. Chọn epoch và λ trên Validation trước khi đánh giá Test.
- D/E giữ λ đã chọn của M1 theo fold. F dùng nhánh ACC-only theo đề cương, không chạy DeepPPG.
- Huấn luyện 25 mô hình chính và 15 mô hình D–F. Seed cố định 42; chưa khảo sát nhiều seed.

## Kết quả chính

| Mô hình | MAE trung bình theo người (bpm) | Accuracy HAR | Macro-F1 HAR |
|---|---:|---:|---:|
| B0 | 16,056 | — | — |
| B1 | 6,535 | — | — |
| B2 | — | 68,94% | 0,7312 |
| M1 | 6,657 | 69,62% | 0,7352 |
| D | 6,747 | 69,27% | 0,7222 |
| E | 6,748 | 42,30% | 0,4275 |
| F | — | 70,11% | 0,7306 |

MAE là trung bình không trọng số của MAE 15 người. Accuracy và Macro-F1 được tính gộp trên các cửa sổ Test; Macro-F1 dùng tám nhãn cố định. Xem CSV để lấy giá trị đầy đủ và kết quả từng fold/lớp/người.

M1 có 422.665 tham số, giảm 47,94% so với B1 + B2 (811.849). MAE M1 vẫn cao hơn B1 0,122 bpm, còn Macro-F1 cao hơn B2 0,00398. Lợi ích chính trong thí nghiệm này là quy mô mô hình; chất lượng chưa cải thiện đồng đều giữa hai nhiệm vụ.

## Môi trường và giới hạn

Các notebook đã chạy trên Tesla T4. Phiên D–F ghi Python 3.13.15, PyTorch 2.11.0+cu130, CUDA 13.0, NumPy 2.1.3 và SciPy 1.16.3. Notebook chính có chênh lệch thông tin phiên giữa output và JSON môi trường; xem báo cáo. Không ép cài lại PyTorch trong Colab chỉ để khớp chuỗi phiên bản.

Độ trễ chỉ đo forward GPU với batch=1, 50 warm-up và 500 lần lặp có đồng bộ CUDA. So sánh độ trễ trong cùng phiên đo, không trộn hai phiên chính và bổ sung. Chưa đo toàn bộ pipeline, triển khai MCU, năng lượng hoặc ý nghĩa thống kê. Lọc hai chiều là tiền xử lý ngoại tuyến.

## Dữ liệu

PPG-DaLiA, UCI Machine Learning Repository: https://doi.org/10.24432/C53890. Tham khảo nguồn và điều kiện sử dụng tại trang dữ liệu; không phát hành lại dữ liệu thô trong repository này.
