# Kết quả bổ sung D E F

D/E giữ λ từ Validation của M1 chính trong từng lượt; F là CNN-HAR chỉ ACC. Tất cả dùng cùng Test và cùng cửa sổ hợp lệ.

D: Accuracy 0.6927, Macro-F1 0.7222; ΔMacro-F1 so với M1: -0.01301. MAE theo người 6.747 ± 3.870 bpm; ΔMAE +0.090 bpm.

E: Accuracy 0.4230, Macro-F1 0.4275; ΔMacro-F1 so với M1: -0.30768. MAE theo người 6.748 ± 2.949 bpm; ΔMAE +0.091 bpm.

F: Accuracy 0.7011, Macro-F1 0.7306; ΔMacro-F1 so với B2: -0.00055.

Chênh lệch chỉ mô tả một seed và năm nhóm người cố định; không tự kết luận có ý nghĩa thống kê.

Latency chỉ đối chiếu với baseline được đo trong cùng benchmark_id; nếu thiếu checkpoint gốc, chưa có đối chứng latency cùng phiên.