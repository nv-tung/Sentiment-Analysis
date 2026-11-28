# Đề tài 6: Phân tích cảm xúc văn bản tiếng Việt

So sánh TF-IDF + Logistic Regression, BiLSTM và PhoBERT; đánh giá ảnh hưởng của tách từ, teencode và văn bản không dấu.

## Dữ liệu

Dữ liệu chính: **UIT-VSFC** (phản hồi sinh viên), dùng đúng split gốc của Hugging Face
(`uitnlp/vietnamese_students_feedback`).

| File (thư mục `data/`) | Số câu |
|---|---|
| `train_segmented.csv` | 11.426 |
| `dev_segmented.csv` (validation) | 1.583 |
| `test_segmented.csv` | 3.166 |

### Các cột

| Cột | Ý nghĩa |
|---|---|
| `id` | Mã câu duy nhất theo tập (`train_0`, `dev_0`, `test_0`, ...), dùng để truy vết lỗi và ghép dự đoán của các mô hình. Không dùng làm đặc trưng, không xáo trộn thứ tự dòng |
| `sentence` | Câu gốc |
| `sentiment` | Nhãn cảm xúc: **0 = negative, 1 = neutral, 2 = positive** |
| `topic` | Chủ đề: 0 = lecturer, 1 = training_program, 2 = facility, 3 = others |
| `clean_sentence` | Câu đã làm sạch, **chưa tách từ** (dùng cho chế độ "không tách từ") |
| `segmented_sentence` | Câu đã làm sạch **và tách từ** VnCoreNLP, từ ghép nối bằng `_` (dùng cho chế độ "có tách từ", và cho PhoBERT) |

### Phân bố nhãn (rất lệch)

| Tập | negative (0) | neutral (1) | positive (2) |
|---|---|---|---|
| Train | 46,6% | **4,0%** | 49,4% |
| Dev | 44,5% | 4,6% | 50,9% |
| Test | 44,5% | 5,3% | 50,2% |

Vì lớp neutral rất ít, **chỉ số chính là macro-F1**, không phải accuracy.

### Quy trình làm sạch

1. Chuẩn hóa Unicode NFC, chuyển chữ thường.
2. Rút gọn chữ cái lặp từ 3 lần trở lên (`hayyyy` thành `hay`), không đụng chữ số.
3. Thay teencode/viết tắt/không dấu phổ biến bằng từ điển (`gv` thành `giảng viên`, `ko` thành `không`, ...).
4. Xóa ký tự đặc biệt và emoji, giữ lại `!` và `?`.
5. Tách từ bằng VnCoreNLP (`py_vncorenlp`).

## Đánh giá thống nhất: `src/evaluate.py`

Mọi mô hình dùng chung một hàm để so sánh công bằng:

```python
import sys
sys.path.append("src")          # chạy từ thư mục gốc của repo
from evaluate import evaluate, summarize

evaluate(test_df["sentiment"], y_pred,
         model_name="bilstm_fasttext_segmented", save_dir="results", show_plot=False)

summarize("results")   # bảng so sánh mọi mô hình, sắp theo macro-F1
```

Quy ước đặt tên: `<mô_hình>_<segmented|raw>_<biến_thể>`, ví dụ `tfidf_lr_segmented_uni`, `phobert_segmented`.
Kết quả lưu ở `results/<tên>_metrics.json` và `results/<tên>_cm.png`.

## Kết quả baseline (TF-IDF unigram + Logistic Regression, có tách từ, tập test)

| Cấu hình | Accuracy | Macro-F1 | F1 neutral |
|---|---|---|---|
| C=1 mặc định | 88,57% | 0,6455 | 0,1196 |
| Tuning theo macro-F1 (C=10) | 89,10% | 0,7258 | 0,3525 |

Ở đây C=10 được chọn bằng cross-validation trên tập train (`GridSearchCV`, chấm theo macro-F1).
Lớp neutral là điểm yếu chính: với C=1 mô hình gần như không nhận ra câu trung tính (recall 6,6%),
sau tuning recall lên 25,8% nhưng F1 vẫn chỉ 0,35.

## Thí nghiệm (TF-IDF + Logistic Regression, tập test)

Ở các thí nghiệm dưới đây, C được chọn theo macro-F1 trên **tập dev** (lưới C: 0,1 / 1 / 10 / 100 / 1000), nên số của
`segmented_uni` (0,7111) khác con số baseline C=10 (0,7258) ở trên. Hai cách chọn C cho kết quả hơi khác nhau,
đó là mức nhiễu bình thường vì lớp neutral chỉ có 167 câu test. `max_features=5000`.

Tóm tắt:
- Có hay không tách từ, unigram hay bigram: chênh nhau khoảng 0,01, chưa thấy khác biệt rõ ràng.
- Bỏ dấu làm macro-F1 giảm mạnh (từ khoảng 0,71 xuống 0,28 đến 0,38); từ điển khôi phục được một phần.
- Huấn luyện thêm bản không dấu gần như xóa được mức giảm này, mà gần như không ảnh hưởng đến văn bản có dấu.
- Trên văn bản có dấu của VSFC, teencode không tạo khác biệt đo được.

### Tách từ × n-gram (macro-F1)

| | Unigram | Bigram |
|---|---|---|
| Có tách từ | 0,7111 | 0,7145 |
| Không tách từ | 0,7242 | 0,7174 |

Bốn cấu hình chênh nhau khoảng 0,01 (lớp neutral chỉ có 167 câu test), nên chưa thấy khác biệt rõ ràng giữa có và không tách từ.

### Văn bản không dấu (macro-F1, mô hình chỉ học trên văn bản có dấu)

Tập test được bỏ dấu tự động. "Không từ điển": chỉ làm sạch tối thiểu. "Có từ điển": cho qua bước làm sạch đầy đủ (từ điển teencode và không dấu).

| Cấu hình | Có dấu | Không dấu, không từ điển | Không dấu, có từ điển |
|---|---|---|---|
| segmented_uni | 0,7111 | 0,3773 | 0,4882 |
| segmented_bi | 0,7145 | 0,3727 | 0,5279 |
| raw_uni | 0,7242 | 0,2759 | 0,3664 |
| raw_bi | 0,7174 | 0,3549 | 0,4966 |

### Huấn luyện thêm bản không dấu (macro-F1)

Dữ liệu huấn luyện gồm bản có dấu và bản không dấu (qua từ điển) của mỗi câu, nhãn giữ nguyên. Chấm trên cùng 3 phiên bản tập test như bảng trên.

| Cấu hình | Có dấu | Không dấu, không từ điển | Không dấu, có từ điển |
|---|---|---|---|
| segmented_uni | 0,7050 | 0,6620 | 0,6891 |
| segmented_bi | 0,7054 | 0,6679 | 0,7054 |
| raw_uni | 0,7248 | 0,6800 | 0,7078 |
| raw_bi | 0,7147 | 0,6868 | 0,7160 |

Bản không dấu của tập test được tạo bằng cách bỏ dấu tự động, nên sạch hơn văn bản thật; kết quả là cận trên.

### Ảnh hưởng của teencode trên dữ liệu gốc (macro-F1, tập test)

Bước teencode (từ điển + rút gọn chữ lặp) chỉ thay đổi 15/11.426 câu train (0,13%).
Khoảng tin cậy 95% lấy từ 1000 lần lấy mẫu lại tập test.

| Cấu hình | Không teencode | Có teencode | Chênh lệch (khoảng tin cậy 95%) |
|---|---|---|---|
| segmented_uni | 0,7121 | 0,7111 | −0,0010 [−0,0055; +0,0040] |
| segmented_bi | 0,7097 | 0,7145 | +0,0048 [−0,0089; +0,0195] |
| raw_uni | 0,7243 | 0,7242 | −0,0001 [−0,0080; +0,0070] |
| raw_bi | 0,7179 | 0,7174 | −0,0004 [−0,0086; +0,0077] |

Mọi khoảng đều chứa 0: trên văn bản có dấu của VSFC, teencode không tạo khác biệt đo được.

### Các file kết quả trong `results/`

| File | Nội dung |
|---|---|
| `tfidf_lr_*_metrics.json`, `tfidf_lr_*_cm.png` | Chỉ số và ma trận nhầm lẫn của từng cấu hình |
| `ablation_2x2.csv` | Bảng tách từ × n-gram |
| `unaccented_drop.csv` | Bảng văn bản không dấu |
| `augmented_unaccented.csv` | Bảng huấn luyện thêm bản không dấu |
| `teencode_effect.csv` | Bảng ảnh hưởng của teencode |

## Cài đặt

```
pip install datasets==2.19.0 py_vncorenlp scikit-learn pandas matplotlib seaborn
```

`py_vncorenlp` cần Java. Mô hình VnCoreNLP tải về thư mục `vncorenlp_fresh/` (đã bỏ qua trong git).
Mỗi phiên Python chỉ khởi động VnCoreNLP được một lần; nếu gặp lỗi `VM is already running`, hãy khởi động lại phiên rồi chạy lại.
