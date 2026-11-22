# Đề tài 6: Phân tích cảm xúc văn bản tiếng Việt

So sánh TF-IDF + Logistic Regression, BiLSTM và PhoBERT; đánh giá ảnh hưởng của tách từ, teencode và văn bản không dấu.

## Dữ liệu 

Dữ liệu chính: **UIT-VSFC** (phản hồi sinh viên), dùng đúng split gốc của Hugging Face
(`uitnlp/vietnamese_students_feedback`).

| File | Số câu |
|---|---|
| `train_segmented.csv` | 11.426 |
| `dev_segmented.csv` (validation) | 1.583 |
| `test_segmented.csv` | 3.166 |

### Các cột

| Cột | Ý nghĩa |
|---|---|
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

## Đánh giá thống nhất: `evaluate.py`

Mọi mô hình dùng chung một hàm để so sánh công bằng:

```python
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

Lớp neutral là điểm yếu chính: với C=1 mô hình gần như không nhận ra câu trung tính (recall 6,6%),
sau tuning recall lên 25,8% nhưng F1 vẫn chỉ 0,35.

## Cài đặt

```
pip install datasets==2.19.0 py_vncorenlp scikit-learn pandas matplotlib seaborn
```

`py_vncorenlp` cần Java. Mô hình VnCoreNLP tải về thư mục `vncorenlp_fresh/` (đã bỏ qua trong git).
