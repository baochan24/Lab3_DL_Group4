# BÁO CÁO LAB 3 — PHÂN TÍCH CẢM XÚC VÀ FINE-TUNING DISTILBERT
**Học phần:** Deep Learning · **Nhóm:** 4

## Mục lục
1. [Mục tiêu và phạm vi](#1-mục-tiêu-và-phạm-vi)
2. [Môi trường thực nghiệm](#2-môi-trường-thực-nghiệm)
3. [Cơ sở kỹ thuật](#3-cơ-sở-kỹ-thuật)
4. [Exercise 1: suy luận cảm xúc](#4-exercise-1-suy-luận-cảm-xúc)
5. [Exercise 2: dữ liệu và tiền xử lý](#5-exercise-2-dữ-liệu-và-tiền-xử-lý)
6. [Exercise 2: mô hình và huấn luyện](#6-exercise-2-mô-hình-và-huấn-luyện)
7. [Đánh giá test và phân tích kết quả](#7-đánh-giá-test-và-phân-tích-kết-quả)
8. [Dự đoán mới, lưu kết quả và xuất mô hình](#8-dự-đoán-mới-lưu-kết-quả-và-xuất-mô-hình)
9. [Đối chiếu triển khai và các giới hạn](#9-đối-chiếu-triển-khai-và-các-giới-hạn)
10. [Hướng dẫn sử dụng và kết luận](#10-hướng-dẫn-sử-dụng-và-kết-luận)
11. [Phụ lục truy vết notebook](#11-phụ-lục-truy-vết-notebook)

## 1. Mục tiêu và phạm vi

Lab gồm hai exercise liên quan đến phân loại cảm xúc văn bản tiếng Anh thành hai lớp: **NEGATIVE (0)** và **POSITIVE (1)**.

| Nội dung | Exercise 1 | Exercise 2 |
|---|---|---|
| Notebook | [Sentiment Analysis](../Lab3_Exercise1_Sentiment_Analysis.ipynb) | [FineTunning](../Lab3_Exercise2_FineTunning.ipynb) |
| Mô hình khởi đầu | DistilBERT đã fine-tune trên SST-2 | DistilBERT pretrained cơ sở |
| Công việc | Tải mô hình, tokenize, forward, giải thích dự đoán | Chuẩn bị IMDb, fine-tune, chọn checkpoint, đánh giá |
| Cập nhật trọng số | Không | Có, toàn bộ tham số được phép học |
| Bằng chứng kết quả | Output một câu trong notebook | Output huấn luyện, 200 mẫu test, 6 file báo cáo |

EX1 giúp hiểu cách sử dụng mô hình đã huấn luyện. EX2 thực hành điều chỉnh mô hình pretrained theo dữ liệu có nhãn. Hai exercise không tạo chatbot và không thực hiện sinh văn bản.

**Căn cứ báo cáo:** code, Markdown và output đã lưu tại commit nguồn `3db8be65c1ad06feb7ed11909e9dd40380a6d0de` của repo. Đây là báo cáo phân tích hiện vật thực nghiệm; không phải biên bản chạy lại notebook trên một môi trường sạch. Bản đề gốc không có trong repo ở thời điểm đối chiếu, nên báo cáo không tự bổ sung yêu cầu của giảng viên. Các số thứ tự cell ở phụ lục được đếm từ 1, bao gồm cả Markdown.

## 2. Môi trường thực nghiệm

Hai notebook lưu output từ hai môi trường khác nhau; không gộp chúng thành một cấu hình duy nhất.

| Thành phần | EX1 | EX2 |
|---|---|---|
| Python | Metadata notebook: 3.13.7 | Output: 3.11.9 |
| Môi trường | `.venv` | `.venv311` |
| PyTorch | 2.14.1+cpu | 2.14.1+cpu |
| Transformers | 5.21.0 | 4.57.1 |
| CUDA | False | CPU |
| Datasets | Không in phiên bản | 3.3.2 |
| Evaluate | Không dùng trong code EX1 | 0.4.6 |
| Accelerate | Không in phiên bản | 1.11.0 |
| Scikit-learn | Không dùng trong code EX1 | 1.9.1 |

Các phiên bản trên được chép từ output/metadata đã lưu, không phải khuyến nghị rằng đây là bộ phiên bản mới nhất. EX2 có import và kiểm tra `evaluate`, nhưng các metric thực tế được tính bằng **scikit-learn**.

Notebook có cell cài đặt thư viện và cell in thông tin để phát hiện việc chọn sai kernel. Các cảnh báo symlink cache trên Windows và thiếu IProgress xuất hiện ở output EX1; output sau đó vẫn xác nhận tải mô hình và dự đoán thành công. EX2 có cell cài bổ sung `matplotlib` trước khi xuất biểu đồ.

## 3. Cơ sở kỹ thuật

### 3.1. Từ văn bản đến dự đoán

Quy trình chung:

1. Tokenizer tách văn bản thành token theo bộ từ vựng của mô hình.
2. Chuyển token thành `input_ids`, bổ sung token đặc biệt và tạo `attention_mask`.
3. Mô hình tính biểu diễn ngữ cảnh và đưa qua đầu phân loại.
4. Đầu ra `logits` có hai điểm số cho mỗi mẫu.
5. Softmax chuyển logits thành phân phối hai lớp; argmax chọn nhãn.

Với batch B mẫu, chiều dài sau padding L và hai nhãn, các kích thước cần theo dõi là:

| Tensor | Kích thước | Ý nghĩa |
|---|---|---|
| input_ids | B × L | Mã token |
| attention_mask | B × L | Vị trí token hợp lệ/padding |
| labels khi huấn luyện | B | Nhãn thật |
| logits | B × 2 | Điểm số hai lớp |
| probabilities | B × 2 | Kết quả softmax |

Mã token là chỉ số trong từ vựng, không phải điểm cảm xúc. Attention mask không phải ma trận trọng số attention mà mô hình học.

### 3.2. Logits, softmax và hàm mất mát

Với logits z₀, z₁:

`p(k) = exp(z_k) / [exp(z_0) + exp(z_1)]`.

Nhãn dự đoán là lớp có xác suất lớn nhất. Khi chỉ cần argmax, có thể lấy trực tiếp từ logits vì softmax giữ thứ tự hai điểm số. Notebook dùng cách này khi tính metric theo batch.

Trong phân loại một nhãn với hai logits và nhãn nguyên 0/1, mô hình tính cross-entropy: `loss = -log p(nhãn thật)`. Trainer sử dụng loss để backpropagation và cập nhật trọng số; precision/recall/F1 chỉ dùng để đánh giá.

Điểm softmax cao là mức tự tin của mô hình trong hai lớp đã định nghĩa, không phải bảo đảm xác suất đúng ngoài thực tế. Lab chưa thực hiện hiệu chỉnh xác suất.

### 3.3. Suy luận và fine-tuning

`model.eval()` chuyển các lớp như dropout sang hành vi đánh giá. `torch.no_grad()` tắt ghi đồ thị gradient để giảm chi phí khi suy luận. Hai thao tác có vai trò khác nhau và được dùng cùng nhau trong EX1 cũng như hàm dự đoán mới EX2.

Fine-tuning EX2 giữ kiến thức pretrained làm điểm khởi đầu nhưng cho phép gradient cập nhật encoder và đầu phân loại. Không có đoạn đóng băng encoder trong notebook đã kiểm tra.

## 4. Exercise 1: suy luận cảm xúc

### 4.1. Tải mô hình

Notebook dùng:

`distilbert/distilbert-base-uncased-finetuned-sst-2-english`

Tokenizer và mô hình đều tải từ cùng tên bằng `AutoTokenizer.from_pretrained` và `AutoModelForSequenceClassification.from_pretrained`. Đây là checkpoint đã có đầu phân loại cảm xúc. Output ánh xạ nhãn là `{0: 'NEGATIVE', 1: 'POSITIVE'}`.

EX1 không tải IMDb, không tạo Trainer, không tính loss huấn luyện và không cập nhật trọng số.

### 4.2. Tokenization của câu minh họa

Câu đầu vào:

> I love this movie. It is amazing!

Cấu hình: `return_tensors="pt"`, `truncation=True`, `max_length=512`.

| Vị trí | Token | ID |
|---:|---|---:|
| 0 | [CLS] | 101 |
| 1 | i | 1045 |
| 2 | love | 2293 |
| 3 | this | 2023 |
| 4 | movie | 3185 |
| 5 | . | 1012 |
| 6 | it | 2009 |
| 7 | is | 2003 |
| 8 | amazing | 6429 |
| 9 | ! | 999 |
| 10 | [SEP] | 102 |

Output `input_ids.shape = [1, 11]`: một mẫu và 11 token, tính cả token đặc biệt. Mô hình uncased xử lý chữ thường theo tokenizer. Attention mask gồm 11 giá trị 1 vì ví dụ không có padding. `max_length=512` là giới hạn trên, không ép câu ngắn thành 512 token.

### 4.3. Forward và kết quả

Trong `torch.no_grad()`, notebook gọi `model(**inputs)`, áp dụng softmax theo chiều cuối rồi chọn argmax.

| Đại lượng | Kết quả đã lưu |
|---|---|
| Logits | [-4.3436, 4.7032] |
| Xác suất NEGATIVE | 0,0118% |
| Xác suất POSITIVE | 99,9882% |
| Nhãn dự đoán | POSITIVE |

Kết quả phù hợp với nội dung lời khen. Đây là kiểm tra suy luận của một câu; không thể gọi 99,9882% là accuracy của mô hình. EX1 không có tập test có nhãn để tính accuracy, precision, recall hay F1.

## 5. Exercise 2: dữ liệu và tiền xử lý

### 5.1. Nguồn dữ liệu

Code tải hai split có nhãn từ `stanfordnlp/imdb`:

| Split gốc | Số mẫu | NEGATIVE | POSITIVE |
|---|---:|---:|---:|
| train | 25.000 | 12.500 | 12.500 |
| test | 25.000 | 12.500 | 12.500 |

Mỗi mẫu gồm `text` và `label`; output ClassLabel là `neg/pos`. Code không sử dụng split không có nhãn.

### 5.2. Chia tập và lấy mẫu cân bằng

`SEED = 42`. Split train gốc được chia với `test_size=0.1`, `stratify_by_column="label"`: train pool 22.500 và validation pool 2.500.

Hàm `make_balanced_subset` lọc riêng từng nhãn, shuffle, chọn số lượng cần thiết, kiểm tra đủ mẫu, ghép hai lớp rồi shuffle lại:

| Tập con ban đầu | Cách lấy | Tổng | Phân bố |
|---|---|---:|---|
| train_dataset | train pool, 500/lớp, seed 42 | 1.000 | 500/500 |
| val_dataset | validation pool, 100/lớp, seed 43 | 200 | 100/100 |
| test_dataset | test gốc, 250/lớp, seed 44 | 500 | 250/250 |

Train và validation lấy từ hai phần tách biệt của train gốc; test lấy từ test gốc. Cách chia này tách vai trò học, chọn checkpoint và đánh giá cuối. Báo cáo chưa có kiểm tra bổ sung về văn bản trùng nội dung giữa các split.

### 5.3. Tokenization và padding

Tokenizer EX2 đi cùng `distilbert/distilbert-base-uncased`. Hàm tiền xử lý dùng:

```python
tokenizer(examples["text"], truncation=True, max_length=256)
```

Mỗi tập được `.map(..., batched=True, remove_columns=["text"])` riêng. Các cột sau xử lý là `label`, `input_ids`, `attention_mask`. Loại cột text khỏi tập tokenized không làm mất nhãn, cũng không xóa văn bản trong tập gốc.

`DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")` bổ sung padding khi tạo batch, tới chiều dài mẫu dài nhất trong batch. Điều này tránh ép mọi mẫu ngắn lên 256 token. Bản ghi dài hơn giới hạn bị cắt; thông tin cảm xúc ở phần cuối có thể bị bỏ qua.

Không có bước tự viết để xóa HTML, bỏ stopword hay stemming. Output mẫu vẫn có các token từ `<br />`; báo cáo không mô tả những bước làm sạch mà code chưa thực hiện.

### 5.4. Điểm quan trọng về thứ tự cell

Cell 18 giảm các **biến gốc** xuống 400 train, 100 validation và 200 test sau khi đã tạo các tập tokenized. Các phép shuffle/select trả về dataset mới, không tự thay đổi dataset đã tokenization trước đó.

| Biến/thời điểm | Số mẫu | Vai trò thực tế |
|---|---:|---|
| train_dataset sau cell 18 | 400 | Không được truyền trực tiếp vào Trainer |
| val_dataset sau cell 18 | 100 | Không được truyền trực tiếp vào Trainer |
| tokenized_train | 1.000 | Tập huấn luyện thực tế |
| tokenized_val | 200 | Tập validation thực tế |
| tokenized_test cũ | 500 | Không dùng cho đánh giá cuối |
| test_dataset sau cell 18 | 200 | Nguồn test cuối |
| tokenized_test_final | 200 | Tập test truyền vào predict cuối |

Vì vậy, kết quả của lab phải báo cáo **1.000 train / 200 validation / 200 test**, không ghi train/validation là 400/100.

Chẩn đoán ở cell 21–22 phát hiện raw test 200 nhưng tokenized test 500. Cell 23 tạo lại `tokenized_test_final` từ đúng `test_dataset`, kiểm tra số mẫu và các cột bằng assert, cho output `TEST_DATASET_VALIDATION_OK`.

Tập test 200 được chọn bằng shuffle/select từ tập 500 cân bằng, không lấy lại số mẫu bằng nhau theo lớp, nên phân bố cuối là **89 NEGATIVE / 111 POSITIVE**. Đây không phải tập test cân bằng 100/100.

## 6. Exercise 2: mô hình và huấn luyện

### 6.1. Khởi tạo mô hình

Mô hình: `distilbert/distilbert-base-uncased`, `num_labels=2`, ánh xạ 0/1 như trên.

Output cảnh báo `pre_classifier` và `classifier` được khởi tạo mới. Điều này phù hợp khi tải encoder pretrained cơ sở để làm nhiệm vụ phân loại mới: cần huấn luyện đầu phân loại trước khi sử dụng kết quả.

| Tham số | Output |
|---|---:|
| Vocabulary size | 30.522 |
| Tổng số tham số | 66.955.010 |
| Số tham số có requires_grad=True | 66.955.010 |

Đây là **fine-tuning toàn bộ mô hình theo cấu hình code**, không phải chỉ học classifier trên encoder đóng băng. Số trainable parameters cho biết tham số được phép cập nhật, không chứng minh từng giá trị đều thay đổi sau mỗi batch.

### 6.2. TrainingArguments

| Cấu hình | Giá trị | Mục đích |
|---|---|---|
| output_dir | ./results_imdb | Lưu checkpoint |
| num_train_epochs | 2 | Hai lượt qua tập train |
| Train/eval batch size | 8/8 | Số mẫu mỗi batch trên CPU |
| learning_rate | 2e-5 | Learning rate khởi đầu |
| weight_decay | 0.01 | Regularization trong optimizer |
| optim | adamw_torch | AdamW của PyTorch |
| eval_strategy | epoch | Validation sau mỗi epoch |
| save_strategy | epoch | Lưu theo epoch |
| load_best_model_at_end | True | Nạp checkpoint tốt nhất khi kết thúc |
| metric_for_best_model | accuracy | Tiêu chí lựa chọn |
| greater_is_better | True | Ưu tiên accuracy cao |
| logging_steps | 25 | Ghi log theo bước |
| save_total_limit | 1 | Giới hạn lưu checkpoint theo cơ chế Trainer |
| report_to | none | Không gửi log tới dịch vụ theo dõi |
| seed / data_seed | 42 / 42 | Seed trong TrainingArguments |
| use_cpu | True | Huấn luyện CPU |

Seed cho chia dữ liệu và Trainer đã có trong code. Tuy nhiên mô hình với đầu phân loại ngẫu nhiên được tạo **trước** Trainer và notebook không đặt seed toàn cục ngay trước khi tạo model. Vì vậy không khẳng định seed 42 đủ để tái lập tuyệt đối trọng số đầu phân loại và các chỉ số.

### 6.3. Trainer và vòng học

Trainer nhận `model`, `training_args`, `tokenized_train`, `tokenized_val`, collator, `compute_metrics` và `processing_class=tokenizer`.

Mỗi bước huấn luyện về nguyên tắc gồm lấy batch → forward có labels → tính cross-entropy → backward → optimizer cập nhật trọng số. Trainer điều phối các bước này, thực hiện validation và lưu checkpoint theo cấu hình; notebook không tự viết training loop.

Hàm `compute_metrics` lấy logits, xử lý trường hợp predictions là tuple, argmax rồi gọi scikit-learn. Precision/recall/F1 dùng `average="binary"`, tức lớp dương 1 = POSITIVE; `zero_division=0` xử lý trường hợp mẫu số bằng 0.

Với 1.000 mẫu và batch size 8, có 125 batch/epoch. Output 250 bước sau 2 epoch khớp với kích thước tập tokenized. Thanh tiến trình lưu thời gian khoảng **33 phút 06 giây** cho lần chạy này; không coi đây là thời gian chuẩn cho mọi máy.

### 6.4. Kết quả huấn luyện và chọn checkpoint

| Epoch | Training loss hiển thị | Validation loss | Val accuracy | Val precision (+) | Val recall (+) | Val F1 (+) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.378100 | 0.283972 | 0.905000 | 0.909091 | 0.900000 | 0.904523 |
| 2 | 0.176000 | 0.299779 | 0.895000 | 0.907216 | 0.880000 | 0.893401 |

Training loss trong bảng là giá trị log hiển thị tại mốc epoch, không mặc định diễn giải là trung bình toàn epoch. `train_result.training_loss` của toàn lượt train là **0.3713298969**.

Loss train hiển thị giảm, trong khi validation loss tăng và validation accuracy giảm 1 điểm phần trăm. Diễn biến gợi ý khả năng bắt đầu overfit; hai mốc epoch chưa đủ cho một kết luận mạnh về xu hướng tổng quát.

Tiêu chí chọn là validation accuracy, không phải test accuracy hoặc training loss. Epoch 1 tốt hơn, nên checkpoint **checkpoint-125** được chọn; output chẩn đoán xác nhận đường dẫn tồn tại. Epoch 2 vẫn đã được chạy đủ, nhưng không phải trọng số được chọn để báo cáo cuối.

## 7. Đánh giá test và phân tích kết quả

### 7.1. Quy trình

Cell đánh giá lấy đường dẫn từ `trainer.state.best_model_checkpoint`, kiểm tra tồn tại, nạp `best_model` rồi gán cho `trainer.model`, chuyển CPU và gọi `trainer.predict(tokenized_test_final)`. Nhãn thật lấy từ `test_output.label_ids`, nhãn dự đoán lấy bằng argmax.

Output đã lưu thể hiện 200 mẫu. Báo cáo không sử dụng kết quả từ `tokenized_test` cũ 500 mẫu. Việc thay model trực tiếp trên Trainer là cách notebook đã chạy; báo cáo chưa kiểm tra lại cách này trong một phiên mới hoặc các phiên bản Transformers khác.

### 7.2. Ma trận nhầm lẫn

| Thực tế / Dự đoán | NEGATIVE | POSITIVE | Tổng |
|---|---:|---:|---:|
| NEGATIVE | 77 | 12 | 89 |
| POSITIVE | 10 | 101 | 111 |
| Tổng | 87 | 113 | 200 |

![Confusion matrix](../outputs/exercise2/confusion_matrix.png)

Lấy POSITIVE làm lớp dương: TP = 101, TN = 77, FP = 12, FN = 10. Mô hình đúng 178 mẫu và sai 22 mẫu.

| Chỉ số | Phép tính từ ma trận | Giá trị |
|---|---|---:|
| Accuracy | (77 + 101) / 200 | 89,00% |
| Precision POSITIVE | 101 / (101 + 12) | 89,38% |
| Recall POSITIVE | 101 / (101 + 10) | 90,99% |
| F1 POSITIVE | 2 × 101 / (2 × 101 + 12 + 10) | 90,18% |

Precision trả lời: trong các mẫu được dự đoán tích cực, bao nhiêu thực sự tích cực? Recall trả lời: trong các mẫu thực sự tích cực, mô hình phát hiện được bao nhiêu? F1 là trung bình điều hòa của precision và recall.

### 7.3. Báo cáo theo lớp

Nguồn: [classification_report.txt](../outputs/exercise2/classification_report.txt).

| Lớp / trung bình | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NEGATIVE | 0.8851 | 0.8652 | 0.8750 | 89 |
| POSITIVE | 0.8938 | 0.9099 | 0.9018 | 111 |
| Macro avg | 0.8894 | 0.8875 | 0.8884 | 200 |
| Weighted avg | 0.8899 | 0.8900 | 0.8899 | 200 |

Macro average cho hai lớp trọng số bằng nhau; weighted average cân theo support. **F1 90,18% là F1 của lớp POSITIVE**, không phải macro-F1 (88,84%).

Recall lớp tiêu cực thấp hơn lớp tích cực trong tập này. Có 12 review tiêu cực bị đoán tích cực và 10 review tích cực bị đoán tiêu cực. Chưa có phân tích gán nguyên nhân từng lỗi, nên không khẳng định các lỗi chắc chắn do mỉa mai, phủ định hay cắt token.

Tập test có 111/200 nhãn tích cực; nếu luôn đoán POSITIVE thì accuracy là 55,5%. Đây là mốc tính trực tiếp từ phân bố nhãn để đọc kết quả 89%, không phải một mô hình baseline đã được huấn luyện hoặc thử nghiệm riêng.

### 7.4. Giới hạn diễn giải

Kết quả chỉ áp dụng cho 200 mẫu được chọn. Chưa đánh giá toàn bộ 25.000 mẫu test, nhiều seed, cross-validation, khoảng tin cậy hay dữ liệu ngoài IMDb. Không có đánh giá EX1 trên cùng tập test này, vì vậy không được kết luận EX2 tốt hơn EX1 dựa trên xác suất của các câu minh họa.

## 8. Dự đoán mới, lưu kết quả và xuất mô hình

### 8.1. Bốn câu minh họa

Hàm `predict_sentiment` dùng `best_model.eval()`, CPU, tokenization tối đa 256 token, `torch.no_grad()`, softmax và ánh xạ nhãn. Xác suất được làm tròn 4 chữ số khi lưu.

| Câu | Nhãn | P(NEGATIVE) | P(POSITIVE) |
|---|---|---:|---:|
| This movie is wonderful. I really enjoyed it! | POSITIVE | 0.0937 | 0.9063 |
| This film is terrible and completely boring. | NEGATIVE | 0.8998 | 0.1002 |
| The story was interesting, and the actors did a great job. | POSITIVE | 0.1164 | 0.8836 |
| I regret watching this movie. It was a waste of time. | NEGATIVE | 0.9051 | 0.0949 |

Bốn nhãn phù hợp với nội dung mong đợi. Đây là demo chức năng, không phải một phép đánh giá tổng quát với đủ độ khó hoặc độ đại diện.

### 8.2. Các file đầu ra

| File trong outputs/exercise2 | Nội dung và công dụng |
|---|---|
| [test_metrics.json](../outputs/exercise2/test_metrics.json) | Số mẫu, bốn chỉ số, đường dẫn best checkpoint |
| [classification_report.txt](../outputs/exercise2/classification_report.txt) | Metric theo lớp và trung bình |
| [confusion_matrix.csv](../outputs/exercise2/confusion_matrix.csv) | Ma trận có nhãn hàng/cột |
| [confusion_matrix.png](../outputs/exercise2/confusion_matrix.png) | Hình đưa vào báo cáo |
| [test_predictions.csv](../outputs/exercise2/test_predictions.csv) | Text, true_label, predicted_label, correct của từng mẫu |
| [sample_predictions.csv](../outputs/exercise2/sample_predictions.csv) | Bốn câu và xác suất hai lớp |

Code kiểm tra số dòng text và các mảng nhãn bằng nhau trước khi xuất CSV. Các file CSV dùng UTF-8 BOM để thuận tiện mở bằng ứng dụng bảng tính. Hình được tạo với Matplotlib và ConfusionMatrixDisplay, lưu 200 dpi.

### 8.3. Xuất mô hình

Cell 28 thiết lập ánh xạ nhãn, gọi `best_model.save_pretrained(..., safe_serialization=True)` và `tokenizer.save_pretrained(...)` tới:

`outputs/exercise2/final_model/`

Output `MODEL_EXPORT_OK` kèm danh sách:

- config.json
- model.safetensors
- tokenizer.json
- tokenizer_config.json
- special_tokens_map.json
- vocab.txt
- inference_settings.json

`inference_settings.json` ghi `max_length=256` và `truncation=true`. Khi dùng lại cần đọc hoặc áp dụng cấu hình này; `from_pretrained` không tự diễn giải file tùy chỉnh này.

Notebook ghi nhận kết luận tải lại thành công ở tiến trình riêng, và người thực hiện xác nhận trong quá trình làm bài. Repo không có log terminal độc lập của bước đó. Bằng chứng trực tiếp trong output code của notebook là việc xuất mô hình thành công.

Trọng số xuất ra và checkpoint được loại khỏi Git. Clone repo không đồng nghĩa đã có mô hình fine-tune cục bộ.

## 9. Đối chiếu triển khai và các giới hạn

### 9.1. Bằng chứng cho các bước EX2

Bảng này đối chiếu quy trình triển khai trong notebook, không thay thế việc đọc đề gốc.

| Bước | Hiện vật xác minh |
|---|---|
| Cài/kiểm tra transformers, datasets, evaluate | Cell 2–4, phiên bản in thành công |
| Tải dữ liệu nhị phân | Cell 6–8: IMDb, nhãn 0/1 |
| Tải pretrained và tokenizer | Cell 10: base DistilBERT, 2 nhãn |
| Tiền xử lý | Cell 12 và 23: map, truncation, collator, test cuối |
| TrainingArguments | Cell 16 |
| Trainer và fine-tune | Cell 19–20: 1.000/200, 250 bước |
| Đánh giá | Cell 24 và các file metrics/report |

Dự đoán câu mới và xuất mô hình là phần hoàn thiện pipeline thể hiện trong notebook. Không tự coi đây là yêu cầu bắt buộc nếu đề không ghi.

### 9.2. Các điểm cần đọc đúng khi trình bày

- Không gọi EX1 là tự huấn luyện DistilBERT: EX1 chỉ suy luận.
- Không gọi EX2 là huấn luyện từ đầu: encoder tải pretrained.
- Không ghi encoder bị đóng băng: output cho thấy toàn bộ tham số trainable.
- Không ghi train thực tế là 400 mẫu: Trainer dùng tokenized_train 1.000 mẫu.
- Không ghi test cuối là 500 mẫu hoặc cân bằng 100/100.
- Không gọi best checkpoint là epoch 2: checkpoint được chọn ở bước 125.
- Không gọi F1 positive là macro-F1.
- Không coi xác suất câu ví dụ là accuracy.

### 9.3. Hạn chế tái lập và phạm vi rà soát

Notebook còn cell chẩn đoán và các biến gốc/tokenized có kích thước khác nhau. Phần sửa test cuối đã làm rõ dữ liệu được đánh giá, nhưng người chạy phải theo đúng thứ tự và không tùy ý chạy lại các cell shuffle/select nhiều lần trên biến đã bị thu nhỏ.

Repo chưa có file khóa đầy đủ dependencies; các cell pip không cố định toàn bộ phiên bản. Mô hình được khởi tạo trước khi seed của Trainer được áp dụng. Chưa chạy lại notebook từ đầu trên môi trường sạch trong lần viết báo cáo này. Vì vậy kết quả được xác nhận bằng code và output đã lưu, không bảo đảm một lần chạy mới cho đúng từng chữ số.

Báo cáo này không sửa cấu hình, trọng số hay các notebook và không tạo thêm kết quả huấn luyện.

## 10. Hướng dẫn sử dụng và kết luận

### 10.1. Đọc hoặc tái chạy

Đọc hai notebook trên GitHub và đối chiếu các file trong `outputs/exercise2/` để xem kết quả mà không cần huấn luyện lại. Hướng dẫn môi trường ở [README](../README.md#5-chuẩn-bị-và-chạy).

Khi chạy bằng VS Code/Jupyter, chọn đúng kernel; cài dependencies trước khi import; chạy từ trên xuống; cần Internet cho lần tải đầu mô hình/dataset. Cell `trainer.train()` huấn luyện lại và có thể tốn tài nguyên. Các cell evaluate/export cần model hoặc checkpoint có sẵn trong phiên làm việc.

Có thể tải lại bản đã xuất bằng đoạn tham khảo sau, khi thư mục final_model có trên máy; đoạn này là hướng dẫn sử dụng, không phải một output thực nghiệm mới của báo cáo:

```python
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = Path("outputs/exercise2/final_model")
settings = json.loads(
    (model_dir / "inference_settings.json").read_text(encoding="utf-8")
)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir, local_files_only=True
).to("cpu")
model.eval()

inputs = tokenizer(
    "This movie is wonderful. I really enjoyed it!",
    return_tensors="pt",
    **settings,
)
with torch.no_grad():
    probabilities = model(**inputs).logits.softmax(dim=-1)[0]
label_id = int(probabilities.argmax().item())
print(model.config.id2label[label_id], probabilities[label_id].item())
```

### 10.2. Kết luận

EX1 hoàn thành tải checkpoint sentiment, quan sát token/ID/mask và suy luận câu minh họa thành POSITIVE. EX2 thể hiện pipeline fine-tuning đầy đủ: tải và chia IMDb, tokenization, collator, TrainingArguments, Trainer, chọn checkpoint theo validation, đánh giá test, dự đoán mới và xuất mô hình.

Lần chạy được báo cáo huấn luyện trên **1.000 mẫu**, validation **200 mẫu**, thực hiện **2 epoch/250 bước** trên CPU. Checkpoint được chọn là **checkpoint-125**. Trên **200 mẫu test**, mô hình phân loại đúng **178 mẫu**, accuracy **89,00%**, precision POSITIVE **89,38%**, recall POSITIVE **90,99%**, F1 POSITIVE **90,18%**. Kết luận giới hạn trong tập con và cấu hình đã lưu.

## 11. Phụ lục truy vết notebook

### EX1

| Cell | Vai trò |
|---|---|
| 1 | Cài transformers |
| 2 | In phiên bản và CUDA |
| 3 | Tải tokenizer/model SST-2, eval và ánh xạ nhãn |
| 4 | Kiểm tra executable và tích hợp PyTorch |
| 5 | Tokenize câu, in tokens/IDs/mask/shape |
| 6 | Forward, softmax, nhãn và confidence |

### EX2

| Cell | Vai trò |
|---|---|
| 1 | Mục tiêu và workflow |
| 2–4 | Môi trường, cài dependencies, kiểm tra phiên bản/CPU |
| 5–8 | Giải thích dữ liệu, tải IMDb, chia pool và lấy subset |
| 9–10 | Pretrained model, tokenizer, nhãn, tham số |
| 11–12 | Giải thích và thực hiện tokenization/collator |
| 13–14 | Khai báo metric |
| 15–16 | TrainingArguments |
| 17–18 | Giới thiệu fine-tune và cell giảm biến dữ liệu gốc |
| 19 | Tạo Trainer trên tokenized_train/tokenized_val |
| 20 | Huấn luyện, log hai epoch và 250 bước |
| 21–22 | Chẩn đoán checkpoint và bất nhất test 200/500 |
| 23 | Tạo lại tokenized_test_final 200 mẫu |
| 24 | Nạp best checkpoint và đánh giá test |
| 25 | Dự đoán bốn câu mới |
| 26 | Cài matplotlib |
| 27 | Lưu sáu file kết quả |
| 28 | Xuất model/tokenizer và inference settings |
| 29 | Tổng kết trong notebook |

**Nguồn nội bộ:** [EX1](../Lab3_Exercise1_Sentiment_Analysis.ipynb), [EX2](../Lab3_Exercise2_FineTunning.ipynb), [metrics](../outputs/exercise2/test_metrics.json), [classification report](../outputs/exercise2/classification_report.txt), [confusion matrix](../outputs/exercise2/confusion_matrix.csv), [sample predictions](../outputs/exercise2/sample_predictions.csv). Các liên kết sử dụng đường dẫn tương đối trong chính repo.
