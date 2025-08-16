Vietnamese Fake News Detection
Tokenization from Pre-trained Models + Word Embeddings BiLSTM

Phát hiện tin giả trên mạng xã hội là nhiệm vụ quan trọng nhằm đảm bảo tính toàn vẹn của thông tin. Trong nghiên cứu này, chúng tôi đề xuất một phương pháp phát hiện tin giả tiếng Việt kết hợp tokenization từ mô hình tiền huấn luyện (Bartpho) và word embeddings với BiLSTM 2 lớp.

Mô hình được huấn luyện và đánh giá trên bộ dữ liệu ReINTEL 2020, cho kết quả:

🎯 Accuracy: 95.83%

📊 F1-score: 86.84%

Kết quả cho thấy mô hình có khả năng phát hiện tin giả tiếng Việt hiệu quả và ổn định.

📑 Mục lục

Cài đặt

Nguồn dữ liệu

Các chức năng đã triển khai

Đóng góp

⚙️ Cài đặt

Clone dự án và cài đặt môi trường:

git clone https://github.com/TranPhucBinhh/LLMPhatHienTinGia.git
cd LLMPhatHienTinGia


Tạo virtual environment và cài đặt dependencies:

python -m venv venv
source venv/bin/activate   # hoặc .\venv\Scripts\activate trên Windows
pip install -r requirements.txt

📂 Nguồn dữ liệu

Chúng tôi sử dụng ReINTEL 2020 dataset (Lê et al.), được thu thập trong 2 tháng (8–10/2020), gồm 9713 mẫu từ:

Mạng xã hội (nhóm tin tức, KOLs).

Báo chí Việt Nam (bài báo về các tin giả đã bị gỡ).

Bộ dữ liệu bao phủ nhiều lĩnh vực: giải trí, thể thao, tài chính, y tế và đặc biệt là Covid-19 infodemic
Đây là tập dữ liệu công khai, được đánh giá cân bằng lớp và phù hợp để huấn luyện mô hình phát hiện tin giả.

🚀 Các chức năng nhóm đã thực hiện và triển khai

✔️ Tiền xử lý dữ liệu văn bản tiếng Việt (chuẩn hóa, tokenization với Bartpho).
✔️ Word embeddings huấn luyện sẵn cho tiếng Việt.
✔️ Mô hình BiLSTM 2 lớp xử lý ngữ cảnh hai chiều.
✔️ Huấn luyện và đánh giá trên tập ReINTEL 2020.
✔️ Độ chính xác cao: 95.83% Accuracy, 86.84% F1-score.
✔️ Script huấn luyện + đánh giá có thể chạy lại dễ dàng.

🤝 Đóng góp

Mô hình của chúng tôi chứng minh rằng các phương pháp Deep Learning (LSTM, BiLSTM) kết hợp với ngôn ngữ tiền huấn luyện (BERT/Bartpho) có thể đạt độ chính xác vượt trội trong bài toán phát hiện tin giả tiếng Việt.
Một số yếu tố quan trọng dẫn đến kết quả cao:
BiLSTM xử lý được ngữ cảnh cả trước và sau.
Embedding từ Bartpho mang tính ngữ nghĩa và cú pháp phù hợp cho tiếng Việt.
Bộ dữ liệu ReINTEL lớn và đa dạng, giúp mô hình học được nhiều mẫu tin giả phức tạp.
