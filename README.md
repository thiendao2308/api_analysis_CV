# AI CV Analyzer

Hệ thống phân tích và tối ưu CV sử dụng AI để đánh giá khả năng qua ATS (Applicant Tracking System).

## Cấu trúc dự án

Dự án bao gồm 2 phần chính:

- **`backend`**: Xây dựng bằng FastAPI, chịu trách nhiệm xử lý logic, phân tích CV và cung cấp API.
- **`frontend`**: Xây dựng bằng React, là giao diện người dùng để tương tác với hệ thống.

## Hướng dẫn cài đặt và khởi chạy

### 1. Chuẩn bị

Clone repository về máy của bạn:

```bash
git clone [repository-url]
cd [repository-folder]
```

### 2. Backend (FastAPI)

a. **Di chuyển vào thư mục backend và tạo môi trường ảo:**

```bash
# Thư mục backend chính là thư mục gốc của dự án
python -m venv venv
```

b. **Kích hoạt môi trường ảo:**

- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```

c. **Cài đặt các thư viện cần thiết:**

```bash
pip install -r requirements.txt
```

d. **Tải model ngôn ngữ của spaCy:**

```bash
python -m spacy download en_core_web_lg
```

e. **Khởi động server backend:**

```bash
uvicorn ml_architecture.main:app --host 0.0.0.0 --port 8000 --reload
doccano createuser --username admin --password admin123 --email admin@example.com
```

Server sẽ chạy tại `http://localhost:8000`.

### 3. Frontend (React)

a. **Di chuyển vào thư mục frontend:**
Mở một cửa sổ terminal mới và chạy lệnh:

```bash
cd frontend
```

b. **Cài đặt các dependencies:**

```bash
npm install
```

c. **Khởi động ứng dụng frontend:**

```bash
npm start
```

Ứng dụng sẽ tự động mở trong trình duyệt tại `http://localhost:3000`.

## Sử dụng

1. Đảm bảo cả server backend và frontend đều đang chạy.
2. Truy cập `http://localhost:3000` trên trình duyệt.
3. Upload CV (định dạng PDF hoặc DOCX).
4. Dán nội dung mô tả công việc (Job Description) vào ô tương ứng.
5. Nhấn nút "Phân tích" và chờ xem kết quả.
