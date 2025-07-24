# AI CV Analyzer - Backend

## 1. Luồng hoạt động của Backend

1. **Nhận request từ client**: Người dùng (hoặc web client) gửi file CV (PDF/DOCX/TXT) và nội dung JD (Job Description) lên API `/analyze-cv`.
2. **Xử lý file CV**: Backend trích xuất nội dung text từ file CV.
3. **Trích xuất thông tin từ CV**: Sử dụng OpenAI API để phân tích, trích xuất kinh nghiệm, học vấn, kỹ năng, dự án từ CV.
4. **Trích xuất kỹ năng từ JD**: Sử dụng OpenAI API để trích xuất danh sách kỹ năng từ JD.
5. **So khớp kỹ năng CV-JD**: So sánh kỹ năng giữa CV và JD, xác định kỹ năng phù hợp và còn thiếu.
6. **Đánh giá chất lượng CV**: Phân tích cấu trúc, nội dung, trình bày của CV để chấm điểm chất lượng tổng thể.
7. **Tổng hợp kết quả**: Trả về điểm số ATS, điểm tổng thể, kỹ năng phù hợp, kỹ năng thiếu, gợi ý cải thiện, và các phân tích chi tiết khác cho client.

---

## 2. Công nghệ & thư viện sử dụng

- **Ngôn ngữ:** Python 3.10+
- **Framework:** FastAPI (API backend)
- **Xử lý file:** PyPDF2, python-docx, PyMuPDF
- **NLP & ML:** spaCy, transformers, torch, scikit-learn
- **LLM API:** openai (sử dụng GPT-3.5/4 qua API)
- **Quản lý biến môi trường:** python-dotenv
- **Khác:** pydantic, uvicorn, numpy, pandas, tqdm, matplotlib, seaborn

---

## 3. Hướng dẫn sử dụng backend

### A. Cài đặt môi trường

1. **Clone code về máy:**
   ```bash
   git clone <link-repo-cua-ban>
   cd <ten-thu-muc-du-an>
   ```
2. **Tạo virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # hoặc venv\Scripts\activate trên Windows
   ```
3. **Cài đặt dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Tạo file .env:**
   ```env
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

### B. Chạy server backend

```bash
uvicorn ml_architecture.main:app --host 0.0.0.0 --port 8000 --reload
```

Server sẽ chạy tại: `http://localhost:8000`

---

## 4. Logic chính của hệ thống

- **/analyze-cv** (POST):
  - Nhận file CV, job_category, job_position, jd_text, job_requirements (tùy chọn).
  - Xử lý file CV, trích xuất text.
  - Gọi OpenAI API để phân tích CV và JD.
  - So khớp kỹ năng, tính điểm ATS, điểm tổng thể, phân tích chất lượng.
  - Trả về kết quả dạng JSON:
    ```json
    {
      "cv_analysis": {...},
      "jd_analysis": {...},
      "matching_analysis": {...},
      "quality_analysis": {...},
      "ml_insights": {...},
      "scores": {"ats_score": 50, "overall_score": 40},
      "feedback": "...",
      "suggestions": [...],
      "job_category": "...",
      "job_position": "..."
    }
    ```
- **/analyze-jd** (POST):
  - Nhận jd_text, trả về danh sách kỹ năng trích xuất từ JD.

---

## 5. Hướng dẫn test API bằng Postman

### A. Test endpoint `/analyze-cv`

1. **Chọn method:** POST
2. **URL:** `http://localhost:8000/analyze-cv`
3. **Tab Body:**
   - Chọn `form-data`
   - Thêm các trường:
     - `cv_file`: (type: File) chọn file PDF/DOCX/TXT
     - `job_category`: (type: Text) VD: INFORMATION-TECHNOLOGY
     - `job_position`: (type: Text) VD: FULLSTACK_DEVELOPER
     - `jd_text`: (type: Text) dán nội dung JD
     - `job_requirements`: (type: Text, optional)
4. **Send** và xem kết quả trả về ở tab Response.

### B. Test endpoint `/analyze-jd`

1. **Chọn method:** POST
2. **URL:** `http://localhost:8000/analyze-jd`
3. **Tab Body:**
   - Chọn `x-www-form-urlencoded`
   - Thêm trường:
     - `jd_text`: (type: Text) dán nội dung JD
4. **Send** và xem kết quả trả về.

---

## 6. Lưu ý bảo mật

- Không commit file `.env` chứa API key lên git.
- Đảm bảo server backend chỉ mở cổng cần thiết (thường là 8000 hoặc qua reverse proxy).
- Nếu deploy production, nên dùng HTTPS và giới hạn CORS cho domain web client.
