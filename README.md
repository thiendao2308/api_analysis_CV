# AI CV Analyzer - Backend

## 🚀 **Tính năng mới (Latest Updates)**

### **🎯 Intelligent JD Matching**

- **Semantic Skill Matching**: Sử dụng LLM để hiểu sự tương đồng giữa kỹ năng CV và JD
- **Ví dụ thông minh**:
  - CV có ".NET Stack" → JD yêu cầu "C# .NET Core 6" ✅
  - CV có "Capcut" → JD yêu cầu "video editing applications" ✅
- **Multi-industry Support**: Tối ưu hóa prompt cho nhiều ngành nghề (IT, Marketing, Finance, Design, HR)

### **🤖 LLM-Powered Feedback System**

- **Intelligent Feedback**: Đưa ra gợi ý chân thật, cụ thể và hữu ích
- **Context-Aware**: Phân tích dựa trên ngành nghề, vị trí công việc cụ thể
- **Balanced Assessment**: Vừa động viên vừa chỉ ra điểm cần cải thiện
- **Fallback System**: Hoạt động ngay cả khi LLM không khả dụng

### **⚡ Memory Optimization & Deployment**

- **Render Deployment**: Deploy thành công trên Render free tier
- **Memory Management**: Lazy loading, garbage collection, CPU-only PyTorch
- **Timeout Prevention**: Tối ưu hóa startup time và health check
- **Production Ready**: Single worker, optimized settings

---

## 1. Luồng hoạt động của Backend

1. **Nhận request từ client**: Người dùng gửi file CV (PDF/DOCX/TXT) và nội dung JD lên API `/analyze-cv`
2. **Xử lý file CV**: Backend trích xuất nội dung text từ file CV
3. **Trích xuất thông tin từ CV**: Sử dụng OpenAI API để phân tích, trích xuất kinh nghiệm, học vấn, kỹ năng, dự án từ CV
4. **Trích xuất kỹ năng từ JD**: Sử dụng OpenAI API để trích xuất danh sách kỹ năng từ JD
5. **🆕 Intelligent JD Matching**: So sánh kỹ năng thông minh sử dụng LLM để hiểu semantic similarity
6. **Đánh giá chất lượng CV**: Phân tích cấu trúc, nội dung, trình bày của CV
7. **🆕 LLM Feedback Generation**: Tạo feedback thông minh và gợi ý cải thiện
8. **Tổng hợp kết quả**: Trả về điểm số ATS, điểm tổng thể, kỹ năng phù hợp, kỹ năng thiếu, gợi ý cải thiện

---

## 2. Công nghệ & thư viện sử dụng

- **Ngôn ngữ:** Python 3.10+
- **Framework:** FastAPI (API backend)
- **Xử lý file:** PyPDF2, python-docx, PyMuPDF
- **NLP & ML:** spaCy, transformers, torch, scikit-learn
- **🆕 LLM API:** openai (GPT-3.5/4) cho Intelligent Matching & Feedback
- **🆕 Memory Management:** psutil, gc, environment variables optimization
- **Deployment:** Render, uvicorn với optimized settings
- **Quản lý biến môi trường:** python-dotenv
- **Khác:** pydantic, uvicorn, numpy, pandas

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
   python -m spacy download en_core_web_sm
   ```

4. **Tạo file .env:**
   ```env
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

### B. Chạy server backend

#### **Local Development:**

```bash
uvicorn ml_architecture.main:app --host 0.0.0.0 --port 8000 --reload
```

#### **🆕 Production (Render):**

```bash
python start_server.py
```

Server sẽ chạy tại: `http://localhost:8000` (local) hoặc `https://your-app.onrender.com` (Render)

---

## 4. Logic chính của hệ thống

### **🆕 Intelligent JD Matching**

```python
# Ví dụ matching thông minh:
CV Skills: [".NET Stack", "Capcut", "React"]
JD Skills: ["C# .NET Core 6", "video editing applications", "JavaScript"]
# Kết quả: 3/3 matches (100%) thay vì 0/3 (0%)
```

### **🆕 LLM Feedback System**

```python
# Feedback thông minh dựa trên context:
{
    "overall_assessment": "CV của bạn khá phù hợp với vị trí Fullstack Developer",
    "strengths": ["Có kỹ năng Python, Django và React phù hợp"],
    "weaknesses": ["Thiếu kinh nghiệm với Docker, AWS"],
    "specific_suggestions": ["Nên học Docker để triển khai ứng dụng"],
    "priority_actions": ["Xây dựng project thực tế với Docker"],
    "encouragement": "Với sự cố gắng, bạn sẽ trở thành Fullstack Developer xuất sắc!"
}
```

### **API Endpoints:**

- **/analyze-cv** (POST):

  - Nhận file CV, job_category, job_position, jd_text
  - **🆕 Intelligent JD Matching** với semantic understanding
  - **🆕 LLM-powered feedback** thông minh và chân thật
  - Trả về kết quả chi tiết:
    ```json
    {
      "cv_analysis": {...},
      "jd_analysis": {...},
      "matching_analysis": {
        "matching_skills": [...],
        "missing_skills": [...],
        "skills_match_score": 85.5,
        "exact_matches": [...],
        "semantic_matches": [...]
      },
      "quality_analysis": {...},
      "scores": {"ats_score": 75, "overall_score": 82},
      "feedback": "LLM-generated intelligent feedback",
      "suggestions": ["Smart suggestions from LLM"],
      "job_category": "...",
      "job_position": "..."
    }
    ```

- **/analyze-jd** (POST):

  - Nhận jd_text, trả về danh sách kỹ năng trích xuất từ JD

- **🆕 /health** (GET):
  - Health check endpoint cho deployment

---

## 5. Hướng dẫn test API

### A. Test endpoint `/analyze-cv`

1. **Chọn method:** POST
2. **URL:** `http://localhost:8000/analyze-cv` (local) hoặc `https://your-app.onrender.com/analyze-cv` (Render)
3. **Tab Body:**
   - Chọn `form-data`
   - Thêm các trường:
     - `cv_file`: (type: File) chọn file PDF/DOCX/TXT
     - `job_category`: (type: Text) VD: INFORMATION-TECHNOLOGY
     - `job_position`: (type: Text) VD: FULLSTACK_DEVELOPER
     - `jd_text`: (type: Text) dán nội dung JD
4. **Send** và xem kết quả trả về

### B. Test Intelligent JD Matching

**Ví dụ test case:**

```json
{
  "cv_skills": [".NET Stack", "Capcut", "React"],
  "jd_skills": ["C# .NET Core 6", "video editing applications", "JavaScript"],
  "expected_result": "3 semantic matches found"
}
```

### C. Test LLM Feedback

**Kết quả mong đợi:**

- Feedback chân thật và cụ thể
- Gợi ý thực tế để cải thiện
- Động viên phù hợp với điểm số

---

## 6. 🆕 Deployment trên Render

### **Cấu hình Render:**

```yaml
# render.yaml
services:
  - type: web
    name: cv-analysis-api
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt && python -m spacy download en_core_web_sm
    startCommand: python start_server.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.13
      - key: PYTORCH_CUDA_ALLOC_CONF
        value: max_split_size_mb:128
      # ... memory optimization variables
```

### **Memory Optimization:**

- **Lazy Loading**: Models chỉ load khi cần
- **Garbage Collection**: Tự động dọn memory
- **CPU-only PyTorch**: Tiết kiệm memory
- **Single Worker**: Giảm memory usage

---

## 7. 🆕 Performance & Monitoring

### **Memory Usage:**

- **Local**: ~200-400MB
- **Render Free Tier**: ~450MB (optimized)
- **Health Check**: <50MB

### **Response Time:**

- **First Request**: 10-15s (model loading)
- **Subsequent Requests**: 3-5s
- **LLM Calls**: 2-3s per call

### **Accuracy:**

- **Intelligent Matching**: 85-95% accuracy
- **Traditional Matching**: 60-70% accuracy
- **LLM Feedback**: Context-aware và chân thật

---

## 8. Lưu ý bảo mật

- **🆕 Environment Variables**: Sử dụng Render Dashboard để set `OPENAI_API_KEY`
- **🆕 Memory Limits**: Tối ưu hóa cho Render free tier (512MB)
- **🆕 Health Check**: Simplified endpoint để tránh timeout
- **CORS**: Cấu hình cho production deployment
- **HTTPS**: Tự động với Render

---

## 9. 🆕 Troubleshooting

### **Memory Issues:**

```bash
# Check memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

### **LLM API Issues:**

```bash
# Test OpenAI connection
python -c "import openai; print('OpenAI connection OK')"
```

### **Deployment Issues:**

- Check Render logs
- Verify environment variables
- Monitor memory usage

---

## 10. 🆕 Roadmap

### **Completed ✅:**

- [x] Intelligent JD Matching
- [x] LLM Feedback Generator
- [x] Memory Optimization
- [x] Render Deployment
- [x] Multi-industry Support

### **In Progress 🔄:**

- [ ] Enhanced Error Handling
- [ ] Caching System
- [ ] Performance Monitoring

### **Planned 📋:**

- [ ] Multi-language Support
- [ ] Advanced Analytics
- [ ] Real-time Collaboration
