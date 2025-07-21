# Frontend - AI-Powered CV-JD Analyzer

## 🚀 Tổng quan

Frontend được cập nhật để hỗ trợ đầy đủ **luồng hoạt động 6 bước** của hệ thống phân tích CV-JD:

1. **BƯỚC 1**: Nhận diện thành phần từ CV (NLP/NER model)
2. **BƯỚC 2**: Trích xuất yêu cầu từ JD (NLP/NER model)
3. **BƯỚC 3**: So sánh CV-JD để tính độ phù hợp (MML)
4. **BƯỚC 4**: Gợi ý chỉnh sửa CV (MML)
5. **BƯỚC 5**: Liệt kê kỹ năng còn thiếu (MML)
6. **BƯỚC 6**: Chấm điểm tổng thể ATS (MML)

## 🎯 Tính năng mới

### Input Form được cải tiến:

- **Accordion UI**: Giao diện gấp mở cho từng bước
- **Job Category Selection**: Chọn ngành nghề ứng tuyển
- **JD Text Input**: Nhập mô tả công việc
- **Additional Requirements**: Yêu cầu bổ sung (tùy chọn)
- **Workflow Visualization**: Hiển thị luồng 6 bước

### Results Display được nâng cấp:

- **3 Tabs chính**:
  - 📊 Tổng quan: Điểm số và kết quả chính
  - 🔄 Luồng 6 bước: Chi tiết từng bước xử lý
  - 📋 Chi tiết: Thông tin kỹ thuật và phân tích

## 🛠️ Cài đặt và chạy

```bash
# Cài đặt dependencies
npm install

# Chạy development server
npm start
```

## 📋 Cấu trúc Components

### InputForm.js

- **Accordion sections** cho từng bước input
- **Job category dropdown** với 13 ngành nghề
- **File upload** cho CV (PDF, DOCX, TXT)
- **JD text input** với placeholder mẫu
- **Validation** đầy đủ cho tất cả fields

### ResultsDisplay.js

- **WorkflowStep component** hiển thị từng bước
- **3 tabs** để tổ chức kết quả
- **Score visualization** với circular progress
- **Keywords display** cho skills
- **Suggestions list** với icons

### App.js

- **State management** cho tất cả inputs
- **API integration** với backend ML
- **Error handling** và loading states
- **Form validation** trước khi submit

## 🎨 UI/UX Features

### Design System:

- **Gradient backgrounds** cho modern look
- **Glassmorphism effects** với backdrop-filter
- **Smooth animations** và transitions
- **Responsive design** cho mobile/desktop
- **Color-coded workflow steps**

### User Experience:

- **Progressive disclosure** với accordions
- **Visual feedback** cho mỗi action
- **Loading states** với animations
- **Error handling** với clear messages
- **Accessibility** với proper ARIA labels

## 🔧 API Integration

### Endpoint:

```
POST http://localhost:8000/analyze-cv
```

### Request Format:

```javascript
const formData = new FormData();
formData.append("cv_file", cvFile);
formData.append("job_category", jobCategory);
formData.append("jd_text", jdText);
formData.append("job_requirements", jobRequirements); // optional
```

### Response Structure:

```javascript
{
  overall_score: number,
  ats_score: number,
  parsed_cv: object,
  jd_skills: array,
  ml_insights: object,
  quality_analysis: object,
  feedback: string,
  suggestions: array
}
```

## 📱 Responsive Design

### Breakpoints:

- **Desktop**: > 1024px - Full layout
- **Tablet**: 768px - 1024px - Adjusted spacing
- **Mobile**: < 768px - Stacked layout

### Mobile Optimizations:

- **Touch-friendly** buttons và inputs
- **Simplified** accordion interactions
- **Optimized** text sizes và spacing
- **Swipe gestures** cho tabs

## 🎯 Workflow Steps Visualization

### Tab 1: Tổng quan

- Điểm tổng thể và ATS
- Skills từ JD
- Gợi ý cải thiện
- Feedback tổng hợp

### Tab 2: Luồng 6 bước

- **B1**: Skills trích xuất từ CV
- **B2**: Skills trích xuất từ JD
- **B3**: Điểm so sánh và ML score
- **B4**: Số lượng gợi ý tạo ra
- **B5**: Skills còn thiếu
- **B6**: Điểm ATS và chất lượng

### Tab 3: Chi tiết

- ML Insights và important features
- Quality analysis scores
- Parsed CV details
- Technical metrics

## 🚀 Deployment

### Build cho Production:

```bash
npm run build
```

### Environment Variables:

```bash
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENVIRONMENT=development
```

## 🔍 Troubleshooting

### Common Issues:

1. **API Connection Error**: Kiểm tra backend server
2. **File Upload Issues**: Kiểm tra file format và size
3. **Validation Errors**: Đảm bảo đầy đủ required fields
4. **Performance Issues**: Kiểm tra network và server response

### Debug Mode:

```javascript
// Enable console logging
console.log("🚀 Bắt đầu phân tích với luồng 6 bước...");
console.log("📋 Job Category:", jobCategory);
console.log("📄 JD Text length:", jdText.length);
console.log("📁 CV File:", cvFile.name);
```

## 📈 Performance Optimizations

- **Lazy loading** cho components
- **Memoization** cho expensive calculations
- **Debounced** input handling
- **Optimized** re-renders với React.memo
- **Compressed** assets và images

## 🎨 Customization

### Theme Colors:

```css
--primary-color: #667eea
--secondary-color: #764ba2
--success-color: #27ae60
--warning-color: #f39c12
--error-color: #e74c3c
```

### Adding New Job Categories:

```javascript
const jobCategories = [
  { value: "NEW_CATEGORY", label: "Tên ngành mới" },
  // ... existing categories
];
```

## 🔮 Future Enhancements

- **Real-time analysis** với WebSocket
- **Batch processing** cho nhiều CV
- **Export results** sang PDF/Excel
- **Advanced filtering** và sorting
- **Multi-language** support
- **Dark mode** theme
