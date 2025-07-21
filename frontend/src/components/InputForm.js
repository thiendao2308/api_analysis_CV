import React from "react";
import {
  TextField,
  Button,
  CircularProgress,
  Paper,
  Typography,
  Box,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
} from "@mui/material";
import { useDropzone } from "react-dropzone";

const InputForm = ({
  cvFile,
  jdText,
  jobCategory,
  jobPosition,
  jobRequirements,
  loading,
  error,
  onDrop,
  setJdText,
  setJobCategory,
  setJobPosition,
  setJobRequirements,
  handleSubmit,
}) => {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        [".docx"],
      "text/plain": [".txt"],
    },
    maxFiles: 1,
  });

  // Danh sách ngành nghề
  const jobCategories = [
    { value: "INFORMATION-TECHNOLOGY", label: "Công nghệ thông tin" },
    { value: "ENGINEERING", label: "Kỹ thuật" },
    { value: "FINANCE", label: "Tài chính" },
    { value: "SALES", label: "Kinh doanh/Bán hàng" },
    { value: "HR", label: "Nhân sự" },
    { value: "MARKETING", label: "Marketing" },
    { value: "HEALTHCARE", label: "Y tế" },
    { value: "EDUCATION", label: "Giáo dục" },
    { value: "CONSULTANT", label: "Tư vấn" },
    { value: "DESIGNER", label: "Thiết kế" },
    { value: "ACCOUNTANT", label: "Kế toán" },
    { value: "LAWYER", label: "Luật sư" },
    { value: "OTHER", label: "Khác" },
  ];

  // Mapping vị trí cụ thể theo ngành nghề
  const jobPositions = {
    "INFORMATION-TECHNOLOGY": [
      { value: "FRONTEND_DEVELOPER", label: "Lập trình viên Front-end" },
      { value: "BACKEND_DEVELOPER", label: "Lập trình viên Back-end" },
      { value: "FULLSTACK_DEVELOPER", label: "Lập trình viên Full-stack" },
      { value: "MOBILE_DEVELOPER", label: "Lập trình viên Mobile" },
      { value: "DATA_SCIENTIST", label: "Nhà khoa học dữ liệu" },
      { value: "DEVOPS_ENGINEER", label: "Kỹ sư DevOps" },
      { value: "QA_ENGINEER", label: "Kỹ sư QA/Testing" },
      { value: "UI_UX_DESIGNER", label: "Thiết kế UI/UX" },
      { value: "SYSTEM_ADMIN", label: "Quản trị hệ thống" },
      { value: "CYBERSECURITY", label: "Bảo mật thông tin" },
    ],
    ENGINEERING: [
      { value: "SOFTWARE_ENGINEER", label: "Kỹ sư phần mềm" },
      { value: "MECHANICAL_ENGINEER", label: "Kỹ sư cơ khí" },
      { value: "ELECTRICAL_ENGINEER", label: "Kỹ sư điện" },
      { value: "CIVIL_ENGINEER", label: "Kỹ sư xây dựng" },
      { value: "CHEMICAL_ENGINEER", label: "Kỹ sư hóa học" },
      { value: "INDUSTRIAL_ENGINEER", label: "Kỹ sư công nghiệp" },
    ],
    FINANCE: [
      { value: "FINANCIAL_ANALYST", label: "Chuyên viên phân tích tài chính" },
      { value: "ACCOUNTANT", label: "Kế toán viên" },
      { value: "AUDITOR", label: "Kiểm toán viên" },
      { value: "INVESTMENT_BANKER", label: "Chuyên viên đầu tư" },
      { value: "FINANCIAL_ADVISOR", label: "Cố vấn tài chính" },
      { value: "RISK_MANAGER", label: "Quản lý rủi ro" },
    ],
    SALES: [
      { value: "SALES_REPRESENTATIVE", label: "Đại diện bán hàng" },
      { value: "SALES_MANAGER", label: "Quản lý bán hàng" },
      { value: "BUSINESS_DEVELOPMENT", label: "Phát triển kinh doanh" },
      { value: "ACCOUNT_MANAGER", label: "Quản lý khách hàng" },
      { value: "SALES_DIRECTOR", label: "Giám đốc bán hàng" },
    ],
    HR: [
      { value: "HR_SPECIALIST", label: "Chuyên viên nhân sự" },
      { value: "HR_MANAGER", label: "Quản lý nhân sự" },
      { value: "RECRUITER", label: "Tuyển dụng" },
      { value: "COMPENSATION_SPECIALIST", label: "Chuyên viên lương thưởng" },
      { value: "TRAINING_SPECIALIST", label: "Chuyên viên đào tạo" },
    ],
    MARKETING: [
      { value: "DIGITAL_MARKETING", label: "Marketing kỹ thuật số" },
      { value: "SEO_SPECIALIST", label: "Chuyên viên SEO" },
      { value: "CONTENT_MARKETING", label: "Marketing nội dung" },
      { value: "SOCIAL_MEDIA_MANAGER", label: "Quản lý mạng xã hội" },
      { value: "MARKETING_MANAGER", label: "Quản lý marketing" },
      { value: "BRAND_MANAGER", label: "Quản lý thương hiệu" },
      { value: "MARKETING_ANALYST", label: "Phân tích marketing" },
    ],
    HEALTHCARE: [
      { value: "DOCTOR", label: "Bác sĩ" },
      { value: "NURSE", label: "Y tá" },
      { value: "PHARMACIST", label: "Dược sĩ" },
      { value: "MEDICAL_TECHNOLOGIST", label: "Kỹ thuật viên y tế" },
      { value: "HEALTHCARE_ADMIN", label: "Quản lý y tế" },
    ],
    EDUCATION: [
      { value: "TEACHER", label: "Giáo viên" },
      { value: "PROFESSOR", label: "Giảng viên" },
      { value: "EDUCATION_ADMIN", label: "Quản lý giáo dục" },
      { value: "CURRICULUM_SPECIALIST", label: "Chuyên viên chương trình" },
    ],
    CONSULTANT: [
      { value: "MANAGEMENT_CONSULTANT", label: "Tư vấn quản lý" },
      { value: "IT_CONSULTANT", label: "Tư vấn CNTT" },
      { value: "FINANCIAL_CONSULTANT", label: "Tư vấn tài chính" },
      { value: "STRATEGY_CONSULTANT", label: "Tư vấn chiến lược" },
    ],
    DESIGNER: [
      { value: "GRAPHIC_DESIGNER", label: "Thiết kế đồ họa" },
      { value: "UI_UX_DESIGNER", label: "Thiết kế UI/UX" },
      { value: "WEB_DESIGNER", label: "Thiết kế web" },
      { value: "PRODUCT_DESIGNER", label: "Thiết kế sản phẩm" },
      { value: "INTERIOR_DESIGNER", label: "Thiết kế nội thất" },
    ],
    ACCOUNTANT: [
      { value: "ACCOUNTANT", label: "Kế toán viên" },
      { value: "SENIOR_ACCOUNTANT", label: "Kế toán trưởng" },
      { value: "AUDITOR", label: "Kiểm toán viên" },
      { value: "TAX_SPECIALIST", label: "Chuyên viên thuế" },
    ],
    LAWYER: [
      { value: "LAWYER", label: "Luật sư" },
      { value: "LEGAL_ADVISOR", label: "Cố vấn pháp lý" },
      { value: "PARALEGAL", label: "Trợ lý luật sư" },
      { value: "COMPLIANCE_SPECIALIST", label: "Chuyên viên tuân thủ" },
    ],
    OTHER: [{ value: "GENERAL", label: "Vị trí khác" }],
  };

  // Lấy danh sách vị trí theo ngành nghề đã chọn
  const getPositionsForCategory = (category) => {
    return jobPositions[category] || [];
  };

  return (
    <Paper className="main-card" elevation={3}>
      <Typography
        variant="h5"
        component="h1"
        gutterBottom
        className="form-title"
      >
        🚀 Hệ thống phân tích CV-JD AI
      </Typography>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Luồng hoạt động 6 bước: Nhận diện CV → Phân tích JD → So sánh → Gợi ý →
        Skills thiếu → Chấm điểm ATS
      </Typography>

      <form onSubmit={handleSubmit}>
        {/* BƯỚC 1: Upload CV */}
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon="▼">
            <Typography variant="h6">📄 BƯỚC 1: Tải lên CV</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box mb={3}>
              <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
                <Typography variant="subtitle1">Tải lên CV của bạn</Typography>
              </Box>
              <Box
                {...getRootProps()}
                className={`dropzone ${isDragActive ? "active" : ""}`}
              >
                <input {...getInputProps()} />
                {isDragActive ? (
                  <Typography>Thả file vào đây...</Typography>
                ) : (
                  <Typography>
                    Kéo và thả file CV, hoặc nhấn để chọn file
                  </Typography>
                )}
                <Typography variant="caption" color="text.secondary" mt={1}>
                  (PDF, DOCX, TXT)
                </Typography>
              </Box>
              {cvFile && (
                <Alert severity="success" sx={{ mt: 2 }}>
                  ✅ Đã chọn file: {cvFile.name}
                </Alert>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* BƯỚC 2: Chọn ngành nghề và vị trí */}
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon="▼">
            <Typography variant="h6">
              🎯 BƯỚC 2: Chọn ngành nghề & vị trí
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box mb={3}>
              <FormControl fullWidth required sx={{ mb: 2 }}>
                <InputLabel>Ngành nghề ứng tuyển</InputLabel>
                <Select
                  value={jobCategory}
                  label="Ngành nghề ứng tuyển"
                  onChange={(e) => {
                    setJobCategory(e.target.value);
                    setJobPosition(""); // Reset vị trí khi đổi ngành
                  }}
                >
                  {jobCategories.map((category) => (
                    <MenuItem key={category.value} value={category.value}>
                      {category.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {jobCategory && (
                <FormControl fullWidth required>
                  <InputLabel>Vị trí cụ thể</InputLabel>
                  <Select
                    value={jobPosition}
                    label="Vị trí cụ thể"
                    onChange={(e) => setJobPosition(e.target.value)}
                  >
                    {getPositionsForCategory(jobCategory).map((position) => (
                      <MenuItem key={position.value} value={position.value}>
                        {position.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              )}

              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ mt: 1, display: "block" }}
              >
                Hệ thống sẽ phân tích CV theo tiêu chí của ngành nghề và vị trí
                này
              </Typography>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* BƯỚC 3: Nhập JD */}
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon="▼">
            <Typography variant="h6">
              📋 BƯỚC 3: Mô tả công việc (JD)
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box mb={3}>
              <TextField
                label="Dán mô tả công việc vào đây"
                multiline
                rows={8}
                variant="outlined"
                fullWidth
                value={jdText}
                onChange={(e) => setJdText(e.target.value)}
                required
                placeholder="Ví dụ: Tuyển dụng Senior Python Developer...
Yêu cầu:
- Kinh nghiệm 3+ năm với Python
- Thành thạo Django, Flask
- Biết sử dụng PostgreSQL, Redis
- Có kinh nghiệm với AWS, Docker"
              />
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ mt: 1, display: "block" }}
              >
                Hệ thống sẽ trích xuất skills và yêu cầu từ JD này
              </Typography>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* BƯỚC 4: Yêu cầu bổ sung */}
        <Accordion>
          <AccordionSummary expandIcon="▼">
            <Typography variant="h6">
              ⚙️ BƯỚC 4: Yêu cầu bổ sung (tùy chọn)
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box mb={3}>
              <TextField
                label="Nhập các yêu cầu hoặc kỹ năng bổ sung"
                multiline
                rows={3}
                variant="outlined"
                fullWidth
                value={jobRequirements}
                onChange={(e) => setJobRequirements(e.target.value)}
                placeholder="Ví dụ: Kỹ năng leadership, kinh nghiệm quản lý team, chứng chỉ AWS..."
              />
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ mt: 1, display: "block" }}
              >
                Các yêu cầu này sẽ được thêm vào phân tích
              </Typography>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Thông tin luồng hoạt động */}
        <Box
          sx={{
            mb: 3,
            p: 2,
            bgcolor: "background.paper",
            borderRadius: 1,
            border: "1px solid #e0e0e0",
          }}
        >
          <Typography variant="subtitle2" gutterBottom>
            🔄 Luồng hoạt động hệ thống:
          </Typography>
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mt: 1 }}>
            <Chip label="B1: Nhận diện CV" size="small" color="primary" />
            <Chip label="B2: Phân tích JD" size="small" color="primary" />
            <Chip label="B3: So sánh CV-JD" size="small" color="primary" />
            <Chip label="B4: Gợi ý cải thiện" size="small" color="primary" />
            <Chip label="B5: Skills thiếu" size="small" color="primary" />
            <Chip label="B6: Chấm điểm ATS" size="small" color="primary" />
          </Box>
        </Box>

        {/* Nút phân tích */}
        <Button
          type="submit"
          variant="contained"
          color="secondary"
          size="large"
          fullWidth
          disabled={
            loading || !cvFile || !jdText || !jobCategory || !jobPosition
          }
          startIcon={
            loading ? <CircularProgress size={20} color="inherit" /> : null
          }
          sx={{
            py: 1.5,
            fontSize: "1.1rem",
            background: "linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)",
            "&:hover": {
              background: "linear-gradient(45deg, #FF8E53 30%, #FE6B8B 90%)",
            },
          }}
        >
          {loading ? "Đang phân tích..." : "🚀 Bắt đầu phân tích CV-JD"}
        </Button>

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            ❌ Lỗi: {error}
          </Alert>
        )}
      </form>
    </Paper>
  );
};

export default InputForm;
