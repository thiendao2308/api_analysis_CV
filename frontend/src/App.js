import React, { useState, useCallback } from "react";
import axios from "axios";
import { Container, Typography, Grid, AppBar, Toolbar } from "@mui/material";
import InputForm from "./components/InputForm";
import ResultsDisplay from "./components/ResultsDisplay";

function App() {
  const [cvFile, setCvFile] = useState(null);
  const [jdText, setJdText] = useState("");
  const [jobCategory, setJobCategory] = useState("");
  const [jobPosition, setJobPosition] = useState("");
  const [jobRequirements, setJobRequirements] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setCvFile(acceptedFiles[0]);
    }
  }, []);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!cvFile || !jdText || !jobCategory || !jobPosition) {
      setError(
        "Vui lòng tải lên CV, chọn ngành nghề, vị trí và nhập Mô tả công việc."
      );
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("cv_file", cvFile);
    formData.append("job_category", jobCategory);
    formData.append("job_position", jobPosition);
    formData.append("jd_text", jdText);
    if (jobRequirements) {
      formData.append("job_requirements", jobRequirements);
    }

    try {
      console.log("🚀 Bắt đầu phân tích với luồng 6 bước...");
      console.log("📋 Job Category:", jobCategory);
      console.log("🎯 Job Position:", jobPosition);
      console.log("📄 JD Text length:", jdText.length);
      console.log("📁 CV File:", cvFile.name);

      const response = await axios.post(
        "http://localhost:8000/analyze-cv",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      console.log("✅ Kết quả phân tích:", response.data);
      setResult(response.data);
    } catch (err) {
      console.error("❌ API Call failed:", err);
      const errorMsg =
        err.response?.data?.detail ||
        "Đã xảy ra lỗi khi kết nối tới server. Vui lòng đảm bảo backend ML đang chạy.";
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <AppBar position="static" color="primary" elevation={1}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            🤖 AI-Powered CV-JD Analyzer
          </Typography>
          <Typography variant="caption" sx={{ opacity: 0.8 }}>
            Luồng hoạt động 6 bước
          </Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg" className="app-container">
        <Grid container spacing={4}>
          <Grid item xs={12} md={5}>
            <InputForm
              cvFile={cvFile}
              jdText={jdText}
              jobCategory={jobCategory}
              jobPosition={jobPosition}
              jobRequirements={jobRequirements}
              loading={loading}
              error={error}
              onDrop={onDrop}
              setJdText={setJdText}
              setJobCategory={setJobCategory}
              setJobPosition={setJobPosition}
              setJobRequirements={setJobRequirements}
              handleSubmit={handleSubmit}
            />
          </Grid>
          <Grid item xs={12} md={7}>
            <ResultsDisplay result={result} loading={loading} />
          </Grid>
        </Grid>
      </Container>
    </>
  );
}

export default App;
