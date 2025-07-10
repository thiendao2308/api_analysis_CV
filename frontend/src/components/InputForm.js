import React from "react";
import {
  TextField,
  Button,
  CircularProgress,
  Paper,
  Typography,
  Box,
  Alert,
} from "@mui/material";
import { useDropzone } from "react-dropzone";

const InputForm = ({
  cvFile,
  jdText,
  jobRequirements,
  loading,
  error,
  onDrop,
  setJdText,
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

  return (
    <Paper className="main-card" elevation={3}>
      <Typography
        variant="h5"
        component="h1"
        gutterBottom
        className="form-title"
      >
        Nhập thông tin
      </Typography>
      <form onSubmit={handleSubmit}>
        <Box mb={3}>
          <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
            <Typography variant="h6">Tải lên CV của bạn</Typography>
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
              Đã chọn file: {cvFile.name}
            </Alert>
          )}
        </Box>
        <Box mb={3}>
          <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
            <Typography variant="h6">Mô tả công việc (JD)</Typography>
          </Box>
          <TextField
            label="Dán mô tả công việc vào đây"
            multiline
            rows={10}
            variant="outlined"
            fullWidth
            value={jdText}
            onChange={(e) => setJdText(e.target.value)}
            required
          />
        </Box>
        <Box mb={3}>
          <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
            <Typography variant="h6">Yêu cầu khác (tùy chọn)</Typography>
          </Box>
          <TextField
            label="Nhập các yêu cầu hoặc kỹ năng khác"
            multiline
            rows={3}
            variant="outlined"
            fullWidth
            value={jobRequirements}
            onChange={(e) => setJobRequirements(e.target.value)}
          />
        </Box>
        <Button
          type="submit"
          variant="contained"
          color="secondary"
          size="large"
          fullWidth
          disabled={loading}
          startIcon={
            loading ? <CircularProgress size={20} color="inherit" /> : null
          }
        >
          {loading ? "Đang phân tích..." : "Phân Tích"}
        </Button>
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </form>
    </Paper>
  );
};

export default InputForm;
