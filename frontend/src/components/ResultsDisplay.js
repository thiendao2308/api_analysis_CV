import React, { useState } from "react";
import {
  Typography,
  CircularProgress,
  Card,
  CardContent,
  Grid,
  Box,
  Chip,
  Tabs,
  Tab,
  LinearProgress,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Alert,
} from "@mui/material";

const Keywords = ({ keywords, title, color, iconType }) => {
  return (
    <Box>
      <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
        <Typography variant="h6" component="div" sx={{ ml: 1 }}>
          {title}
        </Typography>
      </Box>
      <Box className="keyword-container">
        {keywords && keywords.length > 0 ? (
          keywords.map((kw, index) => (
            <Chip
              key={index}
              label={typeof kw === "string" ? kw : kw.keyword}
              color={color}
              variant="filled"
            />
          ))
        ) : (
          <Typography variant="body2" color="text.secondary">
            Không có.
          </Typography>
        )}
      </Box>
    </Box>
  );
};

const Suggestions = ({ suggestions, title }) => (
  <List>
    <Box sx={{ display: "flex", alignItems: "center" }}>
      <Typography variant="h6">{title}</Typography>
    </Box>
    {suggestions && suggestions.length > 0 ? (
      suggestions.map((suggestion, index) => (
        <ListItem key={index} disableGutters>
          <ListItemIcon sx={{ minWidth: "auto", mr: 1 }}>💡</ListItemIcon>
          <ListItemText primary={suggestion} />
        </ListItem>
      ))
    ) : (
      <Typography variant="body2" color="text.secondary">
        Không có gợi ý nào.
      </Typography>
    )}
  </List>
);

const SectionAnalysis = ({ sections }) => (
  <Box>
    <Typography variant="h6" gutterBottom>
      Phân tích chi tiết
    </Typography>
    {sections && sections.length > 0 ? (
      sections.map((section, index) => (
        <Box key={index} sx={{ mb: 2 }}>
          <Box
            sx={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Typography variant="subtitle1">{section.section_name}</Typography>
            <Typography variant="subtitle1">
              {Math.round(section.score * 100)}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={section.score * 100}
            sx={{ height: 8, borderRadius: 4, mb: 1 }}
          />
          {section.suggestions &&
            section.suggestions.map((s, i) => (
              <Typography
                key={i}
                variant="caption"
                color="text.secondary"
                sx={{ display: "block" }}
              >
                - {s}
              </Typography>
            ))}
        </Box>
      ))
    ) : (
      <Typography variant="body2" color="text.secondary">
        Không có phân tích chi tiết.
      </Typography>
    )}
  </Box>
);

const WorkflowStep = ({ step, title, content, isCompleted = true }) => (
  <Box sx={{ mb: 2, p: 2, border: "1px solid #e0e0e0", borderRadius: 1 }}>
    <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
      <Chip
        label={`B${step}`}
        size="small"
        color={isCompleted ? "success" : "default"}
        sx={{ mr: 1 }}
      />
      <Typography variant="subtitle1" fontWeight="bold">
        {title}
      </Typography>
    </Box>
    <Box sx={{ ml: 4 }}>{content}</Box>
  </Box>
);

const ResultsDisplay = ({ result, loading }) => {
  const [activeTab, setActiveTab] = useState(0);

  if (loading) {
    return (
      <Box className="loading-container">
        <CircularProgress size={60} />
        <Typography variant="h6" mt={2}>
          🤖 AI đang phân tích CV của bạn...
        </Typography>
        <Typography variant="body2" color="text.secondary" mt={1}>
          Luồng hoạt động 6 bước đang được thực hiện
        </Typography>
      </Box>
    );
  }

  if (!result) {
    return (
      <Paper className="main-card placeholder-card" elevation={3}>
        <Typography variant="h5" color="text.secondary" mt={2}>
          Kết quả phân tích sẽ được hiển thị ở đây
        </Typography>
        <Typography color="text.secondary">
          Vui lòng nhập CV, chọn ngành nghề và JD để bắt đầu.
        </Typography>
      </Paper>
    );
  }

  const score = result.scores?.overall_score || result.overall_score || 0;
  const atsScore = result.scores?.ats_score || result.ats_score || 0;

  return (
    <Card className="result-card" elevation={3}>
      <CardContent>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={8}>
            <Typography
              variant="h4"
              component="div"
              gutterBottom
              className="result-title"
            >
              🎯 Kết Quả Phân Tích
            </Typography>
            <Typography variant="h6" color="text.secondary">
              Điểm phù hợp tổng thể:
            </Typography>
          </Grid>
          <Grid item xs={12} md={4} sx={{ textAlign: "center" }}>
            <Box className="score-dial">
              <CircularProgress
                variant="determinate"
                value={score}
                size={120}
                thickness={4}
                className={`score-circle score-${
                  score >= 75 ? "high" : score >= 50 ? "medium" : "low"
                }`}
              />
              <Box className="score-text-box">
                <Typography variant="h3" component="div" className="score-text">
                  {`${score}%`}
                </Typography>
              </Box>
            </Box>
          </Grid>
        </Grid>

        <Box sx={{ borderBottom: 1, borderColor: "divider", mt: 3, mb: 2 }}>
          <Tabs
            value={activeTab}
            onChange={(e, newValue) => setActiveTab(newValue)}
            variant="fullWidth"
            indicatorColor="secondary"
            textColor="secondary"
          >
            <Tab label="📊 Tổng quan" iconPosition="start" />
            <Tab label="🔄 Luồng 6 bước" iconPosition="start" />
            <Tab label="📋 Chi tiết" iconPosition="start" />
          </Tabs>
        </Box>

        <Box className="tab-panel">
          {activeTab === 0 && (
            <Grid container spacing={4}>
              {/* Điểm tổng quan */}
              <Grid item xs={12}>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    📊 Điểm đánh giá
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Box
                        sx={{
                          textAlign: "center",
                          p: 2,
                          bgcolor: "background.paper",
                          borderRadius: 1,
                        }}
                      >
                        <Typography variant="h4" color="primary">
                          {score}%
                        </Typography>
                        <Typography variant="body2">Điểm tổng thể</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Box
                        sx={{
                          textAlign: "center",
                          p: 2,
                          bgcolor: "background.paper",
                          borderRadius: 1,
                        }}
                      >
                        <Typography variant="h4" color="secondary">
                          {atsScore}%
                        </Typography>
                        <Typography variant="body2">Điểm ATS</Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </Box>
              </Grid>

              {/* Skills từ JD */}
              {result.jd_analysis?.extracted_skills &&
                result.jd_analysis.extracted_skills.length > 0 && (
                  <Grid item xs={12}>
                    <Keywords
                      keywords={result.jd_analysis.extracted_skills}
                      title="📋 Skills từ JD"
                      color="primary"
                    />
                  </Grid>
                )}

              {/* Gợi ý cải thiện */}
              {result.suggestions && result.suggestions.length > 0 && (
                <Grid item xs={12}>
                  <Suggestions
                    suggestions={result.suggestions}
                    title="💡 Gợi ý cải thiện"
                  />
                </Grid>
              )}

              {/* Feedback */}
              {result.feedback && (
                <Grid item xs={12}>
                  <Alert severity="info" sx={{ mt: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      💬 Nhận xét
                    </Typography>
                    <Typography>{result.feedback}</Typography>
                  </Alert>
                </Grid>
              )}
            </Grid>
          )}

          {activeTab === 1 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                🔄 Luồng hoạt động 6 bước
              </Typography>

              <WorkflowStep
                step={1}
                title="Nhận diện thành phần từ CV"
                content={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      ✅ Đã trích xuất skills:{" "}
                      {result.cv_analysis?.skills?.length || 0} kỹ năng
                    </Typography>
                    {result.cv_analysis?.skills && (
                      <Box sx={{ mt: 1 }}>
                        <Keywords
                          keywords={result.cv_analysis.skills}
                          color="success"
                        />
                      </Box>
                    )}
                  </Box>
                }
              />

              <WorkflowStep
                step={2}
                title="Trích xuất yêu cầu từ JD"
                content={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      ✅ Đã trích xuất:{" "}
                      {result.jd_analysis?.extracted_skills?.length || 0} skills
                      từ JD
                    </Typography>
                    {result.jd_analysis?.extracted_skills && (
                      <Box sx={{ mt: 1 }}>
                        <Keywords
                          keywords={result.jd_analysis.extracted_skills}
                          color="primary"
                        />
                      </Box>
                    )}
                  </Box>
                }
              />

              <WorkflowStep
                step={3}
                title="So sánh CV-JD để tính độ phù hợp"
                content={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      📊 Điểm tổng thể: {score}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      🤖 ML Score: {result.ml_insights?.ml_score || 0}%
                    </Typography>
                  </Box>
                }
              />

              <WorkflowStep
                step={4}
                title="Gợi ý chỉnh sửa CV"
                content={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      💡 Đã tạo {result.suggestions?.length || 0} gợi ý cải
                      thiện
                    </Typography>
                    {result.suggestions && (
                      <Box sx={{ mt: 1 }}>
                        <Suggestions suggestions={result.suggestions} />
                      </Box>
                    )}
                  </Box>
                }
              />

              <WorkflowStep
                step={5}
                title="Liệt kê kỹ năng còn thiếu"
                content={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      ⚠️ Cần bổ sung các kỹ năng còn thiếu
                    </Typography>
                    {/* Có thể thêm logic tính skills thiếu ở đây */}
                  </Box>
                }
              />

              <WorkflowStep
                step={6}
                title="Chấm điểm tổng thể ATS"
                content={
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      📊 Điểm ATS: {atsScore}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      📋 Chất lượng CV:{" "}
                      {result.quality_analysis?.quality_score
                        ? Math.round(
                            result.quality_analysis.quality_score * 100
                          )
                        : 0}
                      %
                    </Typography>
                  </Box>
                }
              />
            </Box>
          )}

          {activeTab === 2 && (
            <Grid container spacing={4}>
              {/* ML Insights */}
              {result.ml_insights && (
                <Grid item xs={12}>
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      🤖 ML Insights
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      ML Score: {result.ml_insights.ml_score || 0}%
                    </Typography>
                    {result.ml_insights.important_features && (
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="subtitle2">
                          Important Features:
                        </Typography>
                        <Keywords
                          keywords={result.ml_insights.important_features}
                          color="info"
                        />
                      </Box>
                    )}
                  </Box>
                </Grid>
              )}

              {/* Quality Analysis */}
              {result.quality_analysis && (
                <Grid item xs={12}>
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      📊 Phân tích chất lượng
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Structure Score:{" "}
                      {result.quality_analysis.structure_score
                        ? Math.round(
                            result.quality_analysis.structure_score * 100
                          )
                        : 0}
                      %
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Content Score:{" "}
                      {result.quality_analysis.content_score
                        ? Math.round(
                            result.quality_analysis.content_score * 100
                          )
                        : 0}
                      %
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Presentation Score:{" "}
                      {result.quality_analysis.presentation_score
                        ? Math.round(
                            result.quality_analysis.presentation_score * 100
                          )
                        : 0}
                      %
                    </Typography>
                  </Box>
                </Grid>
              )}

              {/* Parsed CV Details */}
              {result.parsed_cv && (
                <Grid item xs={12}>
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      📄 Chi tiết CV đã parse
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Summary: {result.parsed_cv.summary ? "Có" : "Không có"}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Experience:{" "}
                      {result.parsed_cv.experience ? "Có" : "Không có"}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Education:{" "}
                      {result.parsed_cv.education ? "Có" : "Không có"}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Skills: {result.parsed_cv.skills?.length || 0} kỹ năng
                    </Typography>
                  </Box>
                </Grid>
              )}
            </Grid>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ResultsDisplay;
