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
        {keywords.length > 0 ? (
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
    {suggestions.length > 0 ? (
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
    {sections.length > 0 ? (
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
          {section.suggestions.map((s, i) => (
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

const ResultsDisplay = ({ result, loading }) => {
  const [activeTab, setActiveTab] = useState(0);

  if (loading) {
    return (
      <Box className="loading-container">
        <CircularProgress size={60} />
        <Typography variant="h6" mt={2}>
          AI đang phân tích CV của bạn...
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
          Vui lòng nhập CV và JD để bắt đầu.
        </Typography>
      </Paper>
    );
  }

  const score = Math.round(result.overall_score * 100);
  const keywordAnalysis = result.keyword_analysis || {};

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
              Kết Quả Phân Tích
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
            <Tab label="Tổng quan" iconPosition="start" />
            <Tab label="Phân tích chi tiết" iconPosition="start" />
          </Tabs>
        </Box>
        <Box className="tab-panel">
          {activeTab === 0 && (
            <Grid container spacing={4}>
              {result.strengths && result.strengths.length > 0 && (
                <Grid item xs={12}>
                  <Box sx={{ display: "flex", alignItems: "center" }}>
                    <Typography variant="h6">Điểm mạnh</Typography>
                  </Box>
                  <List>
                    {result.strengths.map((strength, index) => (
                      <ListItem key={index} disableGutters>
                        <ListItemIcon
                          sx={{ minWidth: "auto", mr: 1.5 }}
                        ></ListItemIcon>
                        <ListItemText primary={strength} />
                      </ListItem>
                    ))}
                  </List>
                </Grid>
              )}
              <Grid item xs={12} md={6}>
                <Keywords
                  keywords={keywordAnalysis.matched_keywords || []}
                  title="Từ khóa khớp"
                  color="success"
                  iconType="matched"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Keywords
                  keywords={keywordAnalysis.missing_keywords || []}
                  title="Từ khóa còn thiếu"
                  color="error"
                  iconType="missing"
                />
              </Grid>
              {keywordAnalysis.suggestions &&
                keywordAnalysis.suggestions.length > 0 && (
                  <Grid item xs={12}>
                    <Suggestions
                      suggestions={keywordAnalysis.suggestions}
                      title="Gợi ý cải thiện CV"
                    />
                  </Grid>
                )}
            </Grid>
          )}
          {activeTab === 1 && (
            <Box>
              {result.section_analysis ? (
                <SectionAnalysis sections={result.section_analysis} />
              ) : (
                <Typography>Không có phân tích chi tiết.</Typography>
              )}
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ResultsDisplay;
