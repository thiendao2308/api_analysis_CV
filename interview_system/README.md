# Interview Simulation System

AI-powered interview simulation system based on job descriptions and CV analysis.

## 🚀 Features

- **Interview Generation**: Generate personalized interview questions based on JD and CV
- **Response Evaluation**: AI-powered evaluation of interview responses
- **Audio Processing**: Speech-to-text conversion and analysis
- **Speech Analysis**: Sentiment analysis and communication quality assessment
- **Session Scoring**: Calculate overall interview performance scores
- **Real-time Feedback**: Provide instant feedback and suggestions

## 📁 Architecture

```
interview_system/
├── 📁 services/                    # Core business logic
│   ├── interview_simulator.py     # Generate interview questions
│   ├── llm_evaluator.py          # Evaluate responses using LLM
│   ├── audio_processor.py        # Speech-to-text processing
│   └── speech_analyzer.py        # Speech pattern analysis
├── 📁 models/                     # Data models
│   └── session.py                # Pydantic models for sessions
├── 📁 api/                        # API endpoints
│   └── routes.py                 # FastAPI routes
├── main.py                       # FastAPI application
├── quick_start.py                # Optimized startup script
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

## 🔧 API Endpoints

### Core Endpoints

```
POST /api/v1/generate-interview          # Create interview session
POST /api/v1/evaluate-text-response      # Evaluate text response
POST /api/v1/evaluate-audio-response     # Evaluate audio response
POST /api/v1/calculate-session-score     # Calculate overall score
POST /api/v1/analyze-audio              # Audio-only analysis
GET  /api/v1/session/{session_id}       # Get session info
GET  /api/v1/health                     # Health check
```

### Request Examples

#### Generate Interview Session

```bash
curl -X POST "http://localhost:8001/api/v1/generate-interview" \
  -F "job_category=IT" \
  -F "job_position=Software Engineer" \
  -F "jd_skills=Python,React,Docker" \
  -F "cv_skills=Python,JavaScript" \
  -F "missing_skills=React,Docker" \
  -F "overall_score=75.5"
```

#### Evaluate Text Response

```bash
curl -X POST "http://localhost:8001/api/v1/evaluate-text-response" \
  -F "question={\"id\":1,\"type\":\"technical\",\"question\":\"Explain Python\"}" \
  -F "user_response=I have 3 years of Python experience..." \
  -F "job_category=IT" \
  -F "difficulty=medium"
```

#### Evaluate Audio Response

```bash
curl -X POST "http://localhost:8001/api/v1/evaluate-audio-response" \
  -F "question={\"id\":1,\"type\":\"technical\"}" \
  -F "audio_file=@response.wav" \
  -F "job_category=IT" \
  -F "difficulty=medium"
```

## 🛠️ Local Development

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
cd interview_system/
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Running Locally

```bash
python main.py
# or
python quick_start.py
```

The API will be available at `http://localhost:8001`

### Testing

```bash
# Test structure
python test_new_structure.py

# Test API endpoints
python test_api_endpoints.py
```

## 🚀 Deployment on Render

The system is configured for deployment on Render with the following optimizations:

- Memory optimization environment variables
- CPU-only PyTorch configuration
- Optimized Uvicorn settings
- Reduced concurrency and request limits

### Environment Variables

```bash
# Core settings
PYTHON_VERSION=3.10.13
PORT=8001

# Memory optimization
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
TOKENIZERS_PARALLELISM=false
TRANSFORMERS_OFFLINE=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=""
PYTORCH_NO_CUDA_MEMORY_CACHING=1
PYTORCH_JIT=0
TRANSFORMERS_CACHE=/tmp/transformers_cache

# Build optimization
PIP_NO_CACHE_DIR=1
PIP_DISABLE_PIP_VERSION_CHECK=1
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

## 📊 Key Features

### 1. Interview Generation

- **Multi-industry support**: IT, Marketing, Finance
- **Question types**: Technical, Behavioral, Situational, Culture-fit
- **Difficulty adaptation**: Based on CV-JD match score
- **Personalized questions**: Focus on missing skills and JD requirements

### 2. Response Evaluation

- **LLM-powered evaluation**: Using OpenAI GPT models
- **Multi-criteria scoring**: Accuracy, Specificity, Communication, Logic, Relevance
- **Detailed feedback**: Specific suggestions and improvements
- **Session scoring**: Weighted average across all responses

### 3. Audio Processing

- **Speech-to-text**: Google Speech Recognition + Sphinx fallback
- **Audio quality analysis**: Volume, dynamic range, duration
- **Format support**: WAV, MP3, M4A, FLAC, OGG
- **Audio enhancement**: Noise reduction, normalization

### 4. Speech Analysis

- **Sentiment analysis**: Emotional tone and confidence
- **Communication quality**: Clarity, structure, engagement, professionalism
- **Pattern analysis**: Filler words, hesitation indicators, vocabulary diversity
- **Overall scoring**: Weighted combination of all factors

## 🔄 Data Models

### InterviewSession

```python
{
    "session_id": "session_1234567890",
    "job_category": "IT",
    "job_position": "Software Engineer",
    "difficulty": "medium",
    "questions": [...],
    "total_duration": 300,
    "scoring_criteria": {...},
    "tips": [...],
    "estimated_questions": 8
}
```

### SessionEvaluation

```python
{
    "question_id": 1,
    "score": 85.5,
    "accuracy": 22.0,
    "specificity": 18.0,
    "communication": 17.0,
    "logic": 16.0,
    "relevance": 12.5,
    "feedback": {...},
    "suggestions": [...],
    "overall_rating": "good"
}
```

## 🔮 Future Enhancements

### Phase 2: Advanced Features

- [ ] Database integration (PostgreSQL)
- [ ] Real-time WebSocket communication
- [ ] Video analysis and body language
- [ ] Multi-language support
- [ ] Advanced analytics dashboard

### Phase 3: Enterprise Features

- [ ] User authentication and authorization
- [ ] Interview session persistence
- [ ] Advanced reporting and analytics
- [ ] Integration with HR systems
- [ ] Custom question bank management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 📞 Support

For deployment issues:

1. Check Render logs
2. Verify environment variables
3. Test locally first
4. Review service configuration

## 🎯 Advantages

### 1. **Modularity**

- Separate services for different functionalities
- Easy to test and maintain individual components
- Clear separation of concerns

### 2. **Scalability**

- Services can be deployed independently
- Easy to add new features (e.g., video analysis)
- Database integration ready

### 3. **Professional Architecture**

- Industry-standard folder structure
- Proper dependency management
- Comprehensive error handling

### 4. **Advanced Features**

- **Audio processing**: Real-time speech-to-text
- **Sentiment analysis**: Emotional tone detection
- **Communication quality**: Professional speech analysis
- **Multi-format support**: Text and audio responses

### 5. **Production Ready**

- Optimized for Render deployment
- Memory-efficient configuration
- Comprehensive logging and monitoring

uvicorn interview_system.main:app --host 127.0.0.1 --port 8001 --reload
python -m uvicorn interview_system.main:app --host 127.0.0.1 --port 8001 --reload
uvicorn simple_interview:app --host 127.0.0.1 --port 8008 --reload
