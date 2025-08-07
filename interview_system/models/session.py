from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class Question(BaseModel):
    """Model for interview question"""
    id: int
    type: str  # technical, behavioral, situational, culture_fit
    question: str
    difficulty: str  # easy, medium, hard
    expected_points: List[str]
    time_limit: int  # seconds
    scoring_weight: float
    skill: Optional[str] = None  # for technical questions

class InterviewSession(BaseModel):
    """Model for complete interview session"""
    session_id: str
    job_category: str
    job_position: str
    difficulty: str
    questions: List[Question]
    total_duration: int  # seconds
    scoring_criteria: Dict[str, float]
    tips: List[str]
    estimated_questions: int
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = "created"  # created, in_progress, completed

class SessionResponse(BaseModel):
    """Model for user response to a question"""
    question_id: int
    user_response: str
    response_time: int  # seconds
    audio_file_path: Optional[str] = None
    transcript: Optional[str] = None
    confidence: Optional[float] = None
    submitted_at: datetime = Field(default_factory=datetime.now)

class SessionEvaluation(BaseModel):
    """Model for evaluation of a single response"""
    question_id: int
    score: float
    accuracy: float
    specificity: float
    communication: float
    logic: float
    relevance: float
    feedback: Dict[str, str]
    suggestions: List[str]
    overall_rating: str
    question_type: str
    weight: float
    evaluated_at: datetime = Field(default_factory=datetime.now)

class SessionScore(BaseModel):
    """Model for overall session score"""
    session_id: str
    overall_score: float
    category_scores: Dict[str, float]
    recommendations: List[str]
    detailed_feedback: str
    total_questions: int
    score_breakdown: Dict[str, int]
    calculated_at: datetime = Field(default_factory=datetime.now)

class AudioAnalysis(BaseModel):
    """Model for audio analysis results"""
    transcript: str
    confidence: float
    language: str
    duration: float
    quality_score: float
    audio_metrics: Dict[str, float]
    speech_analysis: Optional[Dict] = None

class SpeechAnalysis(BaseModel):
    """Model for speech pattern analysis"""
    text_analysis: Dict
    sentiment_analysis: Dict
    communication_quality: Dict
    audio_analysis: Dict
    overall_score: float 