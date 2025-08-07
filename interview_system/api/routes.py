from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Optional
import logging
from datetime import datetime

from ..models.session import (
    InterviewSession, SessionResponse, SessionEvaluation, 
    SessionScore, AudioAnalysis
)
from ..services.interview_simulator import InterviewSimulator
from ..services.llm_evaluator import LLMEvaluator
from ..services.audio_processor import AudioProcessor
from ..services.speech_analyzer import SpeechAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services with error handling
try:
    interview_simulator = InterviewSimulator()
    llm_evaluator = LLMEvaluator()
    audio_processor = AudioProcessor()
    speech_analyzer = SpeechAnalyzer()
except Exception as e:
    logger.error(f"Error initializing services: {e}")
    # Create mock services for testing
    interview_simulator = None
    llm_evaluator = None
    audio_processor = None
    speech_analyzer = None

@router.post("/generate-interview", response_model=InterviewSession)
async def generate_interview_session(
    job_category: str = Form(...),
    job_position: str = Form(...),
    jd_skills: List[str] = Form(...),
    cv_skills: List[str] = Form(...),
    missing_skills: List[str] = Form(...),
    overall_score: float = Form(...)
):
    """Generate a new interview session based on JD and CV analysis"""
    
    try:
        if interview_simulator is None:
            raise HTTPException(status_code=500, detail="Interview simulator not available")
        
        session_data = interview_simulator.generate_interview_session(
            job_category=job_category,
            job_position=job_position,
            jd_skills=jd_skills,
            cv_skills=cv_skills,
            missing_skills=missing_skills,
            overall_score=overall_score
        )
        
        # Convert to InterviewSession model
        session = InterviewSession(**session_data)
        
        logger.info(f"Generated interview session: {session.session_id}")
        return session
        
    except Exception as e:
        logger.error(f"Error generating interview session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate-text-response", response_model=SessionEvaluation)
async def evaluate_text_response(
    question: Dict = Form(...),
    user_response: str = Form(...),
    job_category: str = Form(...),
    difficulty: str = Form(...)
):
    """Evaluate a text-based interview response"""
    
    try:
        if llm_evaluator is None:
            raise HTTPException(status_code=500, detail="LLM evaluator not available")
        
        evaluation_data = llm_evaluator.evaluate_response(
            question=question,
            user_response=user_response,
            job_category=job_category,
            difficulty=difficulty
        )
        
        # Convert to SessionEvaluation model
        evaluation = SessionEvaluation(**evaluation_data)
        
        logger.info(f"Evaluated text response for question {evaluation.question_id}")
        return evaluation
        
    except Exception as e:
        logger.error(f"Error evaluating text response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate-audio-response", response_model=SessionEvaluation)
async def evaluate_audio_response(
    question: Dict = Form(...),
    audio_file: UploadFile = File(...),
    job_category: str = Form(...),
    difficulty: str = Form(...)
):
    """Evaluate an audio-based interview response"""
    
    try:
        if audio_processor is None or speech_analyzer is None:
            raise HTTPException(status_code=500, detail="Audio processing services not available")
        
        # Validate audio file
        if not audio_processor.validate_audio_format(audio_file.filename):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Process audio file
        audio_data = await audio_file.read()
        audio_result = audio_processor.process_audio_data(audio_data)
        
        if not audio_result.get("success", False):
            raise HTTPException(status_code=400, detail="Audio processing failed")
        
        # Analyze speech patterns
        speech_analysis = speech_analyzer.analyze_speech_patterns(
            transcript=audio_result["transcript"],
            audio_metrics=audio_result.get("audio_metrics")
        )
        
        # Evaluate response using LLM
        evaluation_data = llm_evaluator.evaluate_response(
            question=question,
            user_response=audio_result["transcript"],
            job_category=job_category,
            difficulty=difficulty
        )
        
        # Combine audio analysis with evaluation
        evaluation_data.update({
            "audio_analysis": audio_result,
            "speech_analysis": speech_analysis
        })
        
        # Convert to SessionEvaluation model
        evaluation = SessionEvaluation(**evaluation_data)
        
        logger.info(f"Evaluated audio response for question {evaluation.question_id}")
        return evaluation
        
    except Exception as e:
        logger.error(f"Error evaluating audio response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/calculate-session-score", response_model=SessionScore)
async def calculate_session_score(evaluations: List[Dict] = Form(...)):
    """Calculate overall session score from multiple evaluations"""
    
    try:
        if llm_evaluator is None:
            raise HTTPException(status_code=500, detail="LLM evaluator not available")
        
        score_data = llm_evaluator.calculate_session_score(evaluations)
        
        # Convert to SessionScore model
        session_score = SessionScore(**score_data)
        
        logger.info(f"Calculated session score: {session_score.overall_score}")
        return session_score
        
    except Exception as e:
        logger.error(f"Error calculating session score: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-audio", response_model=AudioAnalysis)
async def analyze_audio_only(audio_file: UploadFile = File(...)):
    """Analyze audio file without evaluation (for testing)"""
    
    try:
        if audio_processor is None or speech_analyzer is None:
            raise HTTPException(status_code=500, detail="Audio processing services not available")
        
        # Validate audio file
        if not audio_processor.validate_audio_format(audio_file.filename):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Process audio file
        audio_data = await audio_file.read()
        audio_result = audio_processor.process_audio_data(audio_data)
        
        if not audio_result.get("success", False):
            raise HTTPException(status_code=400, detail="Audio processing failed")
        
        # Analyze speech patterns
        speech_analysis = speech_analyzer.analyze_speech_patterns(
            transcript=audio_result["transcript"],
            audio_metrics=audio_result.get("audio_metrics")
        )
        
        # Combine results
        analysis_data = {
            **audio_result,
            "speech_analysis": speech_analysis
        }
        
        # Convert to AudioAnalysis model
        analysis = AudioAnalysis(**analysis_data)
        
        logger.info(f"Audio analysis completed. Confidence: {analysis.confidence}")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific interview session"""
    
    try:
        # TODO: Implement database retrieval
        # For now, return mock data
        session_info = {
            "session_id": session_id,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "total_questions": 8,
            "completed_questions": 8,
            "overall_score": 85.5
        }
        
        return session_info
        
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "interview-system",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "interview_simulator": "active" if interview_simulator else "inactive",
            "llm_evaluator": "active" if llm_evaluator else "inactive",
            "audio_processor": "active" if audio_processor else "inactive",
            "speech_analyzer": "active" if speech_analyzer else "inactive"
        }
    }

@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Interview Simulation System API",
        "version": "1.0.0",
        "endpoints": [
            "POST /generate-interview",
            "POST /evaluate-text-response",
            "POST /evaluate-audio-response",
            "POST /calculate-session-score",
            "POST /analyze-audio",
            "GET /session/{session_id}",
            "GET /health"
        ],
        "features": [
            "Interview session generation",
            "Text response evaluation",
            "Audio response evaluation",
            "Speech pattern analysis",
            "Session scoring",
            "Audio processing"
        ]
    } 