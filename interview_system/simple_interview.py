from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import uvicorn
import json
from datetime import datetime
import uuid
import os

# Set environment variables
os.environ['PYTHONUNBUFFERED'] = '1'

app = FastAPI(
    title="Simple Interview System",
    description="AI-powered interview simulation with single endpoint",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Simple Interview System",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "Server is running successfully!",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Simple Interview System is running",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/interview")
async def interview_workflow(
    jd_content: str = Form(...),
    job_category: str = Form(...),
    job_position: str = Form(...),
    user_response: str = Form(...),
    audio_file: Optional[UploadFile] = File(None)
):
    """
    Complete interview workflow in one endpoint:
    1. Process JD and job info
    2. Generate question using OpenAI (mock)
    3. Process user response
    4. Evaluate response using OpenAI (mock)
    """
    
    try:
        # Step 1: Process JD and generate question
        prompt = f"""
        Based on the following job description for {job_position} in {job_category}:
        
        {jd_content}
        
        Generate one professional interview question that tests the candidate's technical skills and knowledge relevant to this position.
        """
        
        # Mock OpenAI question generation
        generated_question = f"What specific technical challenges have you faced while working with {job_category} technologies, and how did you resolve them?"
        
        # Step 2: Process user response
        response_text = user_response
        audio_transcription = None
        
        if audio_file:
            # Mock audio processing
            audio_transcription = "User provided audio response (transcription would be here)"
            response_text = audio_transcription if audio_transcription else user_response
        
        # Step 3: Evaluate response
        evaluation_prompt = f"""
        Evaluate the following interview response for a {job_position} position in {job_category}:
        
        Question: {generated_question}
        User Response: {response_text}
        
        Provide evaluation with score, feedback, and suggestions.
        """
        
        # Mock OpenAI evaluation
        evaluation_score = 8.5
        feedback = "Good response showing technical knowledge. The candidate demonstrates understanding of the subject matter with relevant examples."
        suggestions = [
            "Provide more specific technical details",
            "Include quantifiable results or metrics", 
            "Explain the reasoning behind your approach"
        ]
        
        # Return complete workflow result
        result = {
            "status": "success",
            "message": "Interview workflow completed successfully",
            "data": {
                "session_id": f"session_{uuid.uuid4().hex[:8]}",
                "job_info": {
                    "category": job_category,
                    "position": job_position,
                    "jd_preview": jd_content[:200] + "..." if len(jd_content) > 200 else jd_content
                },
                "generated_question": generated_question,
                "user_response": response_text,
                "evaluation": {
                    "score": evaluation_score,
                    "feedback": feedback,
                    "suggestions": suggestions
                },
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/interview-text-only")
async def interview_text_only(
    jd_content: str = Form(...),
    job_category: str = Form(...),
    job_position: str = Form(...),
    user_response: str = Form(...)
):
    """
    Simplified interview workflow for text-only responses
    """
    return await interview_workflow(
        jd_content=jd_content,
        job_category=job_category,
        job_position=job_position,
        user_response=user_response
    )

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8007
    
    print("=" * 60)
    print("🚀 SIMPLE INTERVIEW SYSTEM")
    print("=" * 60)
    print(f"📍 Server URL: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"💚 Health Check: http://{host}:{port}/health")
    print("=" * 60)
    print("📋 Main Endpoint:")
    print("   POST /api/v1/interview - Complete workflow")
    print("   POST /api/v1/interview-text-only - Text only")
    print("=" * 60)
    print("✅ Server is starting...")
    print("   Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            reload=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}") 