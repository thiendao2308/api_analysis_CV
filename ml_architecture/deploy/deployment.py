"""
Deployment Script for ML Models
"""

import os
import sys
import logging
import torch
import mlflow
from pathlib import Path
from typing import Dict, Any, Optional
import json
import redis
import time
from datetime import datetime
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.cv_jd_matcher import CVJDMatcher
from config.model_config import MODEL_CONFIGS, INFERENCE_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('cv_analyzer_requests_total', 'Total requests', ['model', 'endpoint'])
REQUEST_LATENCY = Histogram('cv_analyzer_request_duration_seconds', 'Request latency', ['model', 'endpoint'])
MODEL_LOAD_TIME = Gauge('cv_analyzer_model_load_time_seconds', 'Model load time')
ACTIVE_MODELS = Gauge('cv_analyzer_active_models', 'Number of active models')

class ModelManager:
    """Manages model loading, caching, and serving"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=0
        )
        self.cache_ttl = INFERENCE_CONFIG.cache_ttl
        
    def load_model(self, model_name: str, model_path: str) -> Any:
        """Load a model from disk"""
        start_time = time.time()
        
        try:
            if model_name == "cv_jd_matcher":
                config = MODEL_CONFIGS[model_name]
                model = CVJDMatcher(config.model_name)
                
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=config.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(config.device)
                model.eval()
                
                self.models[model_name] = {
                    'model': model,
                    'config': config,
                    'tokenizer': checkpoint.get('tokenizer'),
                    'metadata': checkpoint.get('metadata', {})
                }
                
                load_time = time.time() - start_time
                MODEL_LOAD_TIME.set(load_time)
                ACTIVE_MODELS.inc()
                
                logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
                return self.models[model_name]
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
            
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get a loaded model"""
        return self.models.get(model_name)
        
    def predict(self, model_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction with caching"""
        
        # Check cache first
        cache_key = f"{model_name}:{hash(str(inputs))}"
        cached_result = self.redis_client.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for {model_name}")
            return json.loads(cached_result)
            
        # Get model
        model_info = self.get_model(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not loaded")
            
        # Make prediction
        start_time = time.time()
        
        try:
            if model_name == "cv_jd_matcher":
                result = self._predict_cv_jd_matcher(model_info, inputs)
            else:
                raise ValueError(f"Unknown model: {model_name}")
                
            # Cache result
            self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(result)
            )
            
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(model=model_name, endpoint='predict').observe(latency)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {str(e)}")
            raise

    def _predict_cv_jd_matcher(self, model_info: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction with CV-JD matcher"""
        model = model_info['model']
        config = model_info['config']
        tokenizer = model_info['tokenizer']
        
        cv_text = inputs['cv_text']
        jd_text = inputs['jd_text']
        
        # Prepare input
        combined_text = f"[CLS] {cv_text} [SEP] {jd_text} [SEP]"
        encoding = tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=config.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(config.device)
        attention_mask = encoding['attention_mask'].to(config.device)
        
        # Make prediction
        with torch.no_grad():
            _, logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            matching_score = probabilities[0][1].item()
            
        return {
            'matching_score': matching_score,
            'confidence': max(probabilities[0]).item(),
            'model_version': model_info.get('metadata', {}).get('version', 'unknown')
        }

# FastAPI app
app = FastAPI(title="AI CV Analyzer ML API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model manager
model_manager = ModelManager(INFERENCE_CONFIG.__dict__)

# Pydantic models
class CVAnalysisRequest(BaseModel):
    cv_text: str
    jd_text: str
    model_name: str = "cv_jd_matcher"

class CVAnalysisResponse(BaseModel):
    matching_score: float
    confidence: float
    model_version: str
    processing_time: float

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Loading models...")
    
    # Load CV-JD matcher
    model_path = os.getenv('CV_JD_MODEL_PATH', 'models/cv_jd_matcher_best.pt')
    if os.path.exists(model_path):
        model_manager.load_model("cv_jd_matcher", model_path)
    else:
        logger.warning(f"Model file not found: {model_path}")
        
    logger.info("Startup completed")

@app.post("/predict", response_model=CVAnalysisResponse)
async def predict(request: CVAnalysisRequest):
    """
    Make a prediction based on CV and JD text.
    Returns a mock response if the model is not loaded.
    """
    start_time = time.time()
    REQUEST_COUNT.labels(model=request.model_name, endpoint='predict').inc()

    # --- MOCK RESPONSE LOGIC ---
    # If model is not loaded, return a mock response for UI testing
    if not model_manager.get_model(request.model_name):
        logger.warning(f"Model '{request.model_name}' not found. Returning mock response.")
        # Simulate processing time
        await asyncio.sleep(1.5)
        return CVAnalysisResponse(
            matching_score=0.78,
            confidence=0.92,
            model_version="mock_v0.1",
            processing_time=time.time() - start_time
        )
    # --- END MOCK RESPONSE LOGIC ---
    
    try:
        inputs = {"cv_text": request.cv_text, "jd_text": request.jd_text}
        result = model_manager.predict(request.model_name, inputs)
        
        processing_time = time.time() - start_time
        
        return CVAnalysisResponse(
            **result,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_models": len(model_manager.models)
    }

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": list(model_manager.models.keys()),
        "model_info": {
            name: {
                "config": info['config'].__dict__,
                "metadata": info.get('metadata', {})
            }
            for name, info in model_manager.models.items()
        }
    }

@app.post("/models/{model_name}/reload")
async def reload_model(model_name: str):
    """Reload a specific model"""
    try:
        model_path = os.getenv(f'{model_name.upper()}_MODEL_PATH', f'models/{model_name}_best.pt')
        model_manager.load_model(model_name, model_path)
        return {"message": f"Model {model_name} reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return prometheus_client.generate_latest()

# Background task for model monitoring
async def monitor_models():
    """Monitor model performance and health"""
    while True:
        try:
            for model_name, model_info in model_manager.models.items():
                # Check model health
                model = model_info['model']
                if hasattr(model, 'eval'):
                    model.eval()
                    
                # Log model statistics
                logger.info(f"Model {model_name} is healthy")
                
        except Exception as e:
            logger.error(f"Model monitoring error: {str(e)}")
            
        await asyncio.sleep(60)  # Check every minute

@app.on_event("startup")
async def start_monitoring():
    """Start background monitoring task"""
    asyncio.create_task(monitor_models())

# Docker deployment
def create_dockerfile():
    """Create Dockerfile for deployment"""
    dockerfile_content = """
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY ml_architecture/requirements_ml.txt .
RUN pip install --no-cache-dir -r requirements_ml.txt

# Download spaCy model
RUN python -m spacy download en_core_web_lg

# Copy application code
COPY ml_architecture/ ./ml_architecture/
COPY models/ ./models/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "ml_architecture.deploy.deployment:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
        
    logger.info("Dockerfile created successfully")

# Kubernetes deployment
def create_k8s_manifests():
    """Create Kubernetes manifests"""
    
    deployment_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cv-analyzer-ml
  labels:
    app: cv-analyzer-ml
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cv-analyzer-ml
  template:
    metadata:
      labels:
        app: cv-analyzer-ml
    spec:
      containers:
      - name: cv-analyzer-ml
        image: cv-analyzer-ml:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: CV_JD_MODEL_PATH
          value: "/app/models/cv_jd_matcher_best.pt"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: cv-analyzer-ml-service
spec:
  selector:
    app: cv-analyzer-ml
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
"""
    
    with open("k8s-deployment.yaml", "w") as f:
        f.write(deployment_yaml)
        
    logger.info("Kubernetes manifests created successfully")

if __name__ == "__main__":
    # Create deployment files
    create_dockerfile()
    create_k8s_manifests()
    
    # Start server
    uvicorn.run(
        "ml_architecture.deploy.deployment:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 