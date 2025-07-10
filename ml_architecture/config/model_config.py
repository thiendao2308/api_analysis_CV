"""
Model Configuration for AI CV Analyzer
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch

@dataclass
class ModelConfig:
    """Base configuration for all models"""
    model_name: str
    model_type: str
    version: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
@dataclass
class CVJDMatchingConfig(ModelConfig):
    """Configuration for CV-JD Matching Model"""
    model_name: str = "bert-base-uncased"
    model_type: str = "transformer"
    version: str = "1.0"
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 512
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Model architecture
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_layers: int = 12
    dropout: float = 0.1
    
    # Loss function
    loss_function: str = "binary_cross_entropy"
    
    # Evaluation metrics
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1", "auc"]

@dataclass
class KeywordExtractionConfig(ModelConfig):
    """Configuration for Keyword Extraction Model"""
    model_name: str = "bert-base-uncased"
    model_type: str = "ner"
    version: str = "1.0"
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 3e-5
    max_length: int = 256
    num_epochs: int = 15
    
    # NER specific
    num_labels: int = 5  # O, B-SKILL, I-SKILL, B-EXP, I-EXP
    label2id: Dict[str, int] = None
    id2label: Dict[int, str] = None
    
    # Keyword expansion
    expansion_threshold: float = 0.7
    max_expansions: int = 10
    
    def __post_init__(self):
        if self.label2id is None:
            self.label2id = {
                "O": 0,
                "B-SKILL": 1,
                "I-SKILL": 2,
                "B-EXP": 3,
                "I-EXP": 4
            }
        if self.id2label is None:
            self.id2label = {v: k for k, v in self.label2id.items()}

@dataclass
class QualityAssessmentConfig(ModelConfig):
    """Configuration for CV Quality Assessment Model"""
    model_name: str = "distilbert-base-uncased"
    model_type: str = "multi_label_classification"
    version: str = "1.0"
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 512
    num_epochs: int = 12
    
    # Quality dimensions
    quality_dimensions: List[str] = None
    num_classes: int = 8
    
    # Thresholds
    quality_threshold: float = 0.7
    format_threshold: float = 0.8
    content_threshold: float = 0.75
    
    def __post_init__(self):
        if self.quality_dimensions is None:
            self.quality_dimensions = [
                "format_quality",
                "content_quality", 
                "grammar_quality",
                "structure_quality",
                "ats_compatibility",
                "professionalism",
                "completeness",
                "clarity"
            ]

@dataclass
class GrammarCorrectionConfig(ModelConfig):
    """Configuration for Grammar Correction Model"""
    model_name: str = "t5-base"
    model_type: str = "sequence_to_sequence"
    version: str = "1.0"
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 5e-5
    max_length: int = 512
    num_epochs: int = 20
    
    # T5 specific
    max_source_length: int = 256
    max_target_length: int = 256
    padding: str = "max_length"
    truncation: bool = True
    
    # Generation parameters
    num_beams: int = 4
    early_stopping: bool = True
    length_penalty: float = 2.0

@dataclass
class RecommendationConfig(ModelConfig):
    """Configuration for Recommendation Model"""
    model_name: str = "collaborative_filtering"
    model_type: str = "recommendation"
    version: str = "1.0"
    
    # Model parameters
    embedding_dim: int = 128
    num_factors: int = 64
    regularization: float = 0.01
    
    # Training parameters
    batch_size: int = 256
    learning_rate: float = 0.001
    num_epochs: int = 100
    
    # Recommendation parameters
    top_k: int = 10
    similarity_threshold: float = 0.6
    min_interactions: int = 5

@dataclass
class TrainingConfig:
    """Global training configuration"""
    # Data
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    
    # Training
    num_workers: int = 4
    pin_memory: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    
    # Checkpointing
    save_dir: str = "checkpoints"
    load_best_model: bool = True
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001

@dataclass
class InferenceConfig:
    """Inference configuration"""
    # Model serving
    batch_size: int = 1
    max_length: int = 512
    temperature: float = 1.0
    
    # Caching
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour
    
    # Performance
    num_threads: int = 4
    enable_optimization: bool = True
    
    # Monitoring
    enable_metrics: bool = True
    enable_logging: bool = True

# Model registry
MODEL_CONFIGS = {
    "cv_jd_matching": CVJDMatchingConfig(),
    "keyword_extraction": KeywordExtractionConfig(),
    "quality_assessment": QualityAssessmentConfig(),
    "grammar_correction": GrammarCorrectionConfig(),
    "recommendation": RecommendationConfig()
}

# Training configuration
TRAINING_CONFIG = TrainingConfig()

# Inference configuration  
INFERENCE_CONFIG = InferenceConfig() 