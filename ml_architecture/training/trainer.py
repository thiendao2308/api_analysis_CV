"""
Training Pipeline with MLflow Integration
"""

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import os
from pathlib import Path
import json
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)

class MLTrainer:
    """Base trainer class with MLflow integration"""
    
    def __init__(self, config, model, experiment_name: str = "cv_analyzer"):
        self.config = config
        self.model = model
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # MLflow setup
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        
    def log_parameters(self):
        """Log model parameters to MLflow"""
        params = {
            'model_name': self.config.model_name,
            'model_type': self.config.model_type,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'max_length': self.config.max_length,
            'num_epochs': self.config.num_epochs,
            'device': self.config.device
        }
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow"""
        for name, value in metrics.items():
            if step is not None:
                mlflow.log_metric(name, value, step=step)
            else:
                mlflow.log_metric(name, value)
                
    def save_model(self, model_path: str):
        """Save model with MLflow"""
        mlflow.pytorch.log_model(self.model, "model")
        
        # Also save locally
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_path': model_path
        }, model_path)
        
    def train(self, train_loader: TorchDataLoader, val_loader: TorchDataLoader) -> Dict[str, Any]:
        """Train the model with MLflow logging"""
        raise NotImplementedError("Subclasses must implement train method")
        
    def evaluate(self, test_loader: TorchDataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        raise NotImplementedError("Subclasses must implement evaluate method")

class CVJDTrainer(MLTrainer):
    """Trainer for CV-JD matching model"""
    
    def __init__(self, config, model, experiment_name: str = "cv_jd_matching"):
        super().__init__(config, model, experiment_name)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, train_loader: TorchDataLoader, val_loader: TorchDataLoader) -> Dict[str, Any]:
        """Train the CV-JD matching model"""
        
        with mlflow.start_run():
            # Log parameters
            self.log_parameters()
            
            # Setup optimizer and scheduler
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            total_steps = len(train_loader) * self.config.num_epochs
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps
            )
            
            # Training loop
            best_val_f1 = 0.0
            patience_counter = 0
            training_history = []
            
            for epoch in range(self.config.num_epochs):
                # Training phase
                train_metrics = self._train_epoch(train_loader, optimizer, scheduler)
                
                # Validation phase
                val_metrics = self._validate_epoch(val_loader)
                
                # Log metrics
                self.log_metrics({
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'val_auc': val_metrics['auc']
                }, step=epoch)
                
                # Save training history
                training_history.append({
                    'epoch': epoch,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                })
                
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
                
                # Early stopping
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    patience_counter = 0
                    
                    # Save best model
                    model_path = f"models/cv_jd_matcher_best.pt"
                    self.save_model(model_path)
                    mlflow.log_artifact(model_path)
                    
                else:
                    patience_counter += 1
                    
                if patience_counter >= 5:  # Use config.patience
                    logger.info("Early stopping triggered")
                    break
                    
            # Log final metrics
            final_metrics = {
                'best_val_f1': best_val_f1,
                'final_epoch': epoch + 1,
                'training_history': training_history
            }
            
            mlflow.log_dict(training_history, "training_history.json")
            
            return final_metrics
            
    def _train_epoch(self, train_loader: TorchDataLoader, optimizer, scheduler) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            loss, logits = self.model(input_ids, attention_mask, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy
        }
        
    def _validate_epoch(self, val_loader: TorchDataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                loss, logits = self.model(input_ids, attention_mask, labels)
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )
        auc = roc_auc_score(all_labels, all_predictions)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
    def evaluate(self, test_loader: TorchDataLoader) -> Dict[str, float]:
        """Evaluate the model on test set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                _, logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )
        auc = roc_auc_score(all_labels, all_probabilities)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

class ModelRegistry:
    """Model registry for managing model versions"""
    
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
    def register_model(self, model_name: str, model_path: str, 
                      metrics: Dict[str, float], config: Any) -> str:
        """Register a new model version"""
        
        # Create version directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = self.registry_path / model_name / f"v{timestamp}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        import shutil
        shutil.copy2(model_path, version_dir / "model.pt")
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'version': f"v{timestamp}",
            'created_at': timestamp,
            'metrics': metrics,
            'config': config.__dict__ if hasattr(config, '__dict__') else config
        }
        
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Update latest version
        latest_file = self.registry_path / model_name / "latest.txt"
        with open(latest_file, 'w') as f:
            f.write(f"v{timestamp}")
            
        return f"v{timestamp}"
        
    def get_latest_model(self, model_name: str) -> Optional[str]:
        """Get the latest model version"""
        latest_file = self.registry_path / model_name / "latest.txt"
        if latest_file.exists():
            with open(latest_file, 'r') as f:
                return f.read().strip()
        return None
        
    def load_model(self, model_name: str, version: str = None) -> Dict[str, Any]:
        """Load a specific model version"""
        if version is None:
            version = self.get_latest_model(model_name)
            
        if version is None:
            raise ValueError(f"No model found for {model_name}")
            
        model_dir = self.registry_path / model_name / version
        model_path = model_dir / "model.pt"
        metadata_path = model_dir / "metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return {
            'model_state_dict': checkpoint['model_state_dict'],
            'config': checkpoint['config'],
            'metadata': metadata
        }

# Training script
def train_cv_jd_model(config, train_loader, val_loader, test_loader):
    """Complete training pipeline for CV-JD matching"""
    
    # Initialize model and trainer
    from ml_architecture.models.cv_jd_matcher import CVJDMatcher
    
    model = CVJDMatcher(config.model_name)
    trainer = CVJDTrainer(config, model)
    
    # Train model
    logger.info("Starting training...")
    training_results = trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    # Register model
    registry = ModelRegistry()
    model_path = f"models/cv_jd_matcher_best.pt"
    version = registry.register_model(
        "cv_jd_matcher", 
        model_path, 
        test_metrics, 
        config
    )
    
    logger.info(f"Training completed. Model version: {version}")
    logger.info(f"Test metrics: {test_metrics}")
    
    return {
        'training_results': training_results,
        'test_metrics': test_metrics,
        'model_version': version
    }

if __name__ == "__main__":
    # Example usage
    from ml_architecture.config.model_config import CVJDMatchingConfig
    from ml_architecture.data.data_pipeline import DataCollector, DataPreprocessor, DataLoader
    
    # Setup
    config = CVJDMatchingConfig()
    
    # Generate and prepare data
    collector = DataCollector()
    preprocessor = DataPreprocessor()
    data_loader = DataLoader()
    
    # Generate synthetic data
    raw_data = collector.generate_synthetic_data(num_samples=1000)
    processed_data = preprocessor.preprocess_cv_jd_data(raw_data)
    train_df, val_df, test_df = data_loader.split_data(processed_data)
    
    # Create dataloaders
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    train_dataset = data_loader.create_cv_jd_dataset(train_df, tokenizer, config.max_length)
    val_dataset = data_loader.create_cv_jd_dataset(val_df, tokenizer, config.max_length)
    test_dataset = data_loader.create_cv_jd_dataset(test_df, tokenizer, config.max_length)
    
    train_loader = TorchDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Train model
    results = train_cv_jd_model(config, train_loader, val_loader, test_loader)
    print("Training completed successfully!") 