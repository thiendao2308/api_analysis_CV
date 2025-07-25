"""
CV-JD Matching Model using Fine-tuned BERT
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
import re
import math
import spacy
from ..services.cv_parser import CVParser
from ..services.suggestion_generator import SuggestionGenerator
from ..services.cv_quality_analyzer import CVQualityAnalyzer
from .cv_analyzer import CVAnalysisResult, SectionAnalysis, KeywordAnalysis
from .shared_models import ParsedCV

logger = logging.getLogger(__name__)

class CVJDDataset(Dataset):
    """Dataset for CV-JD matching"""
    
    def __init__(self, cv_texts: List[str], jd_texts: List[str], labels: List[int], 
                 tokenizer, max_length: int = 512):
        self.cv_texts = cv_texts
        self.jd_texts = jd_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.cv_texts)
        
    def __getitem__(self, idx):
        cv_text = self.cv_texts[idx]
        jd_text = self.jd_texts[idx]
        label = self.labels[idx]
        
        # Combine CV and JD with special tokens
        combined_text = f"[CLS] {cv_text} [SEP] {jd_text} [SEP]"
        
        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CVJDMatcher(nn.Module):
    """BERT-based model for CV-JD matching"""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 2):
        super(CVJDMatcher, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Classification head
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            
        return loss, logits

class CVJDMatcherTrainer:
    """Trainer for CV-JD matching model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = CVJDMatcher(config.model_name).to(self.device)
        
    def prepare_data(self, cv_texts: List[str], jd_texts: List[str], 
                    labels: List[int], train_ratio: float = 0.8):
        """Prepare train/val/test datasets"""
        
        # Split data
        n = len(cv_texts)
        train_size = int(n * train_ratio)
        val_size = int(n * 0.1)
        
        # Create datasets
        train_dataset = CVJDDataset(
            cv_texts[:train_size], 
            jd_texts[:train_size], 
            labels[:train_size],
            self.tokenizer,
            self.config.max_length
        )
        
        val_dataset = CVJDDataset(
            cv_texts[train_size:train_size+val_size],
            jd_texts[train_size:train_size+val_size],
            labels[train_size:train_size+val_size],
            self.tokenizer,
            self.config.max_length
        )
        
        test_dataset = CVJDDataset(
            cv_texts[train_size+val_size:],
            jd_texts[train_size+val_size:],
            labels[train_size+val_size:],
            self.tokenizer,
            self.config.max_length
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
        
    def train(self, train_loader, val_loader):
        """Train the model"""
        
        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                loss, _ = self.model(input_ids, attention_mask, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}")
            logger.info(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Early stopping
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f"{self.config.model_name}_best.pt")
            else:
                patience_counter += 1
                
            if patience_counter >= 5:
                logger.info("Early stopping triggered")
                break
                
    def evaluate(self, data_loader):
        """Evaluate the model"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                _, logits = self.model(input_ids, attention_mask)
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
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
    def predict(self, cv_text: str, jd_text: str) -> float:
        """Predict matching score for a CV-JD pair"""
        self.model.eval()
        
        # Prepare input
        combined_text = f"[CLS] {cv_text} [SEP] {jd_text} [SEP]"
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            _, logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            matching_score = probabilities[0][1].item()  # Probability of match
            
        return matching_score
        
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'tokenizer': self.tokenizer
        }, path)
        
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']

# --- NEW: DETAILED ANALYSIS PIPELINE ---

class DetailedCVAnalyzer:
    """
    Điều phối việc phân tích chi tiết một CV so với Mô tả công việc (JD) và Yêu cầu công việc (JR).
    Sử dụng các dịch vụ thông minh như CVParser, CVQualityAnalyzer và SuggestionGenerator.
    """

    def __init__(self, model_path: str = "vi_core_news_lg"):
        """Khởi tạo tất cả các trình phân tích và dịch vụ cần thiết."""
        try:
            self.cv_parser = CVParser(model_path)
            self.suggestion_generator = SuggestionGenerator()
            self.quality_analyzer = CVQualityAnalyzer()
            # Nạp mô hình spaCy NER đã train để phân tích JD
            self.jd_ner_model = spacy.load("ml_architecture/spacy_models/model-best")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo DetailedCVAnalyzer: {e}", exc_info=True)
            raise

    def analyze(self, cv_content: str, jd_text: str, jr_text: Optional[str] = None) -> CVAnalysisResult:
        """
        Thực hiện phân tích sâu CV so với JD/JR.
        """
        logger.info("Bắt đầu phân tích chi tiết CV-JD/JR...")

        # 1. Phân tích nội dung CV
        parsed_cv = self.cv_parser.parse(cv_content)
        logger.info(f"CV đã được phân tích. Các mục tìm thấy: {[s for s in ['summary', 'skills', 'experience', 'education'] if getattr(parsed_cv, s)]}")

        # 2. Phân tích chất lượng cấu trúc CV
        quality_analysis = self.quality_analyzer.analyze(parsed_cv)
        quality_score = quality_analysis.get("quality_score", 0.0)
        strengths = quality_analysis.get("strengths", [])
        logger.info(f"Phân tích chất lượng CV hoàn tất. Điểm: {quality_score}, Điểm mạnh: {strengths}")

        # 3. Gộp JD và JR, sau đó trích xuất các kỹ năng yêu cầu
        full_job_text = jd_text
        if jr_text:
            logger.info("Phát hiện Yêu cầu công việc (JR), đang gộp với Mô tả công việc (JD).")
            full_job_text += f"\\n\\n{jr_text}"
        
        required_skills = self._extract_skills_from_jd(full_job_text)
        logger.info(f"Đã trích xuất {len(required_skills)} kỹ năng yêu cầu từ JD/JR: {required_skills}")

        # 4. So sánh kỹ năng CV với kỹ năng yêu cầu
        cv_skills_normalized = self._normalize_skill_set(parsed_cv.skills)
        required_skills_normalized = self._normalize_skill_set(required_skills)
        
        matched_skills, missing_skills = self._compare_skills(cv_skills_normalized, required_skills_normalized)
        logger.info(f"So sánh hoàn tất. Trùng khớp: {len(matched_skills)}, Còn thiếu: {len(missing_skills)}")

        # 5. Tính điểm dựa trên logic 70/30 mới
        # 5a. Điểm so khớp (trị giá 70%)
        matching_score = 0.0
        if required_skills_normalized:
            # Sử dụng căn bậc hai để cho điểm "hào phóng" hơn ở những kết quả khớp đầu tiên
            match_ratio = len(matched_skills) / len(required_skills_normalized)
            matching_score = math.sqrt(match_ratio)
        matching_score = min(matching_score, 1.0) # Đảm bảo không vượt quá 1.0

        # 5b. Điểm tổng kết có trọng số
        overall_score = (matching_score * 0.7) + (quality_score * 0.3)
        logger.info(f"Điểm đã tính: Điểm khớp={matching_score:.2f}, Điểm chất lượng={quality_score:.2f} -> Điểm tổng thể={overall_score:.2f}")

        # 6. Tạo gợi ý như con người
        matched_skill_names = [skill for skill in required_skills if self._normalize_skill(skill) in matched_skills]
        missing_skill_names = [skill for skill in required_skills if self._normalize_skill(skill) in missing_skills]
        suggestions = self.suggestion_generator.generate(matched_skill_names, missing_skill_names)

        # 7. Cấu trúc kết quả cuối cùng theo mô hình mới
        final_result = CVAnalysisResult(
            overall_score=overall_score,
            strengths=strengths,
            section_analysis=[
                SectionAnalysis(section_name="Tóm tắt", content=parsed_cv.summary),
                SectionAnalysis(section_name="Kỹ năng", content=", ".join(parsed_cv.skills)),
                SectionAnalysis(section_name="Kinh nghiệm", content=parsed_cv.experience),
                SectionAnalysis(section_name="Học vấn", content=parsed_cv.education),
            ],
            keyword_analysis=KeywordAnalysis(
                required_keywords=list(required_skills),
                matched_keywords=matched_skill_names,
                missing_keywords=missing_skill_names,
                suggestions=suggestions
            )
        )
        
        logger.info("Phân tích chi tiết hoàn tất.")
        return final_result

    def _extract_skills_from_jd(self, jd_text: str) -> List[str]:
        """Trích xuất các entity (SKILL, POSITION, BENEFIT, ...) từ JD bằng mô hình spaCy NER đã train."""
        doc = self.jd_ner_model(jd_text)
        # Lấy tất cả entity types mà mô hình đã học
        allowed_labels = set(self.jd_ner_model.get_pipe("ner").labels)
        found_entities = set()
        for ent in doc.ents:
            if ent.label_ in allowed_labels:
                found_entities.add(ent.text.strip())
        return list(found_entities)

    def _normalize_skill(self, skill: str) -> str:
        """Chuẩn hóa một chuỗi kỹ năng để so sánh hiệu quả."""
        # Xóa nội dung trong dấu ngoặc đơn, ví dụ: "Git (Github)" -> "Git"
        skill = re.sub(r'\s*\(.*\)\s*', '', skill)
        # Chuyển thành chữ thường và xóa khoảng trắng
        return skill.lower().strip()

    def _normalize_skill_set(self, skills: List[str]) -> Set[str]:
        """Chuẩn hóa một danh sách các kỹ năng thành một tập hợp để tra cứu hiệu quả."""
        if not skills:
            return set()
        return {self._normalize_skill(s) for s in skills}

    def _compare_skills(self, cv_skills: Set[str], required_skills: Set[str]) -> Tuple[Set[str], Set[str]]:
        """So sánh hai tập hợp các kỹ năng đã được chuẩn hóa."""
        if not required_skills:
            return set(), set()
        
        matched = cv_skills.intersection(required_skills)
        missing = required_skills.difference(cv_skills)
        
        return matched, missing

# Example usage
if __name__ == "__main__":
    # Sample data
    cv_texts = [
        "Experienced Python developer with 5 years in machine learning",
        "Frontend developer with React and JavaScript experience",
        "Data scientist with expertise in SQL and Python"
    ]
    
    jd_texts = [
        "Looking for Python developer with ML experience",
        "Frontend developer position with React skills",
        "Data analyst role requiring SQL knowledge"
    ]
    
    labels = [1, 1, 0]  # 1 for match, 0 for no match
    
    # Train model
    config = CVJDMatchingConfig()
    trainer = CVJDMatcherTrainer(config)
    
    train_loader, val_loader, test_loader = trainer.prepare_data(
        cv_texts, jd_texts, labels
    )
    
    trainer.train(train_loader, val_loader)
    
    # Test prediction
    score = trainer.predict(
        "Python developer with ML experience",
        "Looking for machine learning engineer"
    )
    print(f"Matching score: {score:.3f}") 