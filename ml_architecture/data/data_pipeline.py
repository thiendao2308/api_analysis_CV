"""
Data Pipeline for AI CV Analyzer
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import re
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

class DataCollector:
    """Collect and prepare datasets for training"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.nlp = spacy.load("en_core_web_lg")
        
    def collect_cv_jd_pairs(self, sources: List[str]) -> pd.DataFrame:
        """
        Collect CV-JD pairs from various sources
        
        Args:
            sources: List of data sources (files, APIs, etc.)
        """
        data = []
        
        for source in sources:
            if source.endswith('.json'):
                data.extend(self._load_json_data(source))
            elif source.endswith('.csv'):
                data.extend(self._load_csv_data(source))
            elif source.startswith('api://'):
                data.extend(self._load_api_data(source))
                
        return pd.DataFrame(data)
    
    def _load_json_data(self, file_path: str) -> List[Dict]:
        """Load data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_csv_data(self, file_path: str) -> List[Dict]:
        """Load data from CSV file"""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def _load_api_data(self, api_url: str) -> List[Dict]:
        """Load data from API (placeholder)"""
        # Implementation would depend on specific API
        return []
    
    def generate_synthetic_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic CV-JD pairs for training
        """
        # Sample CV templates
        cv_templates = [
            "Experienced {skill} developer with {years} years in {domain}",
            "Senior {skill} engineer specializing in {domain}",
            "Full-stack developer with expertise in {frontend} and {backend}",
            "Data scientist with strong background in {ml_skill} and {data_skill}",
            "DevOps engineer with experience in {cloud} and {container}"
        ]
        
        # Sample JD templates
        jd_templates = [
            "Looking for {skill} developer with {domain} experience",
            "Senior {skill} engineer position in {domain}",
            "Full-stack developer role requiring {frontend} and {backend}",
            "Data scientist position with {ml_skill} and {data_skill} skills",
            "DevOps engineer role with {cloud} and {container} expertise"
        ]
        
        # Skill mappings
        skills = {
            'skill': ['Python', 'Java', 'JavaScript', 'React', 'Node.js'],
            'domain': ['web development', 'machine learning', 'mobile apps', 'cloud computing'],
            'frontend': ['React', 'Vue.js', 'Angular', 'HTML/CSS'],
            'backend': ['Node.js', 'Python', 'Java', 'C#'],
            'ml_skill': ['machine learning', 'deep learning', 'NLP', 'computer vision'],
            'data_skill': ['SQL', 'Pandas', 'NumPy', 'Spark'],
            'cloud': ['AWS', 'Azure', 'GCP', 'Docker'],
            'container': ['Docker', 'Kubernetes', 'ECS', 'AKS']
        }
        
        data = []
        for i in range(num_samples):
            # Generate CV
            cv_template = np.random.choice(cv_templates)
            cv_text = cv_template.format(
                skill=np.random.choice(skills['skill']),
                years=np.random.randint(1, 10),
                domain=np.random.choice(skills['domain']),
                frontend=np.random.choice(skills['frontend']),
                backend=np.random.choice(skills['backend']),
                ml_skill=np.random.choice(skills['ml_skill']),
                data_skill=np.random.choice(skills['data_skill']),
                cloud=np.random.choice(skills['cloud']),
                container=np.random.choice(skills['container'])
            )
            
            # Generate JD
            jd_template = np.random.choice(jd_templates)
            jd_text = jd_template.format(
                skill=np.random.choice(skills['skill']),
                domain=np.random.choice(skills['domain']),
                frontend=np.random.choice(skills['frontend']),
                backend=np.random.choice(skills['backend']),
                ml_skill=np.random.choice(skills['ml_skill']),
                data_skill=np.random.choice(skills['data_skill']),
                cloud=np.random.choice(skills['cloud']),
                container=np.random.choice(skills['container'])
            )
            
            # Determine if it's a match (simple heuristic)
            cv_skills = set(re.findall(r'\b[A-Z][a-z]+\b', cv_text))
            jd_skills = set(re.findall(r'\b[A-Z][a-z]+\b', jd_text))
            match_score = len(cv_skills.intersection(jd_skills)) / max(len(jd_skills), 1)
            is_match = match_score > 0.3
            
            data.append({
                'cv_text': cv_text,
                'jd_text': jd_text,
                'match_score': match_score,
                'is_match': int(is_match),
                'cv_skills': list(cv_skills),
                'jd_skills': list(jd_skills)
            })
            
        return pd.DataFrame(data)

class DataPreprocessor:
    """Preprocess and clean data"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        """
        if not text:
            return ""
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\-\+\&\@\#]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra spaces
        text = text.strip()
        
        return text
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from text using NLP
        """
        doc = self.nlp(text)
        skills = []
        
        # Extract noun phrases that might be skills
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 2 and chunk.text.lower() not in ['the', 'and', 'or']:
                skills.append(chunk.text)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT']:
                skills.append(ent.text)
        
        return list(set(skills))
    
    def extract_experience(self, text: str) -> Dict:
        """
        Extract experience information from text
        """
        # Simple regex patterns for experience extraction
        years_pattern = r'(\d+)\s*(?:years?|yrs?)'
        months_pattern = r'(\d+)\s*(?:months?|mos?)'
        
        years_match = re.search(years_pattern, text.lower())
        months_match = re.search(months_pattern, text.lower())
        
        experience = {
            'years': int(years_match.group(1)) if years_match else 0,
            'months': int(months_match.group(1)) if months_match else 0
        }
        
        return experience
    
    def preprocess_cv_jd_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess CV-JD dataset
        """
        processed_data = []
        
        for _, row in df.iterrows():
            # Clean texts
            cv_clean = self.clean_text(row['cv_text'])
            jd_clean = self.clean_text(row['jd_text'])
            
            # Extract skills
            cv_skills = self.extract_skills(cv_clean)
            jd_skills = self.extract_skills(jd_clean)
            
            # Extract experience
            cv_exp = self.extract_experience(cv_clean)
            jd_exp = self.extract_experience(jd_clean)
            
            processed_data.append({
                'cv_text': cv_clean,
                'jd_text': jd_clean,
                'cv_skills': cv_skills,
                'jd_skills': jd_skills,
                'cv_experience': cv_exp,
                'jd_experience': jd_exp,
                'match_score': row.get('match_score', 0.0),
                'is_match': row.get('is_match', 0)
            })
            
        return pd.DataFrame(processed_data)

class FeatureEngineer:
    """Engineer features for ML models"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
    def create_text_features(self, texts: List[str]) -> np.ndarray:
        """
        Create TF-IDF features from text
        """
        return self.tfidf_vectorizer.fit_transform(texts).toarray()
    
    def create_skill_overlap_features(self, cv_skills: List[List[str]], 
                                    jd_skills: List[List[str]]) -> np.ndarray:
        """
        Create features based on skill overlap
        """
        features = []
        
        for cv_skill_list, jd_skill_list in zip(cv_skills, jd_skills):
            cv_set = set(cv_skill_list)
            jd_set = set(jd_skill_list)
            
            overlap = len(cv_set.intersection(jd_set))
            union = len(cv_set.union(jd_set))
            
            jaccard = overlap / union if union > 0 else 0
            coverage = overlap / len(jd_set) if len(jd_set) > 0 else 0
            
            features.append([overlap, jaccard, coverage])
            
        return np.array(features)
    
    def create_experience_features(self, cv_exp: List[Dict], 
                                 jd_exp: List[Dict]) -> np.ndarray:
        """
        Create features based on experience matching
        """
        features = []
        
        for cv, jd in zip(cv_exp, jd_exp):
            cv_total_months = cv['years'] * 12 + cv['months']
            jd_total_months = jd['years'] * 12 + jd['months']
            
            experience_match = 1 if cv_total_months >= jd_total_months else 0
            experience_ratio = cv_total_months / max(jd_total_months, 1)
            
            features.append([experience_match, experience_ratio])
            
        return np.array(features)

class DataLoader:
    """Load and prepare data for training"""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        
    def create_cv_jd_dataset(self, df: pd.DataFrame, 
                           tokenizer, max_length: int = 512) -> Dataset:
        """
        Create PyTorch dataset for CV-JD matching
        """
        class CVJDDataset(Dataset):
            def __init__(self, cv_texts, jd_texts, labels, tokenizer, max_length):
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
                
                # Combine CV and JD
                combined_text = f"[CLS] {cv_text} [SEP] {jd_text} [SEP]"
                
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
        
        return CVJDDataset(
            df['cv_text'].tolist(),
            df['jd_text'].tolist(),
            df['is_match'].tolist(),
            tokenizer,
            max_length
        )
    
    def split_data(self, df: pd.DataFrame, 
                  train_ratio: float = 0.8, 
                  val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets
        """
        train_df, temp_df = train_test_split(
            df, train_size=train_ratio, random_state=42, stratify=df['is_match']
        )
        
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df, train_size=val_ratio_adjusted, random_state=42, stratify=temp_df['is_match']
        )
        
        return train_df, val_df, test_df

# Example usage
if __name__ == "__main__":
    # Initialize components
    collector = DataCollector()
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    data_loader = DataLoader()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    raw_data = collector.generate_synthetic_data(num_samples=1000)
    
    # Preprocess data
    print("Preprocessing data...")
    processed_data = preprocessor.preprocess_cv_jd_data(raw_data)
    
    # Split data
    print("Splitting data...")
    train_df, val_df, test_df = data_loader.split_data(processed_data)
    
    # Save processed data
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Val set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print("Data pipeline completed!") 