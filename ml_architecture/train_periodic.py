import logging
from pathlib import Path
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader as TorchDataLoader

from .config.model_config import MODEL_CONFIGS, TRAINING_CONFIG
from .data.data_pipeline import DataPreprocessor
from .models.cv_jd_matcher import CVJDMatcher, CVJDDataset
from .training.trainer import CVJDTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LABELED_DATA_PATH = "ml_architecture/data/labeled/labeled_data.jsonl"


def load_labeled_jsonl(path):
    """Load labeled data from JSONL file and return DataFrame with cv_text, jd_text, is_match (dummy label)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                # For now, create a dummy label (e.g., always 1)
                records.append({
                    "cv_text": obj.get("cv_text"),
                    "jd_text": obj.get("jd_text"),
                    "is_match": 1  # TODO: update if you have real label
                })
            except Exception as e:
                logger.warning(f"Skip line due to error: {e}")
    return pd.DataFrame(records)

def run_periodic_training():
    logger.info("Starting periodic retrain pipeline from labeled data...")
    # --- 1. Load Configuration ---
    cv_jd_config = MODEL_CONFIGS['cv_jd_matching']
    training_config = TRAINING_CONFIG
    # --- 2. Load labeled data ---
    logger.info(f"Loading labeled data from {LABELED_DATA_PATH} ...")
    df = load_labeled_jsonl(LABELED_DATA_PATH)
    if len(df) < 10:
        logger.warning("Not enough labeled data to retrain. Need at least 10 samples.")
        return
    logger.info(f"Loaded {len(df)} labeled samples.")
    # --- 3. Preprocess Data ---
    preprocessor = DataPreprocessor()
    df['cv_text'] = df['cv_text'].apply(preprocessor.clean_text)
    df['jd_text'] = df['jd_text'].apply(preprocessor.clean_text)
    # --- 4. Split Data ---
    train_df, test_df = train_test_split(
        df,
        test_size=training_config.test_split,
        random_state=training_config.random_seed,
        stratify=df['is_match'] if 'is_match' in df else None
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=training_config.val_split / (1.0 - training_config.test_split),
        random_state=training_config.random_seed,
        stratify=train_df['is_match'] if 'is_match' in train_df else None
    )
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    # --- 5. Create Datasets and DataLoaders ---
    tokenizer = AutoTokenizer.from_pretrained(cv_jd_config.model_name)
    train_dataset = CVJDDataset(
        cv_texts=train_df['cv_text'].tolist(),
        jd_texts=train_df['jd_text'].tolist(),
        labels=train_df['is_match'].tolist(),
        tokenizer=tokenizer,
        max_length=cv_jd_config.max_length
    )
    val_dataset = CVJDDataset(
        cv_texts=val_df['cv_text'].tolist(),
        jd_texts=val_df['jd_text'].tolist(),
        labels=val_df['is_match'].tolist(),
        tokenizer=tokenizer,
        max_length=cv_jd_config.max_length
    )
    test_dataset = CVJDDataset(
        cv_texts=test_df['cv_text'].tolist(),
        jd_texts=test_df['jd_text'].tolist(),
        labels=test_df['is_match'].tolist(),
        tokenizer=tokenizer,
        max_length=cv_jd_config.max_length
    )
    train_loader = TorchDataLoader(train_dataset, batch_size=cv_jd_config.batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=cv_jd_config.batch_size)
    test_loader = TorchDataLoader(test_dataset, batch_size=cv_jd_config.batch_size)
    # --- 6. Initialize Model and Trainer ---
    model = CVJDMatcher(model_name=cv_jd_config.model_name, num_classes=2)
    trainer = CVJDTrainer(config=cv_jd_config, model=model)
    # --- 7. Train Model ---
    logger.info("Starting model training...")
    training_results = trainer.train(train_loader, val_loader)
    logger.info(f"Training finished. Results: {training_results}")
    # --- 8. Evaluate Model ---
    logger.info("Evaluating model on the test set...")
    best_model_path = "models/cv_jd_matcher_best.pt"
    if Path(best_model_path).exists():
        import torch
        saved_checkpoint = torch.load(best_model_path)
        model.load_state_dict(saved_checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from {best_model_path}")
    else:
        logger.warning("No best model found. Evaluating with the last state of the model.")
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test set evaluation metrics: {test_metrics}")
    logger.info("Periodic retrain pipeline finished successfully.")

if __name__ == "__main__":
    run_periodic_training()