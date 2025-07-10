import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader as TorchDataLoader

# Import from project modules
from .config.model_config import MODEL_CONFIGS, TRAINING_CONFIG
from .data.data_pipeline import DataCollector, DataPreprocessor
from .models.cv_jd_matcher import CVJDMatcher, CVJDDataset
from .training.trainer import CVJDTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training():
    """
    Main function to run the training pipeline.
    """
    logger.info("Starting training pipeline...")

    # --- 1. Load Configuration ---
    logger.info("Loading configurations...")
    cv_jd_config = MODEL_CONFIGS['cv_jd_matching']
    training_config = TRAINING_CONFIG

    # --- 2. Load Data ---
    logger.info("Generating synthetic data for training pipeline validation...")
    data_collector = DataCollector()
    raw_data = data_collector.generate_synthetic_data(num_samples=500) # Using 500 samples for a quick test
    logger.info(f"Generated {len(raw_data)} synthetic CV-JD pairs.")


    # --- 3. Preprocess Data ---
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    # The full preprocessing pipeline in data_pipeline.py is extensive.
    # For the BERT model, we just need to clean the text.
    raw_data['cv_text'] = raw_data['cv_text'].apply(preprocessor.clean_text)
    raw_data['jd_text'] = raw_data['jd_text'].apply(preprocessor.clean_text)


    # --- 4. Split Data ---
    logger.info("Splitting data...")
    train_df, test_df = train_test_split(
        raw_data,
        test_size=training_config.test_split,
        random_state=training_config.random_seed,
        stratify=raw_data['is_match']
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=training_config.val_split / (1.0 - training_config.test_split),
        random_state=training_config.random_seed,
        stratify=train_df['is_match']
    )
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    # --- 5. Create Datasets and DataLoaders ---
    logger.info("Creating datasets and dataloaders...")
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
    logger.info("Initializing model and trainer...")
    model = CVJDMatcher(model_name=cv_jd_config.model_name, num_classes=2)
    trainer = CVJDTrainer(config=cv_jd_config, model=model)

    # --- 7. Train Model ---
    logger.info("Starting model training...")
    training_results = trainer.train(train_loader, val_loader)
    logger.info(f"Training finished. Results: {training_results}")

    # --- 8. Evaluate Model ---
    logger.info("Evaluating model on the test set...")
    # Load the best model saved during training
    best_model_path = "models/cv_jd_matcher_best.pt" # As defined in trainer.py
    if Path(best_model_path).exists():
        saved_checkpoint = torch.load(best_model_path)
        model.load_state_dict(saved_checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from {best_model_path}")
    else:
        logger.warning("No best model found. Evaluating with the last state of the model.")

    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test set evaluation metrics: {test_metrics}")

    # --- 9. Save final artifacts ---
    # The trainer already saves the model with MLflow. We can add more here if needed.
    # For example, saving the test metrics.
    # Note: MLflow logging should be handled inside the trainer for consistency.
    # This is a final summary log.
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    run_training() 