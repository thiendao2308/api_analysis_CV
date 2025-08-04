import time
import psutil
import os

def test_performance():
    print("=== PERFORMANCE TEST ===")
    
    # Test memory usage
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024
    print(f"Start memory: {start_memory:.2f} MB")
    
    # Test spaCy loading
    start_time = time.time()
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024
        print(f"spaCy loading time: {end_time - start_time:.2f} seconds")
        print(f"Memory after spaCy: {end_memory:.2f} MB")
        print(f"Memory increase: {end_memory - start_memory:.2f} MB")
    except Exception as e:
        print(f"spaCy error: {e}")
    
    # Test OpenAI
    start_time = time.time()
    try:
        import openai
        print("OpenAI library available")
        end_time = time.time()
        print(f"OpenAI check time: {end_time - start_time:.2f} seconds")
    except ImportError:
        print("OpenAI not installed")
        end_time = time.time()
        print(f"OpenAI check time: {end_time - start_time:.2f} seconds")
    
    # Test data processing
    start_time = time.time()
    try:
        import pandas as pd
        df = pd.read_csv('data/merged_labeled_dataset.csv')
        end_time = time.time()
        print(f"Data loading time: {end_time - start_time:.2f} seconds")
        print(f"Dataset shape: {df.shape}")
    except Exception as e:
        print(f"Data loading error: {e}")

if __name__ == "__main__":
    test_performance() 