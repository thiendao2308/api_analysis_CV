import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split

# Đọc dữ liệu đã label
df = pd.read_csv('dataset_JD/labeled_sample_batch.csv')

def create_spacy_format(text, skills_str):
    """Chuyển đổi text và skills thành format spaCy NER"""
    if pd.isna(skills_str) or skills_str.strip() == '':
        return None
    
    # Tách skills thành list
    skills_list = [s.strip() for s in skills_str.split(',') if s.strip()]
    
    # Tìm vị trí của skills trong text
    entities = []
    text_lower = text.lower()
    
    for skill in skills_list:
        skill_lower = skill.lower()
        start = 0
        while True:
            pos = text_lower.find(skill_lower, start)
            if pos == -1:
                break
            # Kiểm tra xem có phải là từ hoàn chỉnh không
            if (pos == 0 or not text[pos-1].isalnum()) and \
               (pos + len(skill) >= len(text) or not text[pos + len(skill)].isalnum()):
                entities.append((pos, pos + len(skill), 'SKILL'))
            start = pos + 1
    
    # Loại bỏ entities trùng lặp
    entities = list(set(entities))
    entities.sort(key=lambda x: x[0])
    
    return {
        'text': text,
        'entities': entities
    }

# Chuyển đổi dữ liệu
ner_data = []
for idx, row in df.iterrows():
    result = create_spacy_format(row['text'], row['skills'])
    if result and result['entities']:  # Chỉ lấy những dòng có entities
        ner_data.append(result)

print(f"Tổng số dòng có skills được tìm thấy: {len(ner_data)}")

# Chia train/test
train_data, test_data = train_test_split(ner_data, test_size=0.2, random_state=42)

# Lưu dữ liệu training
with open('ner_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

# Lưu dữ liệu test
with open('ner_test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"Dữ liệu training: {len(train_data)} dòng")
print(f"Dữ liệu test: {len(test_data)} dòng")

# Hiển thị ví dụ
print("\n=== VÍ DỤ DỮ LIỆU NER ===")
for i, example in enumerate(ner_data[:3]):
    print(f"\nVí dụ {i+1}:")
    print(f"Text: {example['text'][:100]}...")
    print(f"Entities: {example['entities']}") 