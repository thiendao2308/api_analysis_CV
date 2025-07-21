import pandas as pd
import re

# Đọc dữ liệu đã label
df = pd.read_csv('dataset_JD/labeled_sample_batch.csv')

print("=== KIỂM TRA CHẤT LƯỢNG DỮ LIỆU ===")
print(f"Tổng số dòng: {len(df)}")

# Kiểm tra skills column
skills_column = df['skills']
print(f"\nSố dòng có skills: {skills_column.notna().sum()}")
print(f"Số dòng trống: {skills_column.isna().sum()}")

# Kiểm tra độ dài skills
df['skills_length'] = df['skills'].str.len()
print(f"\nĐộ dài trung bình skills: {df['skills_length'].mean():.1f} ký tự")
print(f"Độ dài ngắn nhất: {df['skills_length'].min()}")
print(f"Độ dài dài nhất: {df['skills_length'].max()}")

# Kiểm tra số skills per JD
df['skills_count'] = df['skills'].str.count(',') + 1
print(f"\nSố skills trung bình per JD: {df['skills_count'].mean():.1f}")
print(f"Số skills ít nhất: {df['skills_count'].min()}")
print(f"Số skills nhiều nhất: {df['skills_count'].max()}")

# Hiển thị một số ví dụ
print("\n=== VÍ DỤ DỮ LIỆU ===")
for i in range(min(5, len(df))):
    print(f"\nJD {i+1}:")
    print(f"Text: {df.iloc[i]['text'][:100]}...")
    print(f"Skills: {df.iloc[i]['skills']}")

# Kiểm tra skills phổ biến
all_skills = []
for skills in df['skills'].dropna():
    skills_list = [s.strip() for s in skills.split(',')]
    all_skills.extend(skills_list)

from collections import Counter
skill_counts = Counter(all_skills)
print(f"\n=== TOP 10 SKILLS PHỔ BIẾN ===")
for skill, count in skill_counts.most_common(10):
    print(f"{skill}: {count} lần") 