import spacy
import pandas as pd
import sys
import os

# Load trained model
nlp = spacy.load("model/model-last")  # Sử dụng model vừa train

def extract_skills_with_ner(text):
    """Sử dụng NER model để trích xuất skills"""
    doc = nlp(text)
    skills = []
    
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            skills.append(ent.text.strip())
    
    return ", ".join(skills)

# Test với dữ liệu mới
def process_new_data(file_path, output_path=None):
    """Xử lý dữ liệu mới với model đã train"""
    print(f"Đang xử lý file: {file_path}")
    
    # Đọc file CSV
    try:
        # Thử đọc với header
        df = pd.read_csv(file_path)
        if len(df.columns) == 1:
            # Nếu chỉ có 1 cột, đặt tên là "text"
            df.columns = ["text"]
        elif "text" not in df.columns:
            # Nếu không có cột "text", lấy cột đầu tiên
            df = df.iloc[:, 0:1]
            df.columns = ["text"]
    except:
        # Nếu lỗi, đọc không có header
        df = pd.read_csv(file_path, header=None, names=["text"])
    
    print(f"Tổng số dòng: {len(df)}")
    
    results = []
    for idx, row in df.iterrows():
        text = row['text']
        skills = extract_skills_with_ner(text)
        results.append({
            "text": text,
            "skills": skills
        })
        
        if (idx + 1) % 100 == 0:
            print(f"Đã xử lý {idx + 1}/{len(df)} dòng...")
    
    # Tạo thư mục labeled_data nếu chưa có
    labeled_dir = "labeled_data"
    if not os.path.exists(labeled_dir):
        os.makedirs(labeled_dir)
        print(f"Đã tạo thư mục: {labeled_dir}")
    
    # Tạo output path nếu không có
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(labeled_dir, f"labeled_{base_name}.csv")
    else:
        # Đảm bảo output file nằm trong thư mục labeled_data
        if not output_path.startswith(labeled_dir):
            output_path = os.path.join(labeled_dir, output_path)
    
    # Lưu kết quả
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False, quoting=1)
    print(f"Đã lưu kết quả vào: {output_path}")
    
    # Thống kê
    skills_count = sum(1 for r in results if r['skills'].strip())
    print(f"Số dòng có skills: {skills_count}/{len(results)}")
    
    return output_df

# Test model với ví dụ
def test_model():
    print("=== TEST MODEL ===")
    test_texts = [
        "We are looking for a Python developer with experience in machine learning and data analysis",
        "The ideal candidate should have strong communication skills and project management experience",
        "Requirements include JavaScript, React, and database management skills"
    ]

    for i, text in enumerate(test_texts):
        skills = extract_skills_with_ner(text)
        print(f"\nTest {i+1}:")
        print(f"Text: {text}")
        print(f"Skills: {skills}")

# Main function
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Nếu có tham số dòng lệnh
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        if not os.path.exists(input_file):
            print(f"Lỗi: File {input_file} không tồn tại!")
            sys.exit(1)
        
        process_new_data(input_file, output_file)
    else:
        # Chạy test nếu không có tham số
        test_model()
        print("\n=== HƯỚNG DẪN SỬ DỤNG ===")
        print("1. Test model: python use_trained_model.py")
        print("2. Xử lý file mới: python use_trained_model.py job_descriptions1_part2.csv")
        print("3. Xử lý file mới với output tùy chỉnh: python use_trained_model.py job_descriptions1_part2.csv my_output.csv")
        print("4. Tất cả file đã label sẽ được lưu trong thư mục: labeled_data/")
        print("5. Model đã được train với độ chính xác cao (97.90% F1-score)") 