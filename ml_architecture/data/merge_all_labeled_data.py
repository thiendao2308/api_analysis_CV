import pandas as pd
import os

def merge_all_labeled_data():
    """Gộp tất cả file đã label thành một dataset lớn"""
    labeled_dir = "labeled_data"
    
    if not os.path.exists(labeled_dir):
        print(f"Thư mục {labeled_dir} không tồn tại!")
        return
    
    # Lấy tất cả file CSV trong thư mục
    csv_files = [f for f in os.listdir(labeled_dir) if f.endswith('.csv')]
    csv_files.sort()  # Sắp xếp theo tên
    
    print(f"Đang gộp {len(csv_files)} file...")
    
    all_data = []
    total_rows = 0
    
    for i, file in enumerate(csv_files):
        file_path = os.path.join(labeled_dir, file)
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            total_rows += len(df)
            print(f"Đã thêm {file}: {len(df)} dòng")
        except Exception as e:
            print(f"Lỗi khi đọc {file}: {e}")
    
    if all_data:
        # Gộp tất cả DataFrame
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # Lưu dataset gộp
        output_file = "merged_labeled_dataset.csv"
        merged_df.to_csv(output_file, index=False, quoting=1)
        
        print(f"\n=== KẾT QUẢ GỘP ===")
        print(f"Tổng số dòng: {len(merged_df):,}")
        print(f"Số dòng có skills: {merged_df['skills'].notna().sum():,}")
        print(f"Tỷ lệ có skills: {merged_df['skills'].notna().sum()/len(merged_df)*100:.1f}%")
        print(f"File đã lưu: {output_file}")
        
        # Thống kê skills
        all_skills = []
        for skills in merged_df['skills'].dropna():
            skills_list = [s.strip() for s in skills.split(',') if s.strip()]
            all_skills.extend(skills_list)
        
        print(f"Số skills unique: {len(set(all_skills))}")
        print(f"Tổng số skills: {len(all_skills):,}")
        
        return merged_df
    else:
        print("Không có file nào để gộp!")
        return None

if __name__ == "__main__":
    merge_all_labeled_data() 