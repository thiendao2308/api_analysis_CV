import typer
from pathlib import Path
import spacy
from spacy.tokens import DocBin, Doc
from tqdm import tqdm
import pandas as pd
import warnings
import re
import os
import logging
import sys
import shutil
import random
import csv

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# Define the entity patterns for programmatic labeling
# This list is more comprehensive than the previous one
ENTITY_PATTERNS = {
    "SKILL": [
        # --- IT, Lập trình, Công nghệ ---
        r"Python", r"Java", r"C\+\+", r"C#", r"JavaScript", r"TypeScript", r"PHP", r"HTML", r"CSS",
        r"React", r"Vue", r"Angular", r"Node\.js", r"Django", r"Flask", r"Spring Boot", r".NET", r"Ruby", r"Swift", r"Kotlin",
        r"SQL", r"MySQL", r"PostgreSQL", r"MongoDB", r"Redis", r"Oracle", r"SQLite",
        r"Git", r"Github", r"Linux", r"Windows", r"MacOS", r"Jenkins", r"Docker", r"Kubernetes", r"CI/CD", r"Cloud", r"AWS", r"Azure", r"GCP",
        r"Figma", r"Photoshop", r"Illustrator", r"Canva", r"AutoCAD", r"SolidWorks", r"SketchUp",
        r"API", r"REST", r"GraphQL", r"Microservices", r"Agile", r"Scrum", r"JIRA", r"Trello", r"Asana",
        # --- Kinh doanh, Marketing, Kế toán, Tài chính ---
        r"Excel", r"Word", r"PowerPoint", r"Microsoft Office", r"Google Sheets", r"Google Docs",
        r"CRM", r"Salesforce", r"QuickBooks", r"SAP", r"ERP", r"Accounting", r"Bookkeeping",
        r"Digital Marketing", r"SEO", r"SEM", r"Content Marketing", r"Social Media", r"Facebook Ads", r"Google Ads",
        r"Market Research", r"Business Analysis", r"Financial Analysis", r"Data Analysis", r"Project Management",
        # --- Sản xuất, Kỹ thuật, Xây dựng ---
        r"Vận hành máy [^,\.]+", r"Quản lý sản xuất", r"Quản lý chất lượng", r"ISO 9001", r"Six Sigma", r"Lean Manufacturing",
        r"PLC", r"SCADA", r"Điện công nghiệp", r"Điện tử", r"Cơ khí", r"Gia công CNC", r"Thiết kế CAD", r"Đọc bản vẽ kỹ thuật",
        # --- Giáo dục, Đào tạo ---
        r"Giảng dạy", r"Lesson Planning", r"Curriculum Development", r"Classroom Management", r"Teaching English", r"TESOL", r"IELTS", r"TOEIC",
        # --- Y tế, Dược, Chăm sóc sức khỏe ---
        r"Chăm sóc bệnh nhân", r"Điều dưỡng", r"Y tá", r"Bác sĩ", r"Pharmacy", r"Clinical Research", r"Medical Coding", r"First Aid", r"CPR",
        # --- Dịch vụ, Nhà hàng, Khách sạn ---
        r"Phục vụ", r"Barista", r"Bếp trưởng", r"Quản lý nhà hàng", r"Quản lý khách sạn", r"Housekeeping", r"Front Desk", r"Customer Service",
        # --- Logistics, Xuất nhập khẩu ---
        r"Logistics", r"Supply Chain", r"Xuất nhập khẩu", r"Customs Clearance", r"Shipping", r"Inventory Management",
        # --- Kỹ năng chung, mềm, đa ngành ---
        r"thành thạo [^,\.]+", r"biết sử dụng [^,\.]+", r"sử dụng [^,\.]+", r"kinh nghiệm với [^,\.]+",
        r"có kinh nghiệm về [^,\.]+", r"am hiểu [^,\.]+", r"đã từng làm việc với [^,\.]+", r"sử dụng thành thạo [^,\.]+",
        r"nắm vững [^,\.]+", r"ưu tiên biết [^,\.]+", r"có khả năng sử dụng [^,\.]+", r"có kiến thức về [^,\.]+",
        r"từng làm việc với [^,\.]+", r"có hiểu biết về [^,\.]+", r"giỏi [^,\.]+", r"tốt về [^,\.]+",
        # --- English (generic skill patterns) ---
        r"proficient in [^,\.]+", r"experience with [^,\.]+", r"using [^,\.]+", r"skilled in [^,\.]+",
        r"familiar with [^,\.]+", r"knowledge of [^,\.]+", r"hands-on experience with [^,\.]+", r"expert in [^,\.]+",
        r"well-versed in [^,\.]+", r"capable of using [^,\.]+", r"good at [^,\.]+", r"strong in [^,\.]+", r"background in [^,\.]+",
        # --- English (industry-specific) ---
        r"accounting", r"bookkeeping", r"financial modeling", r"market research", r"customer service", r"project management",
        r"data analysis", r"business analysis", r"sales", r"marketing", r"teaching", r"lesson planning", r"classroom management",
        r"clinical research", r"medical coding", r"first aid", r"pharmacy", r"nursing", r"patient care", r"inventory management",
        r"supply chain", r"shipping", r"customs clearance", r"restaurant management", r"hotel management", r"housekeeping", r"barista"
    ],
    "EDUCATION": [
        # --- Tiếng Việt ---
        r"tốt nghiệp (cao đẳng|đại học|THPT|trung cấp|sau đại học|trường nghề|trường quốc tế|trường chuyên)",
        r"cử nhân", r"thạc sĩ", r"tiến sĩ", r"bác sĩ chuyên khoa", r"kỹ sư", r"dược sĩ", r"giáo viên", r"giảng viên",
        r"bằng (Ielts|IELTS|TOEIC|MOS|Tin học|Anh văn|Anh ngữ|Ngoại ngữ|CPA|CFA|ACCA|Chứng chỉ hành nghề|Chứng chỉ sư phạm|Chứng chỉ kế toán|Chứng chỉ quản lý|Chứng chỉ y tế)",
        r"đạt (Ielts|IELTS|TOEIC) [0-9\.]+", r"bằng [^,\.]+", r"chứng chỉ [^,\.]+",
        r"có bằng [^,\.]+", r"có chứng chỉ [^,\.]+", r"được cấp bằng [^,\.]+",
        # --- English ---
        r"bachelor'?s degree", r"master'?s degree", r"phd", r"doctorate", r"associate'?s degree", r"diploma in [^,\.]+",
        r"graduated from [^,\.]+", r"IELTS [0-9\.]+", r"TOEIC [0-9\.]+", r"certificate in [^,\.]+",
        r"CPA", r"CFA", r"ACCA", r"teaching certificate", r"medical license", r"engineering degree", r"pharmacy degree"
    ],
    "POSITION": [
        # --- IT, Công nghệ ---
        r"Front[- ]?end Developer", r"Backend Developer", r"Fullstack Developer", r"UI Developer", r"UX Designer", r"Tester", r"QA Engineer", r"DevOps Engineer", r"Data Scientist", r"Data Analyst", r"Business Analyst", r"Project Manager", r"Product Manager", r"System Administrator",
        # --- Kinh doanh, Marketing, Tài chính ---
        r"Sales Executive", r"Sales Manager", r"Account Manager", r"Marketing Manager", r"Digital Marketing Specialist", r"Content Creator", r"SEO Specialist", r"Financial Analyst", r"Accountant", r"Chief Accountant", r"Auditor", r"Business Development Manager",
        # --- Sản xuất, Kỹ thuật, Xây dựng ---
        r"Kỹ sư sản xuất", r"Kỹ sư cơ khí", r"Kỹ sư điện", r"Kỹ sư xây dựng", r"Quản đốc", r"Trưởng ca", r"Giám sát sản xuất", r"Giám sát công trình", r"Chỉ huy trưởng công trình",
        # --- Giáo dục, Y tế, Dịch vụ ---
        r"Giáo viên", r"Giảng viên", r"Hiệu trưởng", r"Trợ giảng", r"Bác sĩ", r"Y tá", r"Điều dưỡng", r"Dược sĩ", r"Quản lý nhà hàng", r"Quản lý khách sạn", r"Bếp trưởng", r"Barista", r"Nhân viên phục vụ", r"Lễ tân", r"Chăm sóc khách hàng",
        # --- Logistics, Xuất nhập khẩu ---
        r"Logistics Manager", r"Supply Chain Manager", r"Nhân viên xuất nhập khẩu", r"Customs Officer", r"Warehouse Manager",
        # --- English ---
        r"hiring for [^,\.]+", r"looking for [^,\.]+ position", r"position: [^,\.]+", r"apply for [^,\.]+", r"seeking a [^,\.]+",
        r"teacher", r"lecturer", r"principal", r"assistant teacher", r"doctor", r"nurse", r"pharmacist", r"restaurant manager", r"hotel manager", r"chef", r"waiter", r"receptionist", r"customer service", r"logistics manager", r"supply chain manager", r"import export staff", r"customs officer", r"warehouse manager"
    ],
    "EXPERIENCE": [
        # --- Tiếng Việt ---
        r"(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười) năm kinh nghiệm", r"kinh nghiệm trong lĩnh vực [^,\.]+", r"có kinh nghiệm [^,\.]+", r"tối thiểu \d+ năm kinh nghiệm", r"ít nhất \d+ năm kinh nghiệm", r"đã làm việc tại [^,\.]+", r"từng đảm nhiệm vị trí [^,\.]+", r"có thời gian làm việc tại [^,\.]+", r"đã từng làm việc ở [^,\.]+", r"kinh nghiệm thực tế tại [^,\.]+", r"đã từng tham gia dự án [^,\.]+",
        # --- English ---
        r"\d+ years of experience", r"\d+ year of experience", r"experience in [^,\.]+", r"at least \d+ years of experience", r"previous experience as [^,\.]+", r"worked as [^,\.]+", r"hands-on experience in [^,\.]+", r"practical experience in [^,\.]+", r"background in [^,\.]+", r"track record in [^,\.]+"
    ],
    "SOFT_SKILL": [
        # --- Tiếng Việt ---
        r"kỹ năng giao tiếp", r"kỹ năng làm việc nhóm", r"kỹ năng thuyết trình", r"kỹ năng quản lý thời gian", r"kỹ năng giải quyết vấn đề", r"kỹ năng phản biện", r"khả năng giao tiếp", r"khả năng làm việc nhóm", r"khả năng thích nghi", r"khả năng lãnh đạo", r"tư duy phản biện", r"tư duy logic", r"tư duy sáng tạo", r"kỹ năng đàm phán", r"kỹ năng chăm sóc khách hàng", r"kỹ năng tổ chức công việc", r"kỹ năng quản lý stress", r"kỹ năng học hỏi", r"kỹ năng tự học", r"kỹ năng lắng nghe", r"kỹ năng xây dựng mối quan hệ", r"kỹ năng huấn luyện", r"kỹ năng mentoring", r"kỹ năng đào tạo", r"kỹ năng thích nghi với môi trường mới",
        # --- English ---
        r"communication skills", r"teamwork skills", r"presentation skills", r"time management skills", r"problem[- ]solving skills", r"critical thinking skills", r"adaptability", r"leadership skills", r"creative thinking", r"negotiation skills", r"customer service skills", r"organizational skills", r"stress management skills", r"learning skills", r"self-learning skills", r"listening skills", r"relationship building skills", r"coaching skills", r"mentoring skills", r"training skills", r"adaptability to new environments"
    ]
}

# Chỉ giữ lại pattern SKILL
ENTITY_PATTERNS = {
    "SKILL": ENTITY_PATTERNS["SKILL"]
}


def get_labelled_spans(doc, patterns):
    """
    Finds all occurrences of the patterns in the text of a Doc object and creates spaCy Spans.
    This function operates directly on the Doc object to ensure vocabulary consistency.
    """
    spans = []
    for label, pattern_list in patterns.items():
        for pattern in pattern_list:
            try:
                # Use re.finditer to find all non-overlapping matches in the doc's text
                for match in re.finditer(pattern, doc.text, re.IGNORECASE):
                    start, end = match.span()
                    # Create a span from the character indices.
                    # The alignment_mode="contract" helps handle cases where matches cross token boundaries.
                    span = doc.char_span(start, end, label=label, alignment_mode="contract")
                    if span is not None:
                        # Only add the span if it's valid
                        spans.append(span)
            except re.error:
                # Silently ignore regex errors from malformed patterns
                pass

    # Use spaCy's utility to filter out overlapping spans, preferring longer ones.
    return spacy.util.filter_spans(spans)

def stream_csv_data(file_path: Path, chunk_size: int):
    """
    Streams data from a CSV file, handling files with or without headers.
    Yields chunks of DataFrame with a 'text' column.
    """
    if not file_path.exists():
        logging.warning(f"Source file not found: {file_path}. Skipping.")
        return

    # Nếu file có header 'text', đọc bình thường
    try:
        df = pd.read_csv(file_path, nrows=1)
        if 'text' in df.columns:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                yield chunk[['text']]
            return
    except Exception:
        pass

    # Nếu không có header, đọc từng dòng và gán vào cột 'text'
    try:
        for chunk in pd.read_csv(file_path, header=None, names=['text'], chunksize=chunk_size):
            yield chunk
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")

def main(
    output_dir: Path = typer.Option(..., "-o", "--output-dir", help="Directory to save the .spacy shard files."),
    batch_size: int = typer.Option(1000, "-b", "--batch-size", help="Size of batches for nlp.pipe."),
    shard_size: int = typer.Option(50000, "-s", "--shard-size", help="Number of documents per .spacy shard file."),
):
    """
    Builds the training dataset for the NER model by:
    1. Streaming data from large CSV files (JDs and Resumes).
    2. Applying programmatic labeling using a predefined set of REGEX patterns.
    3. Saving the processed documents into multiple smaller, manageable .spacy files (shards).
    """
    nlp = spacy.blank("en")
    
    # Pre-register all entity labels with the nlp object's vocabulary.
    # This prevents the "StringStore" error by ensuring all labels are known upfront.
    logging.info("Registering entity labels with the vocabulary.")
    for label in ENTITY_PATTERNS.keys():
        nlp.vocab.strings.add(label)

    # Định nghĩa thư mục tạm để lưu shard trước khi chia
    shard_temp_dir = Path("ml_architecture/data/processed/shard_temp")
    shard_temp_dir.mkdir(parents=True, exist_ok=True)
    dev_dir = Path("ml_architecture/data/processed/dev")
    train_dir = Path("ml_architecture/data/processed/train")
    dev_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    # Define the data sources
    jd_path = Path(__file__).parent / "dataset_JD" / "job_descriptions1_part1.csv"
    data_sources = [jd_path]

    total_docs_processed = 0
    shard_count = 0
    doc_bin = DocBin()
    shard_paths = []

    # Process each source file
    for source_path in data_sources:
        logging.info(f"Starting processing for: {source_path.name}")
        data_stream = stream_csv_data(source_path, chunk_size=batch_size)
        
        for df_chunk in tqdm(data_stream, desc=f"Streaming {source_path.name}"):
            df_chunk.dropna(subset=['text'], inplace=True)
            texts = df_chunk['text'].astype(str)
            
            # Efficiently process texts as a stream
            for doc in nlp.pipe(texts):
                try:
                    # Apply regex patterns to find entities by passing the whole doc
                    spans = get_labelled_spans(doc, ENTITY_PATTERNS)
                    doc.ents = spans
                    doc_bin.add(doc)
                    total_docs_processed += 1

                    # Check if the shard is full
                    if len(doc_bin) >= shard_size:
                        shard_path = shard_temp_dir / f"shard_{shard_count}.spacy"
                        logging.info(f"Saving shard with {len(doc_bin)} docs to {shard_path}")
                        doc_bin.to_disk(shard_path)
                        shard_paths.append(shard_path)
                        shard_count += 1
                        doc_bin = DocBin() # Create a new bin for the next shard

                except Exception as e:
                    logging.error(f"Error processing a document: {e}")
                    # You might want to log the specific text that failed
                    # logging.error(f"Problematic text snippet: {doc.text[:100]}...")

    # Save any remaining documents in the last shard
    if len(doc_bin) > 0:
        shard_path = shard_temp_dir / f"shard_{shard_count}.spacy"
        logging.info(f"Saving final shard with {len(doc_bin)} docs to {shard_path}")
        doc_bin.to_disk(shard_path)
        shard_paths.append(shard_path)

    # Sau khi tạo xong tất cả shard, chia vào train/dev (không ngẫu nhiên, shard đầu cho dev)
    print("\nChia shard vào train/dev...")
    all_shards = sorted(shard_temp_dir.glob("*.spacy"))
    n_total = len(all_shards)
    n_dev = int(n_total * 0.2)
    dev_shards = all_shards[:n_dev]
    train_shards = all_shards[n_dev:]
    for shard in train_shards:
        shutil.move(str(shard), train_dir / shard.name)
    for shard in dev_shards:
        shutil.move(str(shard), dev_dir / shard.name)
    print(f"Đã chia {len(train_shards)} shard vào train, {len(dev_shards)} shard vào dev.")
    # Xóa thư mục tạm nếu muốn
    shutil.rmtree(shard_temp_dir)

    # Thay vì lưu .spacy, ta lưu ra CSV với text và skills
    output_csv = output_dir / "labeled_jd_skill.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    total_skill_count = 0
    rows = []
    for source_path in data_sources:
        logging.info(f"Starting processing for: {source_path.name}")
        data_stream = stream_csv_data(source_path, chunk_size=batch_size)
        for df_chunk in tqdm(data_stream, desc=f"Streaming {source_path.name}"):
            df_chunk.dropna(subset=['text'], inplace=True)
            texts = df_chunk['text'].astype(str)
            for doc in nlp.pipe(texts):
                try:
                    spans = get_labelled_spans(doc, ENTITY_PATTERNS)
                    doc.ents = spans
                    # Lấy danh sách skill
                    skills = [span.text for span in spans if span.label_ == "SKILL"]
                    total_skill_count += len(skills)
                    rows.append({"text": doc.text, "skills": ", ".join(skills)})
                except Exception as e:
                    logging.error(f"Error processing a document: {e}")
    # Ghi ra file CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "skills"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nĐã xuất dữ liệu đã gán nhãn SKILL ra file: {output_csv}")
    print(f"Tổng số SKILL được gán nhãn: {total_skill_count}")
    logging.info(f"Tổng số SKILL được gán nhãn: {total_skill_count}")

    logging.info(f"--- Processing Complete ---")
    logging.info(f"Total documents processed: {total_docs_processed}")
    logging.info(f"Total shards created: {shard_count}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Run the main function with Typer
    typer.run(main) 