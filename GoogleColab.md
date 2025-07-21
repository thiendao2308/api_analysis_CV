# Hướng dẫn label kỹ năng từ JD bằng LLM mã nguồn mở trên Google Colab (Tối ưu cho file ~10.000 dòng)

## 1. Cài đặt thư viện cần thiết

```python
!pip install transformers accelerate sentencepiece
```

## 2. Tải model LLM mã nguồn mở (TinyLlama, public, không cần xin quyền)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=False
)
```

## 3. Upload file JD lên Colab

```python
from google.colab import files
uploaded = files.upload()  # Chọn file JD từ máy tính

import pandas as pd
df = pd.read_csv(list(uploaded.keys())[0], header=None, names=["text"])
```

## 4. Định nghĩa hàm prompt và làm sạch kết quả

```python
def extract_skills_from_jd(jd_text):
    prompt = (
        "List all professional skills (as a comma-separated list) mentioned in the following job description. "
        "Do not include any equal sign (=) or quotes at the beginning:\n"
        f"{jd_text}\n"
        "Skills:"
    )
    result = pipe(prompt)[0]['generated_text']
    # Lấy phần sau 'Skills:'
    if "Skills:" in result:
        skills = result.split("Skills:")[-1].strip().split("\n")[0]
    else:
        skills = result.strip()
    return skills

def clean_skills_output(skills):
    # Loại bỏ dấu =, dấu nháy, khoảng trắng thừa ở đầu
    skills = str(skills).lstrip('= ').strip('"\' ')
    return skills
```

## 5. Chạy batch labeling (batch inference) và lưu kết quả

```python
batch_size = 8  # Có thể tăng lên 16 hoặc 32 nếu Colab không báo lỗi RAM
results = []

for i in range(0, len(df), batch_size):
    batch = df["text"][i:i+batch_size]
    prompts = [
        "List all professional skills (as a comma-separated list) mentioned in the following job description. "
        "Do not include any equal sign (=) or quotes at the beginning:\n"
        f"{jd}\nSkills:" for jd in batch
    ]
    outputs = pipe(prompts)
    for jd, out in zip(batch, outputs):
        skills = out['generated_text']
        # Làm sạch kết quả
        skills = clean_skills_output(skills)
        results.append({"text": jd, "skills": skills})
    print(f"Đã xử lý {i+len(batch)}/{len(df)} dòng...")

out_path = "/content/labeled_jd_skill_tinyllama.csv"
pd.DataFrame(results).to_csv(out_path, index=False)
print(f"Đã xuất kết quả ra {out_path}")
```

## 6. Tải file kết quả về máy

```python
from google.colab import files
files.download(out_path)
```

---

**Lưu ý:**

- Nếu file lớn, nên thử trước với 100-1000 dòng để kiểm tra chất lượng.
- Có thể tối ưu prompt hoặc batch nếu cần tốc độ nhanh hơn.
- Nếu reset Colab hoặc bị timeout, hãy chạy lại từ bước cài thư viện và tải model.
- Nếu muốn tiếp tục từ dòng đã dừng, hãy lưu checkpoint tạm thời sau mỗi batch.
