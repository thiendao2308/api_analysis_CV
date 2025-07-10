import spacy

# Load model
nlp = spacy.load("ml_architecture/spacy_models/model-best")

# 1. In ra các entity types mà mô hình đã học
print("Các entity types mà mô hình nhận diện:")
print(nlp.get_pipe("ner").labels)

# 2. Thử nhận diện thực thể trên một câu mẫu
text = "Kỹ năng Python, Java và vị trí Data Scientist yêu cầu bằng IELTS 7.5"
doc = nlp(text)
print("\nCác thực thể nhận diện được trong câu mẫu:")
for ent in doc.ents:
    print(ent.text, ent.label_)

# 3. (Nếu có vectors) Kiểm tra vector của một số từ
if nlp.vocab.vectors:
    print("\nVector của từ 'Python':")
    print(nlp("Python")[0].vector)
else:
    print("\nModel không có word vectors.")

# 4. In ra một số từ trong vocab
print("\nMột số từ trong vocab:")
for word in list(nlp.vocab.strings)[:20]:
    print(word) 