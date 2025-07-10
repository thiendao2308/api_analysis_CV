import spacy

# Đường dẫn tới model đã train
MODEL_PATH = "ml_architecture/spacy_models/model-best"

# Một ví dụ JD bằng tiếng Anh để kiểm tra
examples = [
    (
        "UI design principles and best practices Graphic design tools (e.g., Adobe Photoshop, Illustrator) Typography and color theory Visual design and layout Responsive design "
        
    )
]

def main():
    # Load model đã train
    nlp = spacy.load(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}\n")
    print("Các entity label mà model đã học được:")
    print(nlp.pipe_labels['ner'])
    print("-" * 50)
    for text in examples:
        doc = nlp(text)
        print(f"Văn bản: {text}")
        if doc.ents:
            for ent in doc.ents:
                print(f"  - Entity: '{ent.text}' | Label: {ent.label_}")
        else:
            print("  - Không nhận diện được entity nào.")
        print("-" * 50)

if __name__ == "__main__":
    main() 