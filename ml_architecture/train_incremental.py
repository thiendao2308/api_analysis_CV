import spacy
from spacy.training.example import Example
from spacy.tokens import DocBin
from spacy.util import minibatch
from pathlib import Path

# Đường dẫn model và dữ liệu
MODEL_PATH = "ml_architecture/spacy_models/model-best"
SHARD_PATH = "ml_architecture/data/processed/train_small/train_shard_12.spacy"

# Load model đã train trước đó
nlp = spacy.load(MODEL_PATH)

# Load dữ liệu mới
print(f"Loading data from {SHARD_PATH} ...")
doc_bin = DocBin().from_disk(SHARD_PATH)
docs = list(doc_bin.get_docs(nlp.vocab))
examples = [Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}) for doc in docs]

# Fine-tune tiếp tục với minibatch
optimizer = nlp.resume_training()
BATCH_SIZE = 8
for i in range(5):  # Số epoch, có thể tăng nếu muốn
    losses = {}
    batches = minibatch(examples, size=BATCH_SIZE)
    for batch in batches:
        nlp.update(batch, sgd=optimizer, losses=losses)
    print(f"Epoch {i+1}, Losses: {losses}")

# Lưu lại model đã fine-tune
nlp.to_disk(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}") 