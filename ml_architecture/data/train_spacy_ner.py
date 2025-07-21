import spacy
import json
from spacy.tokens import DocBin
from spacy.util import filter_spans
import random

# Load training data
with open('ner_train_data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Load test data
with open('ner_test_data.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"Training data: {len(train_data)} examples")
print(f"Test data: {len(test_data)} examples")

# Tạo spaCy training format
def create_spacy_training_data(data):
    """Chuyển đổi dữ liệu thành format spaCy training"""
    training_data = []
    
    for example in data:
        text = example['text']
        entities = example['entities']
        
        # Tạo Doc object
        doc = nlp.make_doc(text)
        
        # Thêm entities
        ents = []
        for start, end, label in entities:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        
        # Filter overlapping spans
        ents = filter_spans(ents)
        doc.ents = ents
        
        training_data.append(doc)
    
    return training_data

# Tạo model mới
nlp = spacy.blank("en")

# Thêm NER component
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Thêm label
ner.add_label("SKILL")

# Chuẩn bị training data
train_docs = create_spacy_training_data(train_data)
test_docs = create_spacy_training_data(test_data)

# Lưu training data
train_docbin = DocBin(docs=train_docs)
train_docbin.to_disk("./train.spacy")

# Lưu test data
test_docbin = DocBin(docs=test_docs)
test_docbin.to_disk("./test.spacy")

print("Đã tạo training data thành công!")

# Tạo config file
config_content = """
[paths]
train = "train.spacy"
dev = "test.spacy"

[system]
gpu_allocator = "pytorch"

[nlp]
lang = "en"
pipeline = ["ner"]
batch_size = 1000

[components]

[components.ner]
factory = "ner"

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 128
maxout_pieces = 3
use_upper = true
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2Vec.v2"

[components.ner.model.tok2vec.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 128
rows = [5000, 2000, 1000, 1000]
attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
include_static_vectors = false

[components.ner.model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 128
window_size = 1
maxout_pieces = 3
depth = 4

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 0

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"

[training.optimizer]
@optimizers = "Adam.v1"

[training.batcher]
@batchers = "spacy.batch_by_words.v2"
discard_oversize = true
size = 2000
tolerance = 0.2

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 250
total_steps = 20000
initial_rate = 0.025
final_rate = 0.001

[training.score_weights]
ner_f = 1.0
"""

with open('config.cfg', 'w') as f:
    f.write(config_content)

print("Đã tạo config file!")

# Training command (chạy trong terminal)
print("\n=== COMMANDS ĐỂ TRAIN MODEL ===")
print("1. Tạo project:")
print("python -m spacy project create . --name skill-ner --force")

print("\n2. Copy files:")
print("cp train.spacy test.spacy config.cfg .spacy/projects/skill-ner/")

print("\n3. Train model:")
print("python -m spacy project run train .spacy/projects/skill-ner/")

print("\n4. Test model:")
print("python -m spacy project run evaluate .spacy/projects/skill-ner/") 