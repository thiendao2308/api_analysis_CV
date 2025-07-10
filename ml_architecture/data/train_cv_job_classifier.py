import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Đường dẫn file dữ liệu (đảm bảo script nằm cùng thư mục với all_cv_texts.csv)
DATA_PATH = os.path.join(os.path.dirname(__file__), 'all_cv_texts.csv')

# Đọc dữ liệu
print('Đang đọc dữ liệu...')
df = pd.read_csv(DATA_PATH)

# Loại bỏ các dòng lỗi (nếu có)
df = df[~df['text'].str.startswith('ERROR')]

# Tiền xử lý cơ bản (có thể mở rộng thêm)
X = df['text']
y = df['job']

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vector hóa text
print('Đang vector hóa dữ liệu...')
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train mô hình
print('Đang train mô hình...')
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_vec, y_train)

# Đánh giá
print('Đánh giá mô hình:')
y_pred = clf.predict(X_test_vec)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Lưu mô hình và vectorizer
joblib.dump(clf, 'cv_job_classifier.pkl')
joblib.dump(vectorizer, 'cv_vectorizer.pkl')
print('Đã lưu mô hình vào cv_job_classifier.pkl và vectorizer vào cv_vectorizer.pkl') 