import pandas as pd
import string
import joblib
import nltk
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Enable progress tracking
tqdm.pandas()

print("📦 Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
print("✅ NLTK resources downloaded.\n")

# Load datasets
print("📁 Loading datasets...")
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")
indian_df = pd.read_csv("indian_data.csv")
print("✅ Datasets loaded successfully.\n")

# Label datasets
fake_df["label"] = 0
true_df["label"] = 1
indian_df = indian_df.rename(columns={'label': 'label', 'text': 'text'})
indian_df['label'] = indian_df['label'].replace({'FAKE': 0, 'REAL': 1})
print("✅ Labels assigned correctly.\n")

# Combine all datasets
df = pd.concat([
    fake_df[['text', 'label']],
    true_df[['text', 'label']],
    indian_df[['text', 'label']]
], axis=0).sample(frac=1).reset_index(drop=True)
print("✅ All datasets combined and shuffled.\n")

# Preprocessing
print("🧼 Starting preprocessing...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(lemmatized)

df['text'] = df['text'].progress_apply(preprocess)
print("✅ Text preprocessing completed.\n")

# Split data
print("✂️ Splitting dataset into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
print("✅ Data split into training and test sets.\n")

# Vectorize text
print("📊 Vectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("✅ Text vectorized.\n")

# Initialize models
print("🤖 Training models...")
log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

pa_model = PassiveAggressiveClassifier(max_iter=1500, C=0.05, random_state=42)

log_model.fit(X_train_vec, y_train)
print("✅ Logistic Regression model trained.")

rf_model.fit(X_train_vec, y_train)
print("✅ Random Forest model trained.")

pa_model.fit(X_train_vec, y_train)
print("✅ Passive Aggressive Classifier trained.\n")

# Evaluate Passive Aggressive model
y_pred = pa_model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Fake", "Real"])
print(f"📈 Passive Aggressive Classifier Accuracy: {accuracy * 100:.2f}%")
print("📋 Classification Report:\n", report)

# Save models and vectorizer
print("💾 Saving all models and vectorizer...")
os.makedirs("model", exist_ok=True)
joblib.dump(log_model, "log_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")

joblib.dump(pa_model, "pa_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ All models and vectorizer saved successfully.\n")
