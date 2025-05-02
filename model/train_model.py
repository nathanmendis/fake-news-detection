import pandas as pd
import string
import joblib
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

# Force fresh downloads of required resources
nltk.download('punkt_tab', force=True)
nltk.download('stopwords', force=True)
nltk.download('wordnet', force=True)

# Load the initial datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels to the initial datasets
fake_df["label"] = 0  # Fake
true_df["label"] = 1  # Real

# Combine the initial datasets
df = pd.concat([fake_df[['text', 'label']], true_df[['text', 'label']]], axis=0).sample(frac=1).reset_index(drop=True)

# Load the new dataset (replace with your actual filename)
new_df = pd.read_csv("indian_data.csv")

# Rename columns of the new DataFrame to match the existing one
new_df = new_df.rename(columns={'label': 'label', 'text': 'text'})

# Convert labels in the new DataFrame to 0 and 1
new_df['label'] = new_df['label'].replace({'FAKE': 0, 'REAL': 1})

# Concatenate the new DataFrame with the existing DataFrame
df = pd.concat([df, new_df[['text', 'label']]], axis=0).sample(frac=1).reset_index(drop=True)

# Preprocessing
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

df['text'] = df['text'].apply(preprocess)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize with potentially better configuration
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Try a slightly different PassiveAggressiveClassifier configuration
model = PassiveAggressiveClassifier(max_iter=1500, C=0.05, random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Fake", "Real"])

print(f"\nUpdated Model Accuracy: {accuracy * 100:.2f}%\n")
print("Updated Classification Report:\n", report)

# Save the updated model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nUpdated model and vectorizer saved successfully!")