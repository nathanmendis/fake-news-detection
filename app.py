import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
import PyPDF2
from sklearn.preprocessing import MinMaxScaler
import numpy as np

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load vectorizer and models
vectorizer = joblib.load("model/vectorizer.pkl")
log_model = joblib.load("model/log_model.pkl")
rf_model = joblib.load("model/rf_model.pkl")
pa_model=joblib.load("model/rf_model.pkl")

scaler = MinMaxScaler(feature_range=(0, 100))

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def display_results(vectorized):
    models = {
        "Logistic Regression": log_model,
        "Random Forest": rf_model,
        "PA MODEL":pa_model,
        
    }
    for name, model in models.items():
        pred = model.predict(vectorized)[0]
        label = "REAL" if pred == 1 else "FAKE"
        if hasattr(model, "decision_function"):
            decision_value = model.decision_function(vectorized)[0]
        else:
            decision_value = model.predict_proba(vectorized)[0][1]
        scaled_value = scaler.fit_transform([[decision_value]])[0][0]

        st.subheader(name)
        if label == "REAL":
            st.success(f"Prediction: **{label}** ")
        else:
            st.warning(f"Prediction: **{label}** ")

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
option = st.radio("Choose input type:", ("Text Input", "Upload PDF"))

if option == "Text Input":
    user_input = st.text_area("Enter News Text Below:", height=200)
    if st.button("Detect"):
        if user_input.strip() == "":
            st.warning("Please enter some news text.")
        else:
            cleaned = preprocess(user_input)
            vectorized = vectorizer.transform([cleaned])
            display_results(vectorized)

elif option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Reading PDF..."):
            try:
                pdf_text = extract_text_from_pdf(uploaded_file)
                st.text_area("Extracted Text", pdf_text, height=300)
                if st.button("Detect"):
                    cleaned = preprocess(pdf_text)
                    vectorized = vectorizer.transform([cleaned])
                    display_results(vectorized)
            except Exception as e:
                st.error(f"Failed to read PDF: {e}")
