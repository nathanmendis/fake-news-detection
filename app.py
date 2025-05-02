import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
import PyPDF2
import io
from sklearn.preprocessing import MinMaxScaler

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

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

# Streamlit app
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")

option = st.radio("Choose input type:", ("Text Input", "Upload PDF"))

# Initialize a scaler outside the prediction loop
scaler = MinMaxScaler(feature_range=(0, 100))

if option == "Text Input":
    user_input = st.text_area("Enter News Text Below:", height=200)
    if st.button("Detect"):
        if user_input.strip() == "":
            st.warning("Please enter some news text.")
        else:
            cleaned = preprocess(user_input)
            vectorized = vectorizer.transform([cleaned])
            decision_value = model.decision_function(vectorized)[0]
            pred = model.predict(vectorized)[0]
            label = "REAL" if pred == 1 else "FAKE"

            # Scale the decision value to a percentage-like range
            scaled_value = scaler.fit_transform([[decision_value]])[0][0]

            if label == "REAL":
                st.success(f"This news appears to be **{label}** ")
            else:
                st.warning(f"This news is likely **{label}** ")

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
                    decision_value = model.decision_function(vectorized)[0]
                    pred = model.predict(vectorized)[0]
                    label = "REAL" if pred == 1 else "FAKE"

                    # Scale the decision value to a percentage-like range
                    scaled_value = scaler.fit_transform([[decision_value]])[0][0]

                    if label == "REAL":
                        st.success(f"This news appears to be **{label}** ")
                    else:
                        st.warning(f"This news is likely **{label}** ")
            except Exception as e:
                st.error(f"Failed to read PDF: {e}")