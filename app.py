import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

@st.cache_data
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return ' '.join([word for word in text.split() if word not in stop_words])

@st.cache_resource
def load_models():
    with open('movie_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_models()

st.set_page_config(page_title="Movie Genre Classifier", page_icon="🍿", layout="wide")
st.title("🍿 AI Movie Genre Classifier")

col1, col2 = st.columns([1.5, 1.2])

with col1:
    st.subheader("1. Input Movie Plot")
    user_input = st.text_area("Enter your plot summary here:", height=200, placeholder="Type a movie plot...")
    predict_btn = st.button("Predict Genre 🚀", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Analysis & Predictions")
    if predict_btn and user_input.strip():
        with st.spinner("Analyzing text patterns..."):
            cleaned = clean_text(user_input)
            vec = tfidf.transform([cleaned])
            pred = model.predict(vec)[0].strip().upper()
            probs = model.predict_proba(vec)[0]
            
            top_3_idx = probs.argsort()[-3:][::-1]
            
            st.success(f"### 🏆 Primary Genre:\n# **{pred}**")
            st.markdown("#### Confidence Breakdown:")
            for i in top_3_idx:
                g = model.classes_[i].strip().upper()
                p = probs[i] * 100
                st.write(f"**{g}** ({p:.1f}%)")
                st.progress(int(p))
