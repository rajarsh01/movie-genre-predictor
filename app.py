import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import altair as alt
from nltk.corpus import stopwords

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Movie Genre Classifier", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR MODERN LOOK ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { border-radius: 20px; font-weight: bold; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# --- SETUP & CACHING ---
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    return set(stopwords.words('english'))

stop_words = setup_nltk()

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

# --- SIDEBAR DASHBOARD ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3171/3171927.png", width=100)
    st.title("System Stats")
    st.markdown("This AI uses Natural Language Processing (NLP) to predict genres.")
    st.divider()
    st.metric(label="Vocabulary Size", value="10,000 phrases")
    st.metric(label="Target Genres", value="27")
    st.metric(label="Algorithm", value="Logistic Regression")
    st.divider()
    st.caption("Developed for Academic Presentation")

# --- MAIN UI HEADER ---
st.title("🎬 AI Movie Genre Predictor")
st.markdown("#### *Type a plot summary below and let the AI analyze the narrative.*")
st.write("---")

# --- RESPONSIVE INPUT SECTION ---
user_input = st.text_area("Enter Plot Summary:", height=150, 
                          placeholder="e.g., A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival...")

# Dynamic word count feedback
word_count = len(user_input.split())
if word_count > 0 and word_count < 15:
    st.warning("⚠️ Try writing a longer plot (at least 15 words) for better accuracy!")

# --- PROCESSING & RESULTS ---
if st.button("Predict Genre 🚀", type="primary", use_container_width=True):
    if not user_input.strip():
        st.error("Please enter a movie plot first!")
    else:
        with st.spinner("Analyzing narrative patterns and vocabulary..."):
            # 1. Process Text
            cleaned_text = clean_text(user_input)
            vectorized_text = tfidf.transform([cleaned_text])
            
            # 2. Get Predictions
            pred = model.predict(vectorized_text)[0].strip().title()
            probs = model.predict_proba(vectorized_text)[0]
            
            # 3. Organize Top 5 Data
            top_5_idx = probs.argsort()[-5:][::-1]
            top_5_genres = [model.classes_[i].strip().title() for i in top_5_idx]
            top_5_probs = [probs[i] * 100 for i in top_5_idx]
            
            # 4. Extract Top AI Keywords (TF-IDF highest scores)
            feature_names = tfidf.get_feature_names_out()
            non_zero_indices = vectorized_text.nonzero()[1]
            scores = vectorized_text.data
            sorted_items = sorted(zip(non_zero_indices, scores), key=lambda x: x[1], reverse=True)
            top_words = [feature_names[i] for i, score in sorted_items[:5]]

            st.toast("Analysis Complete!", icon="✅")

            # --- TABBED RESULTS INTERFACE ---
            st.write("---")
            tab1, tab2, tab3 = st.tabs(["🎯 Final Verdict", "📊 Confidence Chart", "🧠 NLP Insights"])
            
            # TAB 1: Primary Prediction
            with tab1:
                st.markdown(f"<div class='metric-card'><h2 style='text-align: center; color: #1E88E5;'>🏆 Predicted Genre: {pred}</h2></div>", unsafe_allow_html=True)
                st.balloons()
                
            # TAB 2: Interactive Data Visualization
            with tab2:
                st.subheader("Top 5 Likeliest Genres")
                # Create a sleek, interactive horizontal bar chart using Altair
                df_chart = pd.DataFrame({'Genre': top_5_genres, 'Confidence (%)': top_5_probs})
                chart = alt.Chart(df_chart).mark_bar(color='#FF4B4B').encode(
                    x=alt.X('Confidence (%):Q', scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y('Genre:N', sort='-x'),
                    tooltip=['Genre', alt.Tooltip('Confidence (%):Q', format='.1f')]
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

            # TAB 3: Under the Hood (Explainable AI)
            with tab3:
                st.subheader("How the AI made its decision")
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**Most Influential Words/Phrases:**")
                    st.info(", ".join(top_words).title() if top_words else "None detected")
                    st.caption("These specific phrases triggered the AI's highest confidence scores based on its training data.")
                with colB:
                    st.markdown("**Data Preprocessing:**")
                    st.markdown(f"- Original words: **{word_count}**")
                    st.markdown(f"- Words retained after cleaning: **{len(cleaned_text.split())**}")
                
                st.markdown("**What the AI 'saw' (Stopwords removed):**")
                st.text(cleaned_text)
