import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv('all_kindle_review.csv')
    df = data[['reviewText', 'rating']].copy()
    df['rating'] = df['rating'].apply(lambda x: 0 if x < 4 else 1)
    df['reviewText'] = df['reviewText'].apply(preprocess_text)
    return df

@st.cache_data
def train_model(df):
    X = df[['reviewText']]
    y = df['rating']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=16
    )
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_features=50000)
    X_train_tfidf = tfidf.fit_transform(X_train['reviewText'])
    X_test_tfidf = tfidf.transform(X_test['reviewText'])
    X_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf.get_feature_names_out())
    X_test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf.get_feature_names_out())
    model = LGBMClassifier(learning_rate=0.1, n_estimators=300)
    model.fit(X_train_df, y_train)
    y_pred = model.predict(X_test_df)
    acc = accuracy_score(y_test, y_pred)
    return model, tfidf, acc

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Lora:ital@0;1&display=swap');
    
    /* Main Background with subtle texture */
    body {
        background-color: #F5F1E6;
        background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23d1bc9c' fill-opacity='0.1' fill-rule='evenodd'/%3E%3C/svg%3E");
        font-family: 'Lora', serif;
        color: #3A2E28;
    }
    
    /* Main container with parchment effect */
    .main-container {
        max-width: 1000px;
        margin: 20px auto;
        padding: 40px;
        background-color: #F9F5E9;
        border: 1px solid #D4C5A8;
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1), 0 0 30px rgba(0, 0, 0, 0.05);
        position: relative;
    }
    
    /* Ornate header */
    .header {
        text-align: center;
        font-family: 'Playfair Display', serif;
        font-size: 52px;
        font-weight: 900;
        line-height: 1.2;
        color: #3A2E28;
        margin-bottom: 5px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        background-image: linear-gradient(180deg, #5D4037 0%, #8A6552 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
    }
    
    /* Vintage book graphics for header */
    .header::before, .header::after {
        content: "📚";
        font-size: 34px;
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
    }
    
    .header::before {
        left: 20px;
    }
    
    .header::after {
        right: 20px;
    }
    
    /* Elegant subheader */
    .subheader {
        text-align: center;
        font-family: 'Lora', serif;
        font-size: 22px;
        font-style: italic;
        color: #8A6552;
        margin-bottom: 30px;
    }
    
    /* Stylized dividers */
    .divider {
        position: relative;
        height: 30px;
        margin: 40px 0;
        text-align: center;
        overflow: visible;
    }
    
    .divider::before {
        content: "";
        display: block;
        height: 1px;
        width: 100%;
        background: linear-gradient(90deg, transparent, #C8B39C, transparent);
        position: absolute;
        top: 50%;
    }
    
    .divider::after {
        content: "❦";
        background-color: #F9F5E9;
        padding: 0 20px;
        font-size: 24px;
        color: #8A6552;
        position: relative;
        display: inline-block;
    }
    
    /* Vintage buttons */
    .stButton > button {
        font-family: 'Playfair Display', serif;
        background-color: #6D4C41;
        background-image: linear-gradient(to bottom, #8D6E63, #5D4037);
        color: #F9F5E9;
        border: none;
        border-radius: 5px;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        letter-spacing: 1px;
        box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.2), 0px 3px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        background-image: linear-gradient(to bottom, #795548, #4E342E);
        transform: translateY(-2px);
        box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Stylized text area */
    .stTextArea > div > div > textarea {
        font-family: 'Lora', serif;
        font-size: 18px;
        line-height: 1.6;
        padding: 15px;
        border: 1px solid #D4C5A8;
        border-radius: 5px;
        background-color: #FAF8F1;
        color: #3A2E28;
        box-shadow: inset 0px 1px 5px rgba(0, 0, 0, 0.05);
        transition: all 0.3s;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #8A6552;
        box-shadow: inset 0px 1px 5px rgba(0, 0, 0, 0.1), 0px 0px 8px rgba(138, 101, 82, 0.3);
    }
    
    /* Section headers */
    h3 {
        font-family: 'Playfair Display', serif;
        color: #5D4037;
        font-size: 28px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 20px;
        padding-bottom: 8px;
        border-bottom: 1px solid #D4C5A8;
    }
    
    /* Prediction output styling */
    .prediction-container {
        margin: 25px 0;
        padding: 20px;
        background-color: #FAF3E0;
        border: 1px solid #D4C5A8;
        border-radius: 5px;
        text-align: center;
        box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.05);
    }
    
    .prediction-title {
        font-family: 'Playfair Display', serif;
        font-size: 24px;
        color: #5D4037;
        margin-bottom: 5px;
    }
    
    .prediction-result {
        font-family: 'Playfair Display', serif;
        font-size: 36px;
        font-weight: 700;
    }
    
    .prediction-positive {
        color: #2E7D32;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-negative {
        color: #C62828;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    /* Loading animation */
    .stSpinner > div {
        border-color: #8A6552 !important;
    }
    
    /* DataFrame styling */
    .dataframe {
        font-family: 'Lora', serif;
        border-collapse: collapse;
        width: 100%;
        margin: 25px 0;
        font-size: 16px;
        box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.05);
    }
    
    .dataframe th {
        background-color: #8A6552;
        color: #FAF8F1;
        padding: 12px 15px;
        text-align: left;
        font-weight: 700;
    }
    
    .dataframe td {
        padding: 12px 15px;
        border-bottom: 1px solid #D4C5A8;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #FAF8F1;
    }
    
    .dataframe tr:hover {
        background-color: #F3EBD7;
    }
    
    /* Checkbox styling */
    .stCheckbox > div > div > label {
        font-family: 'Lora', serif;
        color: #5D4037;
        font-size: 18px;
    }
    
    /* Success and error message styling */
    .stSuccess, .stError {
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
        font-family: 'Lora', serif;
    }
    
    .stSuccess {
        background-color: #E8F5E9;
        color: #2E7D32;
        border-left: 4px solid #2E7D32;
    }
    
    .stError {
        background-color: #FFEBEE;
        color: #C62828;
        border-left: 4px solid #C62828;
    }
    
    /* Add vintage book corners to the main container */
    .corner {
        position: absolute;
        width: 30px;
        height: 30px;
        background-color: #8A6552;
    }
    
    .corner-top-left {
        top: 0;
        left: 0;
        border-radius: 0 0 100% 0;
    }
    
    .corner-top-right {
        top: 0;
        right: 0;
        border-radius: 0 0 0 100%;
    }
    
    .corner-bottom-left {
        bottom: 0;
        left: 0;
        border-radius: 0 100% 0 0;
    }
    
    .corner-bottom-right {
        bottom: 0;
        right: 0;
        border-radius: 100% 0 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="corner corner-top-left"></div>', unsafe_allow_html=True)
st.markdown('<div class="corner corner-top-right"></div>', unsafe_allow_html=True)
st.markdown('<div class="corner corner-bottom-left"></div>', unsafe_allow_html=True)
st.markdown('<div class="corner corner-bottom-right"></div>', unsafe_allow_html=True)
st.markdown('<div class="header">Kindle Review Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Unveiling the Sentiment of Timeless Literary Treasures.</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Discover the hidden emotions within Kindle book reviews with our sophisticated algorithm..</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

with st.spinner("🔍 Analyzing literary sentiments from the archives..."):
    df = load_and_preprocess_data()

with st.spinner("📚 Training our literary sentiment predictor..."):
    model, tfidf, accuracy = train_model(df)

user_review = st.text_area("", placeholder="Begin typing your review here...", height=150)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("Analyze Sentiment")

if analyze_button:
    if user_review:
        with st.spinner("📖 Deciphering the sentiments within your words..."):
            processed_review = preprocess_text(user_review)
            review_tfidf = tfidf.transform([processed_review])
            review_df = pd.DataFrame(review_tfidf.toarray(), columns=tfidf.get_feature_names_out())
            prediction = model.predict(review_df)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"
            sentiment_class = "prediction-positive" if prediction == 1 else "prediction-negative"
            
        st.markdown(f"""
        <div class="prediction-container">
            <div class="prediction-title">The Sentiment Analysis Reveals:</div>
            <div class="prediction-result {sentiment_class}">{sentiment}</div>
        </div>
        """, unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("""
            <p style="font-family: 'Lora', serif; font-style: italic; text-align: center; margin: 20px 0;">
                "There is no friend as loyal as a book that brings joy to its reader." — Ernest Hemingway
            </p>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <p style="font-family: 'Lora', serif; font-style: italic; text-align: center; margin: 20px 0;">
                "Even the darkest criticism can illuminate the path to literary excellence." — Anonymous
            </p>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="stError">
            Please grace us with your literary insights before analysis.
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
with st.expander("📚 Explore the Literary Archives"):
    st.markdown("""
    <h3 style="font-family: 'Playfair Display', serif; font-size: 24px; margin-bottom: 15px;">
        Sample of Processed Review Data
    </h3>
    <p style="font-family: 'Lora', serif; font-style: italic; margin-bottom: 15px;">
        Behold a glimpse into our collection of preprocessed Kindle reviews.
    </p>
    """, unsafe_allow_html=True)
    st.dataframe(df.head(10))
    st.markdown("""
    <h3 style="font-family: 'Playfair Display', serif; font-size: 24px; margin: 25px 0 15px 0;">
        Literary Collection Statistics
    </h3>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
        st.metric("Positive Reviews", f"{df['rating'].sum():,}")
    with col2:
        st.metric("Negative Reviews", f"{(len(df) - df['rating'].sum()):,}")
        st.metric("Positive Ratio", f"{df['rating'].mean()*100:.1f}%")
st.markdown("""
<div style="text-align: center; margin-top: 50px; font-family: 'Lora', serif; font-style: italic; font-size: 16px; color: #8A6552;">
    "In the pages of a book, we find a different world. In the pages of a review, we find our reflection."
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)