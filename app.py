import streamlit as st
import numpy as np
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Set Streamlit page config
st.set_page_config(
    page_title="📧 Spam Detection",
    layout="centered",
    page_icon="📩",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Libre+Baskerville&display=swap');
        
        /* Main background with vintage paper texture */
        .main {
            background-color: #f5f1e6;
            background-image: url("https://www.transparenttextures.com/patterns/old-paper.png");
            padding: 20px;
        }
        
        /* Elegant typography */
        h1, h2, h3 {
            font-family: 'Playfair Display', serif;
            color: #2c3e50;
        }
        
        p, div {
            font-family: 'Libre Baskerville', serif;
            color: #34495e;
        }
        
        /* Classic styled text area with proper contrast */
        .stTextArea textarea {
            background-color: #fffef8 !important;
            color: #2c3e50 !important;
            border: 2px solid #8b7765 !important;
            border-radius: 5px !important;
            font-family: 'Libre Baskerville', serif !important;
            font-size: 16px !important;
            padding: 15px !important;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1) !important;
        }
        
        /* Vintage button design */
        .stButton > button {
            background-color: #8b7765 !important;
            color: #f5f1e6 !important;
            font-family: 'Playfair Display', serif !important;
            font-size: 18px !important;
            font-weight: bold !important;
            border: none !important;
            border-radius: 5px !important;
            padding: 10px 20px !important;
            width: 100% !important;
            transition: all 0.3s ease !important;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2) !important;
        }
        
        .stButton > button:hover {
            background-color: #6d5d4d !important;
            box-shadow: 1px 1px 3px rgba(0,0,0,0.3) !important;
        }
        
        /* Decorative elements */
        .header-decoration {
            text-align: center;
            margin: 10px auto;
            font-size: 24px;
            color: #8b7765;
        }
        
        /* Result panels with vintage styling */
        .spam-result {
            background-color: #f8f2e9;
            border: 2px solid #8b7765;
            border-radius: 5px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 3px 3px 7px rgba(0,0,0,0.1);
        }
        
        /* Vintage divider */
        .divider {
            border-top: 1px solid #8b7765;
            margin: 25px 0;
            position: relative;
        }
        
        .divider:after {
            content: "✉";
            position: absolute;
            top: -10px;
            left: 50%;
            transform: translateX(-50%);
            background: #f5f1e6;
            padding: 0 15px;
            color: #8b7765;
            font-size: 16px;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-family: 'Playfair Display', serif !important;
            color: #2c3e50 !important;
            background-color: #f8f2e9 !important;
            border: 1px solid #8b7765 !important;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            font-style: italic;
            margin-top: 30px;
            font-size: 14px;
            color: #8b7765;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Decorative header

# Title with vintage styling
st.markdown("<h1 style='text-align: center;'>📩 Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>A classic approach to modern problems</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect if an email or message is spam with elegant precision.</p>", unsafe_allow_html=True)

# Decorative divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
# Initialize NLP components

# Load dataset
data = pd.read_csv('SMSSpamCollection.txt', sep='\t', names=['Prediction', 'Message'], on_bad_lines="skip")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess messages
corpus = []
for i in range(len(data)):
    msg = data['Message'][i].lower()
    msg = re.sub('[^a-zA-Z]', ' ', msg)
    msg = msg.split()
    msg = [lemmatizer.lemmatize(word) for word in msg if word not in stop_words]
    corpus.append(msg)

data['Processed_Message'] = corpus

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

def avg_word2vec(doc):
    vectors = [word2vec_model.wv[word] for word in doc if word in word2vec_model.wv.index_to_key]
    return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec_model.vector_size)

# Feature extraction
X = [avg_word2vec(msg) for msg in corpus]
X = pd.DataFrame(X)

# Encode target variable
ohe = OneHotEncoder(drop='first', sparse_output=False)
y = pd.DataFrame(ohe.fit_transform(data[['Prediction']]), columns=['Prediction'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# Standardization
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Train LGBM model
lgbm_model = lgb.LGBMClassifier()
lgbm_model.fit(X_train, y_train.values.ravel())

# Streamlit UI
st.markdown('<h3 style="text-align: center;">✉️ Compose Your Message</h3>', unsafe_allow_html=True)
st.write(" ")

# User input
user_input = st.text_area("", height=150, placeholder="Type your message here to analyze...", 
                          help="Enter the message you'd like to check for spam classification.")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
def predict(msg):
    msg = msg.lower()
    msg = re.sub('[^a-zA-Z]', ' ', msg)
    msg = word_tokenize(msg)
    msg = [lemmatizer.lemmatize(word) for word in msg if word not in stop_words]
    X_new = pd.DataFrame([avg_word2vec(msg)])
    X_new_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns)
    prediction = lgbm_model.predict(X_new_scaled)
    probability = lgbm_model.predict_proba(X_new_scaled)[0][1]
    return prediction[0], probability

if st.button("Analyze Message"):
    if not user_input:
        st.warning("⚠️ Please enter a message to analyze!")
    if user_input:
        with st.spinner("Analyzing your message..."):
            prediction, probability = predict(user_input)
            st.markdown('<div class="spam-result">', unsafe_allow_html=True)
            if prediction== 1:
                st.markdown(
                    f"""
                    <h2 style='text-align: center; color: #a83232;'>📛 SPAM DETECTED</h2>
                    <p style='text-align: center; font-style: italic;'>
                        With {probability:.1%} certainty, this message appears to be unsolicited correspondence.
                    </p>
                    <p style='text-align: center;'>
                        We advise caution with this communication.
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <h2 style='text-align: center; color: #2e7d32;'>✓ LEGITIMATE CORRESPONDENCE</h2>
                    <p style='text-align: center; font-style: italic;'>
                        With {(1-probability):.1%} certainty, this message appears to be genuine.
                    </p>
                    <p style='text-align: center;'>
                        This communication seems to be proper and legitimate.
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="header-decoration">✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦</div>', unsafe_allow_html=True)
# Information expander with vintage styling
with st.expander("📜 About This Detector"):
    st.markdown("""
    <div style='font-family: "Libre Baskerville", serif;'>
        <p>This spam detector combines time-honored wisdom with modern techniques:</p>
        <ul>
            <li><strong>Word Embeddings:</strong> Messages are transformed into numerical representations using Word2Vec technology.</li>
            <li><strong>Advanced Classification:</strong> A LightGBM classifier examines these representations to determine authenticity.</li>
            <li><strong>Natural Language Processing:</strong> Before analysis, messages undergo proper linguistic preparation.</li>
            <li><strong>Custom Training:</strong> The system has been educated on a collection of both proper and improper correspondences.</li>
        </ul>
        <p style='font-style: italic;'>The pursuit of separating genuine communication from unwanted solicitations is as old as correspondence itself.</p>
    </div>
    """, unsafe_allow_html=True)

# Vintage footer
st.markdown(
    '<div class="footer">Crafted with care and consideration by Atharva, using the finest technologies available.</div>',
    unsafe_allow_html=True
)