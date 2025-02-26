import streamlit as st
import pickle
import numpy as np
import re
import pandas as pd
import nltk
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Set modern Streamlit page config
st.set_page_config(
    page_title="📧 Spam Detection",
    layout="centered",
    page_icon="📩",
    initial_sidebar_state="collapsed"
)

# Vintage-inspired UI styling
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
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
import traceback
@st.cache_resource
def load_word2vec_model():
    try:
        with open("word2vec_model_custom.model", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load models with caching for performance
@st.cache_resource
def load_models():
    try:
        with open('scaler.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)
        with open('lgbm_final_model.pkl', 'rb') as f:
            final_model = pickle.load(f)
        return loaded_scaler, final_model
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found! Error: {str(e)}")
        return None, None

# Word2Vec Feature Extraction
def avg_word2vec(doc, my_model):
    vectors = [my_model.wv[word] for word in doc if word in my_model.wv.index_to_key]
    return np.mean(vectors, axis=0) if vectors else np.zeros(my_model.vector_size)

# Prediction Function
def predict(msg, loaded_scaler, final_model, my_model):
    corpus = []
    msg = msg.lower()
    msg = re.sub('[^a-zA-Z]', ' ', msg)
    msg = msg.split()
    msg = [lemmatizer.lemmatize(word) for word in msg if word not in stop_words]
    msg = ' '.join(msg)
    corpus.append(msg)

    all_tokens = [word_tokenize(w) for w in corpus]
    X = [avg_word2vec(tokens, my_model) for tokens in all_tokens]
    X = pd.DataFrame(X)

    X_scaled = pd.DataFrame(loaded_scaler.transform(X), columns=X.columns)
    prediction = final_model.predict(X_scaled)
    probability = final_model.predict_proba(X_scaled)[0][1]  # Get probability of being spam
    return prediction[0], probability

# Load models
my_model = load_word2vec_model()
loaded_scaler, final_model = load_models()

# Main UI Layout
st.markdown('<h3 style="text-align: center;">✉️ Compose Your Message</h3>', unsafe_allow_html=True)

# Text area for user input with fixed styling to ensure visibility
user_input = st.text_area("", height=150, placeholder="Type your message here to analyze...", 
                          help="Enter the message you'd like to check for spam classification.")

# Decorative divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Submit button with vintage styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    classify_btn = st.button("Analyze Message")

# Handle Classification
if classify_btn:
    if not user_input:
        st.warning("⚠️ Please enter a message to analyze!")
    elif my_model is None or loaded_scaler is None or final_model is None:
        st.error("⚠️ Required models failed to load. Classification cannot proceed.")
    else:
        with st.spinner("Analyzing your message..."):
            prediction_value, probability = predict(user_input, loaded_scaler, final_model, my_model)
            
            # Display result in a vintage-styled panel
            st.markdown('<div class="spam-result">', unsafe_allow_html=True)
            
            if prediction_value == 1:
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

# Decorative divider
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