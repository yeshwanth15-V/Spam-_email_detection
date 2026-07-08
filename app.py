import streamlit as st
import pickle
import re
import nltk
import numpy as np
import time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load the trained model and vectorizer
@st.cache_resource
def load_models():
    try:
        with open('spam_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure 'spam_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
        return None, None

# Text preprocessing function (must match training preprocessing)
def preprocess_text(text):
    """
    Preprocess the input text using the same steps as training:
    - lowercase
    - remove HTML tags
    - remove URLs
    - remove email addresses
    - remove numbers
    - remove punctuation
    - tokenization
    - stopword removal
    - stemming
    """
    # Initialize stemmer and stopwords
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Tokenization and remove extra whitespace
    tokens = text.split()
    
    # Remove stopwords and apply stemming
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    
    # Join tokens back into a string
    processed_text = ' '.join(tokens)
    
    return processed_text

# Function to get email statistics
def get_email_stats(text):
    char_count = len(text)
    word_count = len(text.split())
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    return char_count, word_count, sentence_count

# Function to get risk level based on spam probability
def get_risk_level(spam_prob):
    if spam_prob <= 0.30:
        return "Safe", "#22C55E"
    elif spam_prob <= 0.60:
        return "Suspicious", "#F59E0B"
    else:
        return "High Risk", "#EF4444"

# Sample emails
SAMPLE_SPAM = """URGENT: Your account has been compromised! 
Click here immediately to verify your identity and secure your funds. 
Failure to do so within 24 hours will result in permanent account suspension. 
You need to update your banking information to avoid fraud."""

SAMPLE_HAM = """Hi Team,
 
I hope this email finds you well. I'm writing to follow up on our meeting last week regarding the Q4 marketing strategy. 
I've attached the updated timeline and budget projections for your review.
 
Please let me know if you have any questions or if you'd like to schedule a brief call to discuss further.
 
Looking forward to your feedback.
 
Best regards,
John Smith
Marketing Director"""

# Callback functions for buttons
def set_spam_sample():
    st.session_state.email_input = SAMPLE_SPAM

def set_ham_sample():
    st.session_state.email_input = SAMPLE_HAM

def clear_input():
    st.session_state.email_input = ""

# Inject custom CSS
def inject_custom_css():
    st.markdown("""
        <style>
        /* Global styles */
        .stApp {
            background: #0F172A;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Remove default Streamlit padding */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #1E293B;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
            background: #3B82F6;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #2563EB;
        }
        
        /* Card styling */
        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4);
            border-color: rgba(59, 130, 246, 0.3);
        }
        
        /* Hero section */
        .hero-section {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            position: relative;
            overflow: hidden;
        }
        
        .hero-section::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at 30% 50%, rgba(59, 130, 246, 0.1) 0%, transparent 50%);
            animation: pulse 4s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 1; }
        }
        
        .hero-title {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            position: relative;
            z-index: 1;
        }
        
        .hero-subtitle {
            font-size: 1.2rem;
            color: #94A3B8;
            margin-top: 0.5rem;
            position: relative;
            z-index: 1;
        }
        
        .hero-description {
            font-size: 1rem;
            color: #64748B;
            max-width: 600px;
            margin: 1rem auto 0;
            position: relative;
            z-index: 1;
        }
        
        /* Button styling */
        .stButton > button {
            width: 100%;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            border: none;
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
            color: white;
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(59, 130, 246, 0.5);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* Clear button */
        .clear-btn > button {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #94A3B8;
            box-shadow: none;
        }
        
        .clear-btn > button:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: none;
        }
        
        /* Text area styling */
        .stTextArea > div > div > textarea {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: #E2E8F0;
            padding: 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            min-height: 250px;
        }
        
        .stTextArea > div > div > textarea:focus {
            border-color: #3B82F6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
            background: rgba(15, 23, 42, 0.8);
        }
        
        .stTextArea > div > div > textarea::placeholder {
            color: #64748B;
        }
        
        /* Result cards */
        .result-card {
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            animation: slideUp 0.5s ease-out;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result-ham {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.05) 100%);
            border: 2px solid rgba(34, 197, 94, 0.3);
        }
        
        .result-spam {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.05) 100%);
            border: 2px solid rgba(239, 68, 68, 0.3);
        }
        
        .result-icon {
            font-size: 4rem;
            margin-bottom: 0.5rem;
        }
        
        .result-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }
        
        .result-title-ham {
            color: #22C55E;
        }
        
        .result-title-spam {
            color: #EF4444;
        }
        
        .result-description {
            font-size: 1rem;
            color: #94A3B8;
        }
        
        /* Metric cards */
        .metric-card {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            border-color: rgba(59, 130, 246, 0.3);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        }
        
        .metric-icon {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #E2E8F0;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #94A3B8;
            margin-top: 0.25rem;
        }
        
        /* Risk badge */
        .risk-badge {
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.9rem;
            letter-spacing: 0.5px;
            animation: pulse 2s ease-in-out infinite;
        }
        
        /* Progress bar styling */
        .custom-progress {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 50px;
            height: 8px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .custom-progress-bar {
            height: 100%;
            border-radius: 50px;
            transition: width 1s ease-in-out;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .sidebar-content {
            padding: 1rem 0;
        }
        
        .sidebar-section {
            margin-bottom: 1.5rem;
        }
        
        .sidebar-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #64748B;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }
        
        .sidebar-item {
            display: flex;
            justify-content: space-between;
            padding: 0.4rem 0;
            color: #94A3B8;
            font-size: 0.9rem;
        }
        
        .sidebar-value {
            color: #E2E8F0;
            font-weight: 500;
        }
        
        .sidebar-workflow {
            padding: 0.5rem 0;
        }
        
        .workflow-step {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.3rem 0;
            color: #94A3B8;
            font-size: 0.85rem;
        }
        
        .workflow-arrow {
            color: #3B82F6;
            font-size: 0.8rem;
        }
        
        .tech-badge {
            display: inline-block;
            background: rgba(59, 130, 246, 0.1);
            color: #60A5FA;
            padding: 0.2rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            margin: 0.2rem 0.2rem;
        }
        
        /* Confidence gauge */
        .gauge-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 1rem 0;
        }
        
        .gauge-circle {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: conic-gradient(#3B82F6 var(--progress, 0%), rgba(255, 255, 255, 0.05) var(--progress, 0%));
            position: relative;
        }
        
        .gauge-circle::before {
            content: '';
            position: absolute;
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: #0F172A;
        }
        
        .gauge-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #E2E8F0;
            position: relative;
            z-index: 1;
        }
        
        .gauge-label {
            font-size: 0.85rem;
            color: #94A3B8;
            position: relative;
            z-index: 1;
        }
        
        /* Expandable section */
        .stExpander {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
        }
        
        .stExpander > div:first-child {
            background: transparent;
            color: #E2E8F0;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0;
            margin-top: 3rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .footer-text {
            color: #64748B;
            font-size: 0.85rem;
        }
        
        .footer-tech {
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 0.5rem;
        }
        
        .footer-tech-item {
            background: rgba(30, 41, 59, 0.5);
            padding: 0.3rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            color: #94A3B8;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 2rem;
            }
            
            .hero-subtitle {
                font-size: 1rem;
            }
            
            .result-title {
                font-size: 1.8rem;
            }
            
            .metric-value {
                font-size: 1.5rem;
            }
            
            .gauge-circle {
                width: 120px;
                height: 120px;
            }
            
            .gauge-circle::before {
                width: 96px;
                height: 96px;
            }
            
            .gauge-value {
                font-size: 2rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Smart Email Spam Detection",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for email input if it doesn't exist
    if "email_input" not in st.session_state:
        st.session_state.email_input = ""
    
    # Inject custom CSS
    inject_custom_css()
    
    # Load models
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Project Info
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📊 Project Information</div>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="sidebar-item">
                <span>Model</span>
                <span class="sidebar-value">Logistic Regression</span>
            </div>
            <div class="sidebar-item">
                <span>Vectorizer</span>
                <span class="sidebar-value">TF-IDF</span>
            </div>
            <div class="sidebar-item">
                <span>Dataset</span>
                <span class="sidebar-value">Spam/Ham</span>
            </div>
            <div class="sidebar-item">
                <span>Accuracy</span>
                <span class="sidebar-value">98.16%</span>
            </div>
            <div class="sidebar-item">
                <span>Developer</span>
                <span class="sidebar-value">Yeshwanth</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Workflow
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">⚡ Workflow</div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="sidebar-workflow">
                <div class="workflow-step">
                    <span>📄 Dataset</span>
                </div>
                <div class="workflow-step">
                    <span class="workflow-arrow">↓</span>
                    <span>🧹 Cleaning</span>
                </div>
                <div class="workflow-step">
                    <span class="workflow-arrow">↓</span>
                    <span>📊 TF-IDF</span>
                </div>
                <div class="workflow-step">
                    <span class="workflow-arrow">↓</span>
                    <span>🤖 Logistic Regression</span>
                </div>
                <div class="workflow-step">
                    <span class="workflow-arrow">↓</span>
                    <span>🎯 Prediction</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Technologies
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🛠️ Technologies</div>', unsafe_allow_html=True)
        st.markdown("""
            <div>
                <span class="tech-badge">Python</span>
                <span class="tech-badge">Streamlit</span>
                <span class="tech-badge">Scikit-Learn</span>
                <span class="tech-badge">NLTK</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">📧 Smart Email Spam Detection</h1>
            <p class="hero-subtitle">Powered by NLP • TF-IDF • Logistic Regression</p>
            <p class="hero-description">
                Instantly analyze email content and identify Spam or Ham using Machine Learning.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📝 Email Content")
    
    # Create the text area with session state
    st.text_area(
        "",
        placeholder="Paste your email content here...",
        height=250,
        key="email_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        predict_button = st.button("🔮 Predict Email", use_container_width=True)
    with col2:
        st.button("🗑️ Clear", use_container_width=True, key="clear_btn", on_click=clear_input)
    with col3:
        sample_col1, sample_col2 = st.columns(2)
        with sample_col1:
            st.button("📧 Spam Sample", use_container_width=True, key="spam_sample_btn", on_click=set_spam_sample)
        with sample_col2:
            st.button("✅ Ham Sample", use_container_width=True, key="ham_sample_btn", on_click=set_ham_sample)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle prediction
    if predict_button:
        email_content = st.session_state.email_input
        if not email_content or email_content.strip() == "":
            st.warning("⚠️ Please enter an email content to analyze.")
        else:
            with st.spinner("🔍 Analyzing email content..."):
                # Simulate processing for better UX
                time.sleep(0.8)
                
                # Preprocess the text
                processed_text = preprocess_text(email_content)
                
                # Transform using TF-IDF vectorizer
                text_vectorized = vectorizer.transform([processed_text])
                
                # Make prediction
                prediction = model.predict(text_vectorized)[0]
                prediction_proba = model.predict_proba(text_vectorized)[0]
                
                # Get probabilities
                ham_probability = prediction_proba[0] * 100
                spam_probability = prediction_proba[1] * 100
                
                # Get email statistics
                char_count, word_count, sentence_count = get_email_stats(email_content)
                
                # Determine risk level
                risk_level, risk_color = get_risk_level(spam_probability / 100)
                
                # Result Section
                st.markdown("---")
                
                if prediction == 0:  # Ham
                    st.markdown("""
                        <div class="result-card result-ham">
                            <div class="result-icon">✅</div>
                            <div class="result-title result-title-ham">Not Spam</div>
                            <div class="result-description">This email appears to be legitimate and safe.</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:  # Spam
                    st.markdown("""
                        <div class="result-card result-spam">
                            <div class="result-icon">🚨</div>
                            <div class="result-title result-title-spam">Spam Detected</div>
                            <div class="result-description">This email appears to be suspicious and potentially harmful.</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Confidence and Probability
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📊 Confidence Score")
                    confidence = max(ham_probability, spam_probability)
                    
                    # Gauge circle
                    st.markdown(f"""
                        <div class="gauge-container">
                            <div class="gauge-circle" style="--progress: {confidence}%">
                                <div class="gauge-value">{confidence:.1f}%</div>
                                <div class="gauge-label">Confidence</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if confidence >= 90:
                        st.markdown('<p style="text-align: center; color: #22C55E; font-weight: 600;">High Confidence</p>', unsafe_allow_html=True)
                    elif confidence >= 70:
                        st.markdown('<p style="text-align: center; color: #F59E0B; font-weight: 600;">Medium Confidence</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p style="text-align: center; color: #EF4444; font-weight: 600;">Low Confidence</p>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 📈 Probability Analysis")
                    
                    # Spam Probability
                    st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; color: #94A3B8; font-size: 0.9rem;">
                                <span>🚨 Spam Probability</span>
                                <span>{spam_probability:.1f}%</span>
                            </div>
                            <div class="custom-progress">
                                <div class="custom-progress-bar" style="width: {spam_probability}%; background: linear-gradient(90deg, #EF4444, #DC2626);"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Ham Probability
                    st.markdown(f"""
                        <div style="margin-bottom: 0;">
                            <div style="display: flex; justify-content: space-between; color: #94A3B8; font-size: 0.9rem;">
                                <span>✅ Ham Probability</span>
                                <span>{ham_probability:.1f}%</span>
                            </div>
                            <div class="custom-progress">
                                <div class="custom-progress-bar" style="width: {ham_probability}%; background: linear-gradient(90deg, #22C55E, #16A34A);"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Email Statistics
                st.markdown("### 📊 Email Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-icon">📝</div>
                            <div class="metric-value">{char_count:,}</div>
                            <div class="metric-label">Characters</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-icon">📚</div>
                            <div class="metric-value">{word_count:,}</div>
                            <div class="metric-label">Words</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-icon">📖</div>
                            <div class="metric-value">{sentence_count:,}</div>
                            <div class="metric-label">Sentences</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Risk Level
                st.markdown("### ⚠️ Risk Assessment")
                risk_labels = {
                    "Safe": ("🟢 SAFE", "#22C55E"),
                    "Suspicious": ("🟡 SUSPICIOUS", "#F59E0B"),
                    "High Risk": ("🔴 HIGH RISK", "#EF4444")
                }
                risk_text, risk_color = risk_labels[risk_level]
                st.markdown(f"""
                    <div style="text-align: center;">
                        <span class="risk-badge" style="background: {risk_color}20; color: {risk_color}; border: 1px solid {risk_color}40;">
                            {risk_text}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
                
                # About Model Section
                with st.expander("🤖 About the Model"):
                    st.markdown("""
                        ### How It Works
                        
                        **Natural Language Processing (NLP)**
                        The system uses NLP techniques to understand and process human language, converting email text into a format that machine learning models can analyze.
                        
                        **TF-IDF Vectorization**
                        Term Frequency-Inverse Document Frequency converts text into numerical features by measuring the importance of words in the context of the document and the entire dataset.
                        
                        **Logistic Regression**
                        A powerful yet interpretable classification algorithm that learns patterns in the data to distinguish between spam and legitimate emails. It provides probability scores for informed decision-making.
                        
                        **Prediction Process**
                        1. Email text is cleaned and preprocessed
                        2. Features are extracted using TF-IDF
                        3. Logistic Regression model makes prediction
                        4. Confidence scores and probabilities are calculated
                        5. Results are presented with risk assessment
                    """)
    
    # Footer
    st.markdown("""
        <div class="footer">
            <div class="footer-tech">
                <span class="footer-tech-item">🐍 Python</span>
                <span class="footer-tech-item">🚀 Streamlit</span>
                <span class="footer-tech-item">🤖 Scikit-Learn</span>
                <span class="footer-tech-item">📚 NLTK</span>
            </div>
            <div class="footer-text" style="margin-top: 0.75rem;">
                Developed by <strong style="color: #60A5FA;">Yeshwanth</strong> • 2026
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
