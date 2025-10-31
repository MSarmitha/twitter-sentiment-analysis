import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import requests
import io

# Set page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)

download_nltk_data()

# Text cleaning function
def clean_text(text):
    stopwordlist = [
        'a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
        'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
        'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
        'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
        'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
        'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
        'into', 'is', 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma',
        'me', 'more', 'most', 'my', 'myself', 'needn', 'no', 'nor', 'now',
        'o', 'of', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves',
        'out', 'own', 're', 's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
        't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
        'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
        'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
        'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
        'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
        "youve", 'your', 'yours', 'yourself', 'yourselves'
    ]

    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    text = str(text).lower()
    text = re.sub(r'((www\.[\s]+)|(https?://[\s]+))', ' ', text)
    text = re.sub(r'@\S+', 'USER', text)
    text = re.sub(r'#(\S+)', r'\1', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = " ".join([word for word in text.split() if word not in stopwordlist])

    tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
    tokens = tokenizer.tokenize(text)

    if tokens:
        try:
            pos_tags = nltk.pos_tag(tokens)
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags]
            return " ".join(lemmatized_tokens)
        except:
            return " ".join(tokens)
    else:
        return ""

# Create sample training data
def create_sample_data():
    """Create sample training data for demonstration"""
    positive_tweets = [
        "I love this product! It's amazing!",
        "Great service and excellent support",
        "Wonderful experience, highly recommended",
        "This is fantastic and works perfectly",
        "Outstanding quality and great value",
        "I'm very happy with my purchase",
        "Excellent customer service, very helpful",
        "Perfect solution for my needs",
        "Amazing features and easy to use",
        "Best product I've ever bought",
        "Very satisfied with the results",
        "Impressive performance and quality",
        "Love the design and functionality",
        "Super fast delivery, thank you!",
        "Excellent value for money",
        "Perfect in every way",
        "Highly recommended to everyone",
        "Outstanding job well done",
        "Beautiful and functional design",
        "Couldn't be happier with this"
    ]
    
    negative_tweets = [
        "This is terrible and useless",
        "Worst experience ever",
        "Poor quality and bad service",
        "I hate this product completely",
        "Very disappointed with purchase",
        "Broken and doesn't work at all",
        "Waste of money, avoid this",
        "Terrible customer service",
        "Completely useless product",
        "Regret buying this item",
        "Poor quality materials used",
        "Doesn't work as advertised",
        "Very frustrating experience",
        "Awful performance and slow",
        "Bad design and poor functionality",
        "Not worth the money spent",
        "Disappointing results overall",
        "Poor construction quality",
        "Unreliable and breaks easily",
        "Worst purchase decision ever"
    ]
    
    data = {
        'text': positive_tweets + negative_tweets,
        'sentiment': [1] * len(positive_tweets) + [0] * len(negative_tweets)
    }
    
    return pd.DataFrame(data)

# Train model function
def train_model():
    """Train a logistic regression model with sample data"""
    st.info("🔄 Creating sample training data...")
    
    # Create sample data
    df = create_sample_data()
    
    # Clean the text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['sentiment']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    st.info("🔄 Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")
    
    # Save model and vectorizer
    with open("lr_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    
    with open("vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)
    
    return model, vectorizer, accuracy

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open("lr_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        with open("vectorizer.pkl", "rb") as vec_file:
            vectorizer = pickle.load(vec_file)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

# Main app
def main():
    # Header
    st.title("🐦 Twitter Sentiment Analyzer")
    st.markdown("Analyze if a tweet is **Positive** or **Negative** using Logistic Regression.")
    st.markdown("---")
    
    # Check if model exists, if not train one
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None or vectorizer is None:
        st.warning("⚠️ Model not found! Let's train a model first...")
        
        with st.expander("🚀 Train New Model", expanded=True):
            st.write("""
            **No pre-trained model found.** We'll create a sample dataset and train a model for you.
            This model will be saved and used for future predictions.
            """)
            
            if st.button("🎯 Train Model Now", type="primary"):
                with st.spinner("Training model with sample data..."):
                    model, vectorizer, accuracy = train_model()
                    st.balloons()
                    
                    # Show model info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model Type", "Logistic Regression")
                    with col2:
                        st.metric("Accuracy", f"{accuracy:.2f}")
                    with col3:
                        st.metric("Training Samples", "40")
                    
                    st.info("✅ Model is now ready! Refresh the page to start analyzing tweets.")
                    return
    
    # If model is loaded successfully, show the main interface
    st.success("✅ Model loaded successfully! Ready to analyze tweets.")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a Logistic Regression model trained on Twitter-like data "
        "to classify tweets as Positive or Negative sentiment."
    )
    
    st.sidebar.markdown("### How to use:")
    st.sidebar.markdown("""
    1. Enter a tweet in the text area
    2. Click 'Analyze Sentiment'
    3. View the results and confidence score
    """)
    
    st.sidebar.markdown("### Example tweets:")
    st.sidebar.markdown("- **Positive**: 'I love the new features! Amazing work!'")
    st.sidebar.markdown("- **Negative**: 'This is the worst service ever!'")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter your tweet below:")
        
        # Text input
        tweet_input = st.text_area(
            "Tweet Text:",
            placeholder="Type a tweet to analyze...\nExample: 'I love this product!' or 'This service is terrible.'",
            height=120,
            label_visibility="collapsed"
        )
        
        # Analyze button
        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if tweet_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    # Clean and predict
                    cleaned_tweet = clean_text(tweet_input)
                    vectorized_tweet = vectorizer.transform([cleaned_tweet])
                    prediction = model.predict(vectorized_tweet)[0]
                    probability = model.predict_proba(vectorized_tweet)[0]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    # Result cards
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        if prediction == 1:
                            st.success("## ✅ Positive Sentiment")
                            st.balloons()
                        else:
                            st.error("## ❌ Negative Sentiment")
                    
                    with result_col2:
                        confidence = probability[prediction] * 100
                        st.metric(
                            label="Confidence Level",
                            value=f"{confidence:.1f}%"
                        )
                    
                    # Probability breakdown
                    st.subheader("📈 Probability Breakdown")
                    prob_col1, prob_col2 = st.columns(2)
                    
                    with prob_col1:
                        negative_prob = probability[0] * 100
                        st.progress(negative_prob / 100, text=f"Negative: {negative_prob:.1f}%")
                    
                    with prob_col2:
                        positive_prob = probability[1] * 100
                        st.progress(positive_prob / 100, text=f"Positive: {positive_prob:.1f}%")
                    
                    # Show processed text
                    with st.expander("View processed text"):
                        st.write("**Original:**", tweet_input)
                        st.write("**Cleaned:**", cleaned_tweet)
                        
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    with col2:
        st.subheader("ℹ️ Model Information")
        
        st.metric("Model Type", "Logistic Regression")
        st.metric("Vectorizer", "TF-IDF")
        st.metric("Features", "5000")
        
        st.markdown("### Preprocessing Steps:")
        st.markdown("""
        - Lowercasing
        - URL/mention removal
        - Stopword removal
        - Lemmatization
        - POS tagging
        - Special character handling
        """)
        
        # Retrain button
        if st.button("🔄 Retrain Model"):
            with st.spinner("Training new model..."):
                model, vectorizer, accuracy = train_model()
                st.success("Model retrained successfully!")
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Twitter Sentiment Analysis App | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()
