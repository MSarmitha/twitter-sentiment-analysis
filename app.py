import streamlit as st
import pandas as pd
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------------------------------
# 🧹 TEXT CLEANING FUNCTION
# ------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------
# 🧠 TRAIN MODEL FUNCTION (runs once if no saved model)
# ------------------------------------------------------
def train_model():
    st.info("🔄 Training new sentiment model (since no saved model found)...")

    # Example training dataset (you can replace this with your own)
    data = {
        "text": [
            "I love this app!",
            "This is terrible.",
            "I am so happy today!",
            "Worst experience ever.",
            "I like the new design.",
            "I hate bugs in this app.",
            "Amazing support team!",
            "This is bad and frustrating.",
            "Great update, really enjoyed it!",
            "The product is awful."
        ],
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=Positive, 0=Negative
    }
    df = pd.DataFrame(data)

    # Clean text
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df["cleaned_text"])
    y = df["label"]

    # Train model
    model = LogisticRegression()
    model.fit(X, y)

    # Evaluate
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    st.success(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")

    # Save model & vectorizer (optional)
    with open("lr_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return model, vectorizer


# ------------------------------------------------------
# 📦 LOAD MODEL FUNCTION
# ------------------------------------------------------
def load_model():
    if os.path.exists("lr_model.pkl") and os.path.exists("vectorizer.pkl"):
        with open("lr_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    else:
        return train_model()


# ------------------------------------------------------
# 🚀 MAIN STREAMLIT APP
# ------------------------------------------------------
def main():
    st.set_page_config(page_title="🐦 Twitter Sentiment Analyzer", layout="wide")

    st.title("🐦 Twitter Sentiment Analyzer")
    st.write("Analyze if a tweet is **Positive** or **Negative** using a Logistic Regression model.")
    st.markdown("---")

    # Load or train model
    model, vectorizer = load_model()

    st.sidebar.header("ℹ️ About the App")
    st.sidebar.write("""
    This app uses **TF-IDF** and **Logistic Regression**
    to classify tweets as Positive or Negative.
    """)

    # Input area
    tweet_input = st.text_area("✍️ Enter a tweet to analyze:", height=120)

    if st.button("🔍 Analyze Sentiment", use_container_width=True):
        if tweet_input.strip():
            cleaned_tweet = clean_text(tweet_input)
            vectorized_tweet = vectorizer.transform([cleaned_tweet])
            prediction = model.predict(vectorized_tweet)[0]
            probability = model.predict_proba(vectorized_tweet)[0]

            st.markdown("---")
            st.subheader("📊 Analysis Result")

            if prediction == 1:
                st.success("✅ Positive Sentiment")
                st.balloons()
            else:
                st.error("❌ Negative Sentiment")

            st.metric("Confidence Level", f"{max(probability)*100:.2f}%")

            st.write("**Processed Text:**", cleaned_tweet)

        else:
            st.warning("⚠️ Please enter a tweet to analyze.")

    # Footer
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")


if __name__ == "__main__":
    main()
