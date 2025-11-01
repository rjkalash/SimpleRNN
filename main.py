import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import time

# --- Page Configuration ---
# This is the first Streamlit command, and it sets up the page title, icon, and layout.
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Model and Data Loading ---
# We use caching to prevent reloading the model and data on every single interaction.
# This is a *critical* optimization for Streamlit apps.

@st.cache_resource  # Caches the loaded model
def load_keras_model():
    """Load the pre-trained Keras model."""
    try:
        model = load_model('simple_rnn_imdb_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure 'simple_rnn_imdb_model.h5' is in the same directory.")
        return None

@st.cache_data  # Caches the word index
def load_word_index():
    """Load the IMDB word index."""
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index

model = load_keras_model()
word_index, reverse_word_index = load_word_index()

# --- Core Functions (Unchanged) ---
# Your original functions for processing and prediction.

def decode_review(encoded_review):
    """Function to decode review."""
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

def preprocess_text(review, maxlen=5000):
    """Function to preprocess new review text."""
    # Tokenize the review
    tokens = review.lower().split()
    # Convert words to their corresponding indices
    # 2 is for unknown words, +3 is an offset due to IMDB's reserved indices
    encoded_review = [word_index.get(word, 2) + 3 for word in tokens]
    # Pad the sequence
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review

def predict_sentiment(review):
    """Predict sentiment of a new review."""
    if not model:
        return "Error", 0.0
        
    processed_review = preprocess_text(review)
    
    # Add a small delay to simulate processing (makes the spinner visible)
    time.sleep(0.5) 
    
    prediction = model.predict(processed_review)
    raw_score = prediction[0][0]
    sentiment = "Positive" if raw_score >= 0.5 else "Negative"
    
    # Calculate a more intuitive "confidence" score
    if sentiment == "Positive":
        confidence = raw_score
    else:
        confidence = 1 - raw_score
        
    return sentiment, confidence, raw_score

# --- Streamlit Web App UI ---

# Title and Subheader
st.title("🎬 Movie Review Sentiment Analyzer")
st.subheader("How does the model feel about this review?")
st.caption("This app uses a Simple RNN model trained on the IMDB dataset to predict sentiment.")

# We use a container with a border for a cleaner look
with st.container(border=True):
    # Text area for user input
    user_input = st.text_area(
        "Enter a movie review:", 
        "", 
        height=200,
        placeholder="e.g., 'This movie was absolutely fantastic! The acting was superb and the plot was gripping.' "
    )

    # Predict button
    if st.button("Analyze Sentiment", use_container_width=True, type="primary"):
        if user_input.strip():
            # Show a spinner while processing
            with st.spinner("Analyzing..."):
                sentiment, confidence, raw_score = predict_sentiment(user_input)

            if sentiment == "Error":
                st.error("Model could not be loaded. Please check the logs.")
            else:
                st.write("### Analysis Result")
                
                # Use columns for a balanced layout
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display the final sentiment with a colored box
                    if sentiment == "Positive":
                        st.success(f"**Sentiment: Positive 😃**")
                    else:
                        st.error(f"**Sentiment: Negative 😠**")
                
                with col2:
                    # Display the confidence as a metric
                    st.metric(label="Prediction Confidence", value=f"{confidence:.2%}")

                # Display the raw score in a progress bar for visual context
                st.progress(float(raw_score))
                st.caption(f"Raw Model Output: {raw_score:.4f} (Closer to 1 is Positive, Closer to 0 is Negative)")

        else:
            st.warning("Please enter a review to analyze.")

# "How it works" expander
with st.expander("Learn more about this app"):
    st.write("""
        This tool works by:
        1.  **Tokenizing:** Breaking your review down into individual words.
        2.  **Integer Encoding:** Converting each word into a unique number based on the IMDB dataset's vocabulary.
        3.  **Padding:** Ensuring the review has a fixed length (5000 words) by adding padding, which the RNN model expects.
        4.  **Prediction:** Feeding this sequence of numbers into the pre-trained Simple Recurrent Neural Network (RNN) model.
        5.  **Output:** The model outputs a single number between 0 (Negative) and 1 (Positive), which we translate into the final sentiment.
    """)
    st.write("The model was trained on the IMDB movie reviews dataset, which contains 50,000 reviews labeled as positive or negative. It uses a Simple RNN architecture to capture the sequential nature of text data.")
    st.write("Feel free to test it with different reviews and see how well it predicts the sentiment!")
    