# Import all the libraries needed for the project
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the dataset and preprocess it
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb_model.h5')

# Function to decode review 
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

# Function to preprocess new review text
def preprocess_text(review, maxlen=5000):
    # Tokenize the review
    tokens = review.lower().split()
    # Convert words to their corresponding indices
    encoded_review = [word_index.get(word, 2) +3 for word in tokens]  # 2 is for unknown words
    # Pad the sequence
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review

# Predict sentiment of a new review
def predict_sentiment(review):
    processed_review = preprocess_text(review)
    prediction = model.predict(processed_review)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    return sentiment, prediction[0][0]

# Streamlit web app
st.title("Movie Review Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative).")
user_input = st.text_area("Movie Review:", "")
if st.button("Predict"):
    sentiment, confidence = predict_sentiment(user_input)
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"Confidence: {confidence:.4f}")
