import streamlit as st
import pickle
import numpy as np

# Load the trained model and vectorizer
try:
    with open('/content/best_distilbert_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('/content/best_distilbert_tokenizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

st.title("Emotion Detection")
user_input = st.text_area("Enter text to analyze:")

if st.button("Predict Emotion"):
    if user_input and model and vectorizer:
        try:
            text_vectorized = vectorizer.transform([user_input])
            prediction_probabilities = model.predict_proba(text_vectorized)[0]
            predicted_class_index = np.argmax(prediction_probabilities)
            predicted_class = model.classes_[predicted_class_index]
            confidence = prediction_probabilities[predicted_class_index]

            st.write(f"Predicted Emotion: **{predicted_class}**")
            st.write(f"Confidence: **{confidence:.2f}**")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    elif not user_input:
        st.warning("Please enter some text.")
    else:
        st.error("Model or vectorizer not loaded.")
