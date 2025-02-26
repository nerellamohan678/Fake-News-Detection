import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import GPT2Tokenizer, TFGPT2Model
import streamlit as st

# Load the saved CNN model
cnn_model = load_model('results (2)/cnn_gpt2_classification_model.h5')

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = TFGPT2Model.from_pretrained('gpt2')

# Add padding token if necessary (GPT-2 does not have one by default)
tokenizer.pad_token = tokenizer.eos_token

# Function to extract GPT-2 embeddings for a single input
def get_gpt2_embedding(text, model, tokenizer, max_length=512):
    inputs = tokenizer(text, return_tensors='tf', padding='max_length', truncation=True, max_length=max_length)
    outputs = model(inputs['input_ids'])
    # Take the mean of the last hidden state as the embedding representation
    embedding = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()[0]
    return embedding

# Streamlit app
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

# CSS for custom styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        text-align: center;
        color: #333;
        font-size: 2.5em;
    }
    .description {
        font-size: 1.2em;
        color: #555;
        text-align: center;
    }
    .text-area {
        margin: 20px 0;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #777;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Fake News Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>Enter the news article text below to check its authenticity.</p>", unsafe_allow_html=True)

# Input text area for the user
input_text = st.text_area("News Article Text:", height=200, placeholder="Type or paste the news article here...", key="input_text", label_visibility="collapsed")

# Predict button
if st.button("Predict"):
    if input_text:
        # Get GPT-2 embedding for the input text
        input_embedding = get_gpt2_embedding(input_text, gpt2_model, tokenizer)

        # Reshape to fit CNN input shape
        input_embedding_cnn = input_embedding.reshape((1, input_embedding.shape[0], 1))

        # Make prediction
        prediction = (cnn_model.predict(input_embedding_cnn) > 0.5).astype("int32")

        # Interpret the prediction
        if prediction[0][0] == 1:
            st.success("✅ The input text is predicted to be **Real News**.")
        else:
            st.error("❌ The input text is predicted to be **Fake News**.")
    else:
        st.warning("⚠️ Please enter some text before predicting.")

# Footer
st.markdown("<div class='footer'>Made with ❤️ by VIT</div>", unsafe_allow_html=True)
