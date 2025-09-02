import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown

# ------------------------------
# Download model if not exists
# ------------------------------
MODEL_ID = "1m0MxHbAvkfbWyaU0s0gQNcFakJWt4vQQ"  # your Google Drive file ID
MODEL_PATH = "Cat_and_dog_model.h5"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🐾 Friendly Cat vs Dog Classifier 🐾")
st.write("Upload an image of a cat or dog, and I’ll tell you what I think 🐱🐶")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model:
    # Convert image to RGB to avoid 4-channel issues
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess image
    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)[0][0]

    # Determine label and confidence
    if prediction >= 0.5:
        label = "Dog 🐶"
        percent = prediction * 100
    else:
        label = "Cat 🐱"
        percent = (1 - prediction) * 100

    st.success(f"I’m {percent:.0f}% sure this is a {label}!")





