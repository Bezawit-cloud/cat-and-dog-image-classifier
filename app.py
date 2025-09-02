
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# ------------------------------
# Load model
# ------------------------------
MODEL_PATH = "Cat_and_dog_model.h5"  # Make sure this matches your file name exactly

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found! Make sure it's in the same folder as app.py.")
        return None
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




