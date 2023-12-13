# app.py

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained machine learning model
model_path = 'fixedmodel.h5'

try:
    model = load_model(model_path)
    st.write("Model loaded successfully!")
except Exception as e:
    st.write(f"Error loading the model: {e}")
    st.write(f"Make sure the file path '{model_path}' is correct.")

# Function to preprocess the image and make predictions
def predict(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the pixel values between 0 and 1

        # Convert RGB to grayscale
        img_array = np.mean(img_array, axis=-1, keepdims=True)

        predictions = model.predict(img_array)
        return predictions
    except Exception as e:
        st.write(f"Error making predictions: {e}")
        return None

# Streamlit app
def main():
    st.title('COVID-19 Pneumonia Detection App')

    # Upload an image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Image.", use_column_width=True)

        # Make predictions on the uploaded image
        predictions = predict(uploaded_file)

        if predictions is not None:
            st.subheader("Predictions:")
            # Assuming your model outputs probabilities for each class
            st.write(f"COVID-19 Probability: {predictions[0][0]:.2%}")
            st.write(f"Pneumonia Probability: {predictions[0][1]:.2%}")
            st.write(f"Normal Probability: {
