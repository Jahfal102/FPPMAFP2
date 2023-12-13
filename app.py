# app.py

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained machine learning model
model_path = 'fixedmodel.h5'

try:
    model = load_model(model_path)
    st.write("Model loaded successfully!")
except Exception as e:
    st.write(f"Error loading the model: {e}")
    st.write(f"Make sure the file path '{model_path}' is correct.")

# Function to preprocess the image and make predictions
# Function to preprocess the image and make predictions
def predict(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Assuming your model expects 224x224 images

        # Convert RGB image to grayscale
        img_gray = img.convert("L")

        img_array = np.array(img_gray)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the pixel values between 0 and 1

        # Reshape to match the expected input shape (None, 224, 224, 1)
        img_array = np.expand_dims(img_array, axis=-1)

        predictions = model.predict(img_array)
        return predictions
    except Exception as e:
        st.write(f"Error making predictions: {e}")
        return None

# Streamlit app
def main():
    st.title('X-ray Image Classification App')

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
            
            # Assuming your model outputs class indices (0, 1, 2)
            predicted_class_index = np.argmax(predictions)
            
            # Dictionary of classes
            class_mapping = {0: 'Normal', 1: 'Viral Pneumonia', 2: 'Covid'}

            predicted_class = class_mapping[predicted_class_index]
            
            st.write(f"Predicted Class: {predicted_class}")
            st.write(f"Probability: {predictions[0][predicted_class_index]:.2%}")

if __name__ == "__main__":
    main()
