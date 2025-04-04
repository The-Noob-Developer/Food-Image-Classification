import streamlit as st
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from predict import predict_image, class_labels  # Import prediction function and labels

# Function to merge the model parts
def merge_model_parts(output_file="model_inception.h5"):
    with open(output_file, 'wb') as output:
        part_number = 1
        while True:
            part_filename = f"model_inception.h5.part{part_number}"
            if not os.path.exists(part_filename):
                break
            with open(part_filename, 'rb') as part_file:
                output.write(part_file.read())
            part_number += 1

# Check if full model exists, otherwise merge
if not os.path.exists("model_inception.h5"):
    merge_model_parts()

# Load the model
model = load_model("model_inception.h5")  # Ensure the model file is in the correct path

st.title("Food Image Classifier 🍔🥗")
st.write("Upload an image of a food item, and the model will predict its class with probability.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Save the uploaded file temporarily
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and preprocess the image
    img = image.load_img("temp_image.png", target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make a prediction
    y_pred = model.predict(img_array)
    predicted_class = np.argmax(y_pred, axis=1)[0]
    predicted_label = list(class_labels.keys())[list(class_labels.values()).index(predicted_class)]
    probability = np.max(y_pred) * 100  # Get probability in percentage
    
    # Display result
    st.success(f"Predicted Class: {predicted_label}")
    st.info(f"Confidence: {probability:.2f}%")
