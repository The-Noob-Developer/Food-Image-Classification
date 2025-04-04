import streamlit as st
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Define the model filename
model_filename = "model_inception.h5"

# Function to merge the model parts
def merge_model_parts(output_file):
    with open(output_file, 'wb') as output:
        part_number = 1
        while True:
            part_filename = f"{output_file}.part{part_number}"
            if not os.path.exists(part_filename):
                break
            with open(part_filename, 'rb') as part_file:
                output.write(part_file.read())
            part_number += 1

# Check if full model exists, otherwise merge
if not os.path.exists(model_filename):
    merge_model_parts(model_filename)

# Load the model
model = load_model(model_filename)  # Ensure the model file is in the correct path

# Define class labels (Update based on `training_set.class_indices`)
class_labels = {'Aloo_Tikki': 0, 'Bhel_Puri': 1, 'Bread_Pakora': 2, 'Chana_Chaat': 3, 'Dabeli': 4, 'Dahi_Puri': 5, 'Dhokla': 6, 'Frankie_Roll': 7, 'Kachori': 8, 'Kathi_Roll': 9, 'Masala_Papad': 10, 'Misal_Pav': 11, 'Momos': 12, 'Pakoras': 13, 'Pani_Puri': 14, 'Pav_Bhaji': 15, 'Ragda_Pattice': 16, 'Samosa': 17, 'Sev_Puri': 18, 'Vada_Pav': 19}  # Modify accordingly

# Reverse the dictionary to map index to class name
class_labels_inv = {v: k for k, v in class_labels.items()}

def predict_image(img_path):
    """
    Predict the class of a given image.
    
    Args:
        img_path (str): Path to the image.
    
    Returns:
        str: Predicted class label with confidence score.
    """
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        y_pred = model.predict(img_array)
        predicted_class = np.argmax(y_pred, axis=1)[0]
        confidence = np.max(y_pred) * 100  # Get probability in percentage
        
        return class_labels_inv[predicted_class], confidence
    
    except Exception as e:
        return f"Error: {e}", None

# Streamlit App
st.title("Food Image Classifier 🍔🥗")
st.write("Upload an image of a food item, and the model will predict its class with probability.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Save the uploaded file temporarily
    temp_filename = "temp_image.png"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Predict image class
    predicted_label, probability = predict_image(temp_filename)
    
    if probability is not None:
        st.success(f"Predicted Class: {predicted_label}")
        st.info(f"Confidence: {probability:.2f}%")
    else:
        st.error("Error in prediction.")
