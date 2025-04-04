import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from predict import predict_image, class_labels  # Import prediction function and labels

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
