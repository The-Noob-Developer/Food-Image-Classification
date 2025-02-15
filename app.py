# !pip install streamlit tensorflow numpy requests os
import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os
from tensorflow.keras.preprocessing import image
from PIL import Image

# Function to merge split model files
def merge_model_parts(output_file, parts_prefix):
    """Merges split .h5 model parts into a single file."""
    if os.path.exists(output_file):  # Skip merging if already exists
        return

    with open(output_file, 'wb') as outfile:
        part_num = 1
        while True:
            part_file = f"{parts_prefix}.part{part_num}"
            if not os.path.exists(part_file):
                break
            with open(part_file, 'rb') as infile:
                outfile.write(infile.read())
            part_num += 1

# Merge model parts before loading
model_path = "image_classifier_model.h5"
merge_model_parts(model_path, "image_classifier_model.h5")

# Load trained model
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = {
    0: "Pani Puri", 1: "Samosa", 2: "Vada Pav", 3: "Pav Bhaji", 4: "Dabeli",
    5: "Misal Pav", 6: "Aloo Tikki", 7: "Dahi Puri", 8: "Sev Puri", 9: "Bhel Puri",
    10: "Ragda Pattice", 11: "Kachori", 12: "Kathi Roll", 13: "Frankie Roll", 14: "Momos",
    15: "Bread Pakora", 16: "Dhokla", 17: "Pakoras", 18: "Chana Chaat", 19: "Masala Papad",
    20: "Dosa", 21: "Idli", 22: "Medu Vada", 23: "Uttapam", 24: "Pesarattu",
    25: "Chole Bhature", 26: "Rajma Chawal", 27: "Aloo Paratha", 28: "Chole Kulche", 29: "Tandoori Momos",
    30: "Hakka Noodles", 31: "Chilli Paneer", 32: "Schezwan Fried Rice", 33: "Manchurian", 34: "Spring Rolls",
    35: "Paneer Pizza", 36: "Tandoori Pizza", 37: "Cheese Burst Sandwich", 38: "Burger", 39: "French Fries",
    40: "Maggi", 41: "Grilled Sandwich", 42: "Cheese Toast", 43: "Veg Puff", 44: "Chicken Shawarma",
    45: "Egg Roll", 46: "Fish Fry", 47: "Kebab", 48: "Tandoori Chaap", 49: "Shawarma Wrap"
}

# USDA API Key (Replace with your actual key)
USDA_API_KEY = "P9QTMwyK5ppcAU0yKasgvkM61UTYthCG7GNFuTvQ"

def preprocess_image(img):
    """Preprocess image for model prediction"""
    IMG_SIZE = (128, 128)  
    img = img.resize(IMG_SIZE)  
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def predict_image(img):
    """Predict food item using trained model"""
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_name = class_labels.get(predicted_class, "Unknown")
    confidence = np.max(predictions) * 100
    return class_name, confidence

def fetch_nutrition_info(food_name):
    """Fetches nutrition details from USDA API"""
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_name}&api_key={USDA_API_KEY}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        foods = data.get("foods", [])
        if foods:
            food_item = foods[0]  # Take the first matching item
            nutrients = {nutrient["nutrientName"]: nutrient["value"] for nutrient in food_item.get("foodNutrients", [])}
            return {
                "Calories": nutrients.get("Energy", "N/A"),
                "Protein": nutrients.get("Protein", "N/A"),
                "Fats": nutrients.get("Total lipid (fat)", "N/A"),
                "Carbs": nutrients.get("Carbohydrate, by difference", "N/A")
            }
    return {"Calories": "N/A", "Protein": "N/A", "Fats": "N/A", "Carbs": "N/A"}

# Streamlit UI
st.title("Food Image Classifier with USDA Nutrition Data")
st.write("Upload an image to classify the food item and get nutritional details.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("Processing...")

    predicted_class, confidence = predict_image(img)
    st.write(f"**Prediction:** {predicted_class} (Confidence: {confidence:.2f}%)")

    # Fetch and display nutrition information from API
    nutrition_info = fetch_nutrition_info(predicted_class)
    st.write("### Nutritional Information (per 100g)")
    st.write(f"**Calories:** {nutrition_info['Calories']} kcal")
    st.write(f"**Protein:** {nutrition_info['Protein']} g")
    st.write(f"**Fats:** {nutrition_info['Fats']} g")
    st.write(f"**Carbs:** {nutrition_info['Carbs']} g")
