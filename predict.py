# predict.py
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

# Load the model once when the module is imported
model = load_model("model_inception.h5")  # Update with your actual model path


# Define class labels (Update based on `training_set.class_indices`)
class_labels = {'Aloo_Tikki': 0, 'Bhel_Puri': 1, 'Bread_Pakora': 2, 'Chana_Chaat': 3, 'Dabeli': 4, 'Dahi_Puri': 5, 'Dhokla': 6, 'Frankie_Roll': 7, 'Kachori': 8, 'Kathi_Roll': 9, 'Masala_Papad': 10, 'Misal_Pav': 11, 'Momos': 12, 'Pakoras': 13, 'Pani_Puri': 14, 'Pav_Bhaji': 15, 'Ragda_Pattice': 16, 'Samosa': 17, 'Sev_Puri': 18, 'Vada_Pav': 19}  # Modify accordingly

def predict_image(img_path):
    """
    Predict the class of a given image.

    Args:
        img_path (str): Path to the image.

    Returns:
        str: Predicted class label.
    """
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
        img_array = image.img_to_array(img) / 255.0  # Convert to array and normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        y_pred = model.predict(img_array)

        # Get class with highest probability
        predicted_class = np.argmax(y_pred, axis=1)[0]

        return class_labels[predicted_class]  # Return predicted class label

    except Exception as e:
        return f"Error: {e}"
