# Placeholder for a custom classifier
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model("vehicle_classification_model.h5")

def classify_vehicle(image):
    # Preprocess and predict
    image = preprocess(image)
    prediction = model.predict(image)
    return prediction
