import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model_path = "C:/Users/PRAJWAL B/OneDrive/Documents/Hand_gestures/models.h5"
model = load_model(model_path)

# Map labels to gesture names (update according to your dataset)
gesture_dict = {
    0: "Palm",
    1: "L",
    2: "Index",
    3: "Fist Moved",
    4: "C",
    5: "OK",
    6: "Down",
    7: "Thumb",
    8: "Palm Moved",
    9: "Fist"
}

def predict_image(img_path):
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(64, 64))  # same size used in training
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

        # Predict
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions[0])
        confidence = predictions[0][predicted_label]

        # Map numeric label to gesture name
        gesture_name = gesture_dict.get(predicted_label, "Unknown")
        print(f"Predicted Gesture: {gesture_name} (Confidence: {confidence:.2f})")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    img_path = input("Enter the full path of the image to predict: ")
    predict_image(img_path)
