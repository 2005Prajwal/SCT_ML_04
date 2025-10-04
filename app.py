import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model (models.h5 should be in the same folder as app.py)
model = load_model("models.h5")

# Map model output to gesture names
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

# Streamlit App
st.title("Hand Gesture Recognition App")
st.write("Upload an image of a hand to predict the gesture.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess the image for prediction
    img = img.resize((64, 64))  # same size as training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    img_array = img_array / 255.0  # normalize

    # Predict
    pred_probs = model.predict(img_array)
    pred_class = np.argmax(pred_probs)
    confidence = pred_probs[0][pred_class]

    # Display result
    st.success(f"Predicted Gesture: {gesture_dict[pred_class]} (Confidence: {confidence:.2f})")
