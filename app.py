import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model("C:/Users/PRAJWAL B/OneDrive/Documents/Hand_gestures/models.h5")

# Gesture dictionary
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

st.title("Hand Gesture Recognition App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write(f"Predicted Gesture: **{gesture_dict[predicted_class]}** (Confidence: {confidence:.2f})")
