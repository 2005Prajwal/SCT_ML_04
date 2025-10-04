import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

gesture_dict = {
    0: "Palm",
    1: "Fist",
    2: "Index",
    3: "Fist Moved",
    4: "C",
    5: "OK",
    6: "Down",
    7: "Thumb",
    8: "Palm Moved",
    9: "L"
}

st.title("Hand Gesture Recognition App")
st.write("Upload an image of a hand gesture to predict.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    model = load_model("models.h5")
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    predicted_label = gesture_dict[predicted_class]

    st.write(f"Predicted Gesture: {predicted_label}")
