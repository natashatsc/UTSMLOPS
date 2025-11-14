import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("my_model.h5")
    return model

model = load_model()

st.title("Image Classification App")
st.write("Upload an image and let the model predict its class.")

class_names = ["Cellulitis", "Impetigo", "Athlete-foot", "Ringworm", "Nail-Fungus", "Chickenpox", "Cutaneous-larva-migrans", "Shingles"]   # ganti sesuai dataset kamu

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))            
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, 0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("Prediction Result")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
