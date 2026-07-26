"""
app.py
------
Streamlit web application for binary image classification:
    Healthy Pepper Leaf  vs  Bacterial Spot Pepper Leaf

Run locally:
    streamlit run app.py

Deploy:
    Push this repo to GitHub, then deploy on Streamlit Community Cloud
    (share.streamlit.io) by pointing it at app.py in this repository.
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Pepper Leaf Disease Classifier",
    page_icon="🌶️",
    layout="centered",
)

IMG_SIZE = (224, 224)
MODEL_PATH = "pepper_model.keras"
CLASS_NAMES = ["Bacterial Spot", "Healthy"]  # index 0, 1 (alphabetical folder order)


# ------------------------------------------------------------------
# Load model (cached so it only loads once per session)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr


def predict(model, image: Image.Image):
    arr = preprocess_image(image)
    prob = model.predict(arr, verbose=0)[0][0]  # sigmoid output, class 1 = Healthy
    if prob > 0.5:
        label = CLASS_NAMES[1]
        confidence = prob
    else:
        label = CLASS_NAMES[0]
        confidence = 1 - prob
    return label, float(confidence)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🌶️ Pepper Leaf Disease Classifier")
st.write(
    "Upload an image of a bell pepper leaf, and the model will predict "
    "whether it is **Healthy** or shows signs of **Bacterial Spot** disease."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a Convolutional Neural Network built with "
        "transfer learning (MobileNetV2) trained on the PlantVillage "
        "pepper leaf dataset to perform binary classification."
    )
    st.write("**Classes:**")
    st.write("- Healthy")
    st.write("- Bacterial Spot")

uploaded_file = st.file_uploader(
    "Choose a leaf image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify"):
        with st.spinner("Analyzing image..."):
            try:
                model = load_model()
                label, confidence = predict(model, image)

                st.subheader("Prediction Result")
                if label == "Healthy":
                    st.success(f"✅ {label}  ({confidence * 100:.2f}% confidence)")
                else:
                    st.error(f"⚠️ {label}  ({confidence * 100:.2f}% confidence)")

                st.progress(min(int(confidence * 100), 100))

            except Exception as e:
                st.error(f"Error loading model or making prediction: {e}")
                st.info(
                    "Make sure 'pepper_model.keras' is in the same directory "
                    "as this app and was trained using train_model.py."
                )
else:
    st.info("Please upload a pepper leaf image to get started.")

st.markdown("---")
st.caption("GET 324 - Cloud Computing and AI Model Deployment | Mini-Project")
