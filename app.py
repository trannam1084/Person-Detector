import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

IMG_SIZE = (224, 224)
MODEL_PATH = "final_model.h5"

@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.set_page_config(page_title="PERSON DETECTION", page_icon="🧑🏿")

st.title("TRẦN HẢI NAM - 223332840")
st.write("Upload a image to check model.")

uploaded_file = st.file_uploader("Image Select", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            input_data = preprocess_image(image)
            prob = model.predict(input_data)[0][0]

            label = "PERSON" if prob >= 0.5 else "NON-PERSON"

        st.markdown(f"**Answer:** {label}")
        st.markdown(f"**Prob:** `{prob:.4f}`")
