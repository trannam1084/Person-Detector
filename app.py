import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

IMG_SIZE = (224, 224)
MODEL_PATH = "final_model.h5"

st.set_page_config(
    page_title="Person Detector",
    layout="centered",
)

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

def show_result(prob):
    is_person = prob >= 0.5
    confidence = prob if is_person else 1.0 - prob

    if is_person:
        st.success("**PERSON**")
    else:
        st.warning("**NON-PERSON**")

    st.metric("Độ tin cậy", f"{confidence * 100:.1f}%")

st.title("_:blue[Person Detector]_")
st.markdown("**TRẦN HẢI NAM - 223332840**")
st.markdown("---")

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None

st.markdown("##### Chọn ảnh")
uploaded_file = st.file_uploader(
    "Chọn ảnh...",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

col_img, col_result = st.columns([3, 2])

with col_img:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image)

        if st.button("Dự đoán", type="primary", use_container_width=True):
            with st.spinner("Đang dự đoán..."):
                img_array = preprocess_image(image)
                prob = float(model.predict(img_array, verbose=0)[0][0])
                st.session_state.upload_result = prob

with col_result:
    st.markdown("#### Kết quả dự đoán")
    with st.container(border=True):
        if st.session_state.upload_result is not None:
            show_result(st.session_state.upload_result)
