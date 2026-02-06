import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import time

IMG_SIZE = (224, 224)
MODEL_PATH = "final_model.h5"


@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


st.set_page_config(
    page_title="Person Detector",
    page_icon="🧑",
    layout="centered",
)


st.markdown(
    """
    <style>
        .main {
            padding-top: 2rem;
        }
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #ffffff 55%, #f1f5f9 100%);
            color: #0f172a;
        }
        h1, h2, h3 {
            color: #0f172a !important;
        }
        /* làm nhẹ header của sidebar */
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid rgba(15, 23, 42, 0.08);
        }
        .stButton>button {
            background: linear-gradient(90deg, #22c55e, #16a34a);
            color: white;
            border-radius: 999px;
            border: none;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #16a34a, #22c55e);
        }
        .prob-box {
            padding: 1rem 1.25rem;
            border-radius: 0.75rem;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid rgba(15, 23, 42, 0.10);
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.06);
        }
        .label-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.3rem 0.8rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }
        .label-person {
            background: rgba(34,197,94,0.12);
            color: #14532d;
            border: 1px solid rgba(34,197,94,0.45);
        }
        .label-nonperson {
            background: rgba(248,113,113,0.12);
            color: #7f1d1d;
            border: 1px solid rgba(248,113,113,0.45);
        }
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("🧑 _:blue[Person Detector]_")
st.markdown("**TRẦN HẢI NAM - 223332840**")


model = load_model()

uploaded_file = st.file_uploader(
    "Chọn ảnh",
    type=["jpg", "jpeg", "png"],
)

col_img, col_result = st.columns([3, 2])

with col_img:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image)


with col_result:
    if uploaded_file is not None:
        predict_btn = st.button("Dự đoán")

        if predict_btn:
            with st.spinner("Đang dự đoán..."):
                input_data = preprocess_image(image)
                prob = float(model.predict(input_data)[0][0])

                is_person = prob >= 0.5
                label = "PERSON" if is_person else "NON-PERSON"
                css_label_class = "label-person" if is_person else "label-nonperson"
                confidence = prob if is_person else 1.0 - prob

            st.markdown(
                f"""
                <div class="prob-box">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                        <span style="color:#64748b;">Kết quả phân loại</span>
                        <span class="label-pill {css_label_class}">{label}</span>
                    </div>
                    <div style="margin-top:0.25rem;font-size:0.9rem;color:#475569;">
                        Độ tin cậy: {confidence * 100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
