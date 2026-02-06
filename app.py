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

# ==== Custom CSS cho giao diện sạch và đẹp hơn ====
st.markdown(
    """
    <style>
        .main {
            padding-top: 2rem;
        }
        .stApp {
            background: radial-gradient(circle at top left, #1f2933 0, #111827 45%, #020617 100%);
            color: #e5e7eb;
        }
        h1, h2, h3 {
            color: #f9fafb !important;
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
            background: rgba(15,23,42,0.9);
            border: 1px solid rgba(148,163,184,0.3);
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
            background: rgba(34,197,94,0.1);
            color: #bbf7d0;
            border: 1px solid rgba(34,197,94,0.6);
        }
        .label-nonperson {
            background: rgba(248,113,113,0.1);
            color: #fecaca;
            border: 1px solid rgba(248,113,113,0.6);
        }
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==== Sidebar thông tin ====
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình")
    threshold = st.slider(
        "Ngưỡng phân loại (threshold)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Nếu xác suất ≥ threshold → PERSON, ngược lại → NON-PERSON.",
    )

    st.markdown("---")
    st.markdown("### 👤 Thông tin")
    st.markdown("**TRẦN HẢI NAM - 223332840**")
    st.caption("Bài tập: Nhận diện ảnh có người / không có người bằng TensorFlow/Keras & EfficientNetB0.")

    st.markdown("---")
    st.caption("Model: EfficientNetB0 (fine-tuned) · Input 224x224 · Binary classification.")

# ==== Tiêu đề chính ====
st.markdown("## 🧑‍🤝‍🧑 Person Detector")
st.write(
    "Tải lên một ảnh bất kỳ (jpg / png). Ứng dụng sẽ dự đoán **ảnh có chứa người hay không** "
    "dựa trên mô hình học sâu đã được huấn luyện trên COCO (person vs non-person)."
)

model = load_model()

uploaded_file = st.file_uploader(
    "Chọn ảnh cần kiểm tra",
    type=["jpg", "jpeg", "png"],
    help="Kích thước và tỉ lệ ảnh sẽ được tự động resize về 224x224.",
)

col_img, col_result = st.columns([3, 2])

with col_img:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
    else:
        st.markdown("#### 📷 Hướng dẫn")
        st.write(
            "- Chọn một ảnh chụp người, đường phố, cảnh vật, v.v.\n"
            "- Hệ thống sẽ trả về nhãn **PERSON** hoặc **NON-PERSON** cùng xác suất."
        )

with col_result:
    if uploaded_file is not None:
        predict_btn = st.button("🚀 Predict")

        if predict_btn:
            with st.spinner("Đang dự đoán..."):
                start_time = time.time()
                input_data = preprocess_image(image)
                prob = float(model.predict(input_data)[0][0])
                infer_time = (time.time() - start_time) * 1000  # ms

                is_person = prob >= threshold
                label = "PERSON" if is_person else "NON-PERSON"
                css_label_class = "label-person" if is_person else "label-nonperson"

            st.markdown(
                f"""
                <div class="prob-box">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                        <span style="color:#9ca3af;">Kết quả phân loại</span>
                        <span class="label-pill {css_label_class}">{label}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### 🔢 Xác suất")
            st.progress(prob if prob <= 1 else 1.0, text=f"p(person) = {prob:.4f}")

            st.markdown(
                f"- **Ngưỡng hiện tại**: `{threshold:.2f}`  "
                f"- **p(person)**: `{prob:.4f}`  \n"
                f"- **Thời gian suy luận**: `{infer_time:.1f} ms`"
            )
    else:
        st.info("👆 Hãy tải một ảnh lên để thực hiện dự đoán.")

st.markdown("---")
st.markdown(
    "*Ứng dụng xây dựng bằng **Streamlit** và **TensorFlow/Keras (EfficientNetB0)**. "
    "Model được huấn luyện trên tập dữ liệu COCO (person vs non-person).*"
)
